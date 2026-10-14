//! Aquantia Atlantic 2 (AQC113 series) Ethernet controller driver.
//!
//! Supported devices: AQC113CS (0x04C0), AQC113 (0x04D0–0x04D3), AQC107 (0xD107–0xD109).
//!
//! Register layout references:
//! - Linux kernel `drivers/net/ethernet/aquantia/atlantic/hw_atl/hw_atl_llh_internal.h`
//! - Linux kernel `drivers/net/ethernet/aquantia/atlantic/hw_atl2/hw_atl2_llh_internal.h`
//!
//! Path classification:
//! - Non-RT: `attach`, `up`/`down`, `init`, interrupt setup.
//! - RT (network fast path): `send`, `recv`, `interrupt`, `poll`,
//!   `poll_in_service`.  See the WCET contract on each.
//!
//! The driver uses a single TX/RX ring pair (queue 0).

// Hardening gates for this driver (see the rust-hardening policy):
// panics, silent overflow, lossy casts and swallowed results are compile errors.
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented,
    clippy::indexing_slicing,
    clippy::string_slice,
    clippy::unwrap_in_result,
    clippy::arithmetic_side_effects,
    clippy::let_underscore_must_use,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::allow_attributes_without_reason,
    unsafe_op_in_unsafe_fn
)]

use crate::pcie::{
    mmio::{Mmio, MmioErr},
    pcie_id::AQUANTIA_VENDOR_ID,
    PCIeDevice, PCIeDeviceErr, PCIeInfo,
};
use alloc::{borrow::Cow, boxed::Box, format, sync::Arc, vec::Vec};
use awkernel_async_lib_verified::ringq::RingQ;
use awkernel_lib::{
    addr::Addr,
    dma_pool::DMAPool,
    interrupt::IRQ,
    net::{
        multicast::MulticastAddrs,
        net_device::{
            EtherFrameBuf, EtherFrameRef, LinkStatus, NetCapabilities, NetDevError, NetDevice,
            NetFlags,
        },
    },
    paging::PAGESIZE,
    sync::{
        mutex::{MCSNode, Mutex},
        rwlock::RwLock,
    },
};
use core::{
    ptr::{read_volatile, write_volatile},
    sync::atomic::{fence, Ordering},
};

mod atlantic_regs;

use atlantic_regs::*;

// ── Device list ──────────────────────────────────────────────────────────────

const DEVICE_NAME: &str = "Aquantia AQC Atlantic Ethernet Controller";
const DEVICE_SHORT_NAME: &str = "atl";

/// Supported (vendor_id, device_id) pairs.
pub const AQUANTIA_DEVICES: &[(u16, u16)] = &[
    // Atlantic 2 (hw_atl2) — AQC113 family
    (AQUANTIA_VENDOR_ID, 0x04C0), // AQC113CS
    (AQUANTIA_VENDOR_ID, 0x04D0), // AQC113
    (AQUANTIA_VENDOR_ID, 0x04D1), // AQC113
    (AQUANTIA_VENDOR_ID, 0x04D2), // AQC113
    (AQUANTIA_VENDOR_ID, 0x04D3), // AQC113
    (AQUANTIA_VENDOR_ID, 0x04D4), // AQC113
    (AQUANTIA_VENDOR_ID, 0x0005), // AQC100
    // Atlantic 1 (hw_atl_b0) — AQC107/AQC108 family
    (AQUANTIA_VENDOR_ID, 0x0001), // AQC100
    (AQUANTIA_VENDOR_ID, 0xD100), // AQC100 (OEM)
    (AQUANTIA_VENDOR_ID, 0xD107), // AQC107 (10GbE)
    (AQUANTIA_VENDOR_ID, 0xD108), // AQC108 (5GbE)
    (AQUANTIA_VENDOR_ID, 0xD109), // AQC109 (2.5GbE)
    (AQUANTIA_VENDOR_ID, 0x87B1), // AQC107 (OEM/rebranded)
];

// ── Constants ────────────────────────────────────────────────────────────────

/// Number of TX / RX descriptors per ring (power of 2, ≥ 32).
const NUM_RING_DESC: usize = 256;
/// Ring indices wrap modulo `NUM_RING_DESC`.
const RING_MASK: usize = NUM_RING_DESC - 1;
/// Maximum size of each DMA packet buffer (bytes).
const PKT_BUF_SIZE: usize = 2048;
/// Ethernet FCS that the hardware includes in the RX packet length.
const FCS_LEN: usize = 4;
/// Software RX queue depth (packets buffered before the network stack drains them).
const SW_RX_QUEUE_SIZE: usize = 64;
/// Number of milliseconds between periodic tick calls (link check).
const TICK_MSEC: u64 = 200;
/// Firmware ready poll maximum iterations (1 ms apart).
const FIRMWARE_POLL_ITERS: u32 = 5000;
/// The only queue this driver exposes.
const QUEUE_ID: usize = 0;
/// All DMA allocations go to NUMA node 0.
const NUMA_NODE: usize = 0;

// ── DMA ring types ───────────────────────────────────────────────────────────

/// Raw 16-byte descriptor slot shared by both the TX and RX rings.
type Desc = [u64; 2];
type DescRing = [Desc; NUM_RING_DESC];
/// One packet buffer per descriptor.
type PktBuf = [[u8; PKT_BUF_SIZE]; NUM_RING_DESC];

const _: () = assert!(NUM_RING_DESC.is_power_of_two());
// Both rings use exactly one page: 256 × 16 = 4096 bytes.
const _: () = assert!(
    core::mem::size_of::<DescRing>() == PAGESIZE,
    "DescRing must fit in one page"
);
// Packet buffers: 256 × 2048 = 524 288 bytes = 128 pages.
const _: () = assert!(core::mem::size_of::<PktBuf>().is_multiple_of(PAGESIZE));
const DESC_RING_PAGES: usize = core::mem::size_of::<DescRing>() / PAGESIZE;
const PKT_BUF_PAGES: usize = core::mem::size_of::<PktBuf>() / PAGESIZE;

/// Next index in a ring (modular ring arithmetic).
fn ring_next(i: usize) -> usize {
    i.wrapping_add(1) & RING_MASK
}

// ── Driver error type ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub enum AtlDriverErr {
    NoBar0,
    Mmio(MmioErr),
    DmaAlloc,
    FirmwareTimeout,
    Overflow,
    TxRingFull,
    FrameTooLarge,
    InvalidQueue,
}

impl core::fmt::Display for AtlDriverErr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl core::error::Error for AtlDriverErr {}

impl From<MmioErr> for AtlDriverErr {
    fn from(e: MmioErr) -> Self {
        AtlDriverErr::Mmio(e)
    }
}

impl From<AtlDriverErr> for PCIeDeviceErr {
    fn from(e: AtlDriverErr) -> Self {
        log::error!("atl: {e}");
        PCIeDeviceErr::InitFailure
    }
}

/// Fast-path conversion: no logging (called per packet).
impl From<AtlDriverErr> for NetDevError {
    fn from(_: AtlDriverErr) -> Self {
        NetDevError::DeviceError
    }
}

fn phys_of<T>(pool: &DMAPool<T>) -> Result<u64, AtlDriverErr> {
    u64::try_from(pool.get_phy_addr().as_usize()).map_err(|_| AtlDriverErr::Overflow)
}

/// Physical address of packet buffer `i` inside `pool`.
fn pkt_buf_phys(pool: &DMAPool<PktBuf>, i: usize) -> Result<u64, AtlDriverErr> {
    let offset = i
        .checked_mul(PKT_BUF_SIZE)
        .and_then(|o| u64::try_from(o).ok())
        .ok_or(AtlDriverErr::Overflow)?;
    phys_of(pool)?
        .checked_add(offset)
        .ok_or(AtlDriverErr::Overflow)
}

/// Ring index as the value of a head/tail register.
fn ring_reg(i: usize) -> Result<u32, AtlDriverErr> {
    u32::try_from(i).map_err(|_| AtlDriverErr::Overflow)
}

fn check_queue(que_id: usize) -> Result<(), AtlDriverErr> {
    if que_id == QUEUE_ID {
        Ok(())
    } else {
        Err(AtlDriverErr::InvalidQueue)
    }
}

// ── TX queue ─────────────────────────────────────────────────────────────────

struct Tx {
    /// Next descriptor slot to fill (software tail, driver advances after each send).
    sw_tail: usize,
    /// Oldest in-flight descriptor (updated by reading the hardware head register).
    sw_head: usize,
    /// DMA-mapped descriptor ring.
    desc_ring: DMAPool<DescRing>,
    /// DMA-mapped packet buffers backing the ring.
    pkt_buf: DMAPool<PktBuf>,
}

impl Tx {
    /// Number of free descriptor slots; one slot is kept empty so that
    /// head == tail means "empty".  Computed modulo `NUM_RING_DESC`.
    fn free_slots(&self) -> usize {
        self.sw_head.wrapping_sub(self.sw_tail).wrapping_sub(1) & RING_MASK
    }

    /// Copy `data` into the next DMA buffer and fill its descriptor.
    /// The caller publishes the new tail to hardware afterwards.
    ///
    /// WCET: one copy of at most `PKT_BUF_SIZE` bytes, no allocation.
    fn post_packet(&mut self, data: &[u8]) -> Result<(), AtlDriverErr> {
        if self.free_slots() == 0 {
            return Err(AtlDriverErr::TxRingFull);
        }
        let len = u32::try_from(data.len()).map_err(|_| AtlDriverErr::FrameTooLarge)?;
        if data.len() > PKT_BUF_SIZE {
            return Err(AtlDriverErr::FrameTooLarge);
        }

        let i = self.sw_tail;
        self.pkt_buf
            .as_mut()
            .get_mut(i)
            .and_then(|buf| buf.get_mut(..data.len()))
            .ok_or(AtlDriverErr::Overflow)?
            .copy_from_slice(data);
        let buf_phys = pkt_buf_phys(&self.pkt_buf, i)?;

        // [0] = buffer address, [1] = ctl (low 32) | pay_len (high 32, 0 for non-TSO).
        let ctl = (len & 0xFFFF) | TX_DESC_TYPE_DATA | TX_DESC_EOP | TX_DESC_FCS | TX_DESC_WB;
        let desc = self
            .desc_ring
            .as_mut()
            .get_mut(i)
            .ok_or(AtlDriverErr::Overflow)?;
        // SAFETY: `desc` is a valid, aligned descriptor inside the DMA page.
        unsafe { write_volatile(desc, [buf_phys, u64::from(ctl)]) };

        self.sw_tail = ring_next(i);
        Ok(())
    }

    /// Reclaim descriptors by reading the hardware head pointer.
    fn reclaim(&mut self, mmio: &Mmio) -> Result<(), AtlDriverErr> {
        let head = mmio.read32(TX_RING0_HEAD)?;
        self.sw_head = usize::try_from(head).map_err(|_| AtlDriverErr::Overflow)? & RING_MASK;
        Ok(())
    }

    /// Empty the ring and program its base/length/head/tail registers.
    fn reset(&mut self, mmio: &Mmio) -> Result<(), AtlDriverErr> {
        self.sw_tail = 0;
        self.sw_head = 0;
        for desc in self.desc_ring.as_mut().iter_mut() {
            // SAFETY: `desc` is a valid, aligned descriptor inside the DMA page.
            unsafe { write_volatile(desc, [0; 2]) };
        }
        mmio.write64(
            TX_RING0_BASE_ADDRL,
            TX_RING0_BASE_ADDRH,
            phys_of(&self.desc_ring)?,
        )?;
        mmio.write32(TX_RING0_LEN, ring_reg(RING_MASK)?)?;
        mmio.write32(TX_RING0_HEAD, 0)?;
        mmio.write32(TX_RING0_TAIL, 0)?;
        Ok(())
    }
}

// ── RX queue ─────────────────────────────────────────────────────────────────

struct Rx {
    /// Index of the next descriptor to inspect for a completed packet.
    hw_head: usize,
    /// Index of the last descriptor we handed to hardware (written to the tail register).
    sw_tail: usize,
    /// DMA-mapped descriptor ring.
    desc_ring: DMAPool<DescRing>,
    /// DMA-mapped packet buffers backing the ring.
    pkt_buf: DMAPool<PktBuf>,
    /// Software queue holding received frames waiting to be consumed.
    recv_queue: RingQ<EtherFrameBuf>,
    /// Packets dropped because the software queue was full (saturating counter).
    dropped: u64,
}

impl Rx {
    /// Hand descriptor slot `i` back to hardware with its buffer address.
    fn refill_desc(&mut self, i: usize) -> Result<(), AtlDriverErr> {
        let buf_phys = pkt_buf_phys(&self.pkt_buf, i)?;
        let desc = self
            .desc_ring
            .as_mut()
            .get_mut(i)
            .ok_or(AtlDriverErr::Overflow)?;
        // Submit format: [0] = buf_addr, [1] = 0 (no header split; clears DD).
        // SAFETY: `desc` is a valid, aligned descriptor inside the DMA page.
        unsafe { write_volatile(desc, [buf_phys, 0]) };
        Ok(())
    }

    /// Move completed descriptors into the software queue and recycle them.
    /// Returns the number of frames queued.
    ///
    /// WCET: visits at most `NUM_RING_DESC` descriptors; one heap allocation of
    /// at most `PKT_BUF_SIZE` bytes per received frame (the `EtherFrameBuf` API
    /// owns a `Vec<u8>`); no blocking, no logging.
    fn drain(&mut self, mmio: &Mmio) -> Result<usize, AtlDriverErr> {
        let mut count: usize = 0;

        for _ in 0..NUM_RING_DESC {
            if self.recv_queue.is_full() {
                break;
            }

            let i = self.hw_head;
            let desc = self
                .desc_ring
                .as_ref()
                .get(i)
                .ok_or(AtlDriverErr::Overflow)?;
            // SAFETY: `desc` is a valid, aligned descriptor inside the DMA page;
            // volatile because the NIC writes it back.
            let [_, wb] = unsafe { read_volatile(desc) };
            // Write-back word: status[15:0] | pkt_len[31:16] | next | vlan.
            let [s0, s1, l0, l1, ..] = wb.to_le_bytes();
            if u16::from_le_bytes([s0, s1]) & RX_WB_DD == 0 {
                break; // hardware hasn't written this descriptor yet
            }
            // Read the packet buffer only after observing DD.
            fence(Ordering::Acquire);

            // Runt frames shorter than the FCS end up as length 0 and are dropped.
            let pkt_len = usize::from(u16::from_le_bytes([l0, l1])).saturating_sub(FCS_LEN);
            if (1..=PKT_BUF_SIZE).contains(&pkt_len) {
                let data = self
                    .pkt_buf
                    .as_ref()
                    .get(i)
                    .and_then(|buf| buf.get(..pkt_len))
                    .ok_or(AtlDriverErr::Overflow)?
                    .to_vec();
                if self
                    .recv_queue
                    .push(EtherFrameBuf { data, vlan: None })
                    .is_err()
                {
                    self.dropped = self.dropped.saturating_add(1);
                } else {
                    count = count.saturating_add(1);
                }
            }

            self.refill_desc(i)?;
            self.hw_head = ring_next(self.hw_head);
            self.sw_tail = ring_next(self.sw_tail);
        }

        if count > 0 {
            // Descriptors must be visible before the NIC sees the new tail.
            fence(Ordering::Release);
            mmio.write32(RX_RING0_TAIL, ring_reg(self.sw_tail)?)?;
        }
        Ok(count)
    }

    /// Give every descriptor to hardware and program the ring registers.
    fn reset(&mut self, mmio: &Mmio) -> Result<(), AtlDriverErr> {
        self.hw_head = 0;
        self.sw_tail = RING_MASK;
        for i in 0..NUM_RING_DESC {
            self.refill_desc(i)?;
        }
        mmio.write64(
            RX_RING0_BASE_ADDRL,
            RX_RING0_BASE_ADDRH,
            phys_of(&self.desc_ring)?,
        )?;
        mmio.write32(RX_RING0_LEN, ring_reg(RING_MASK)?)?;
        // Tell hardware where the last valid descriptor is.
        fence(Ordering::Release);
        mmio.write32(RX_RING0_TAIL, ring_reg(self.sw_tail)?)?;
        Ok(())
    }
}

// ── Inner device state ────────────────────────────────────────────────────────

struct AtlInner {
    info: PCIeInfo,
    mmio: Mmio,
    mac_addr: [u8; 6],
    flags: NetFlags,
    capabilities: NetCapabilities,
    /// Registered MSI-X/MSI vector; dropping it unregisters the handler.
    irq: Option<IRQ>,
    irq_num: Option<u16>,
    is_poll_mode: bool,
    link_up: bool,
    link_mbps: u64,
    multicast_addrs: MulticastAddrs,
    tx: Mutex<Tx>,
    rx: Mutex<Rx>,
}

impl AtlInner {
    /// Read the MAC address that firmware loaded into L2 unicast filter 0.
    fn read_mac_addr(mmio: &Mmio) -> Result<[u8; 6], AtlDriverErr> {
        let [m2, m3, m4, m5] = mmio.read32(MAC_FILTER0_ADDRL)?.to_be_bytes();
        let [_, _, m0, m1] = mmio.read32(MAC_FILTER0_ADDRH)?.to_be_bytes();
        Ok([m0, m1, m2, m3, m4, m5])
    }

    /// Wait until firmware signals it has completed boot (≤ FIRMWARE_POLL_ITERS ms).
    fn wait_firmware(mmio: &Mmio) -> Result<(), AtlDriverErr> {
        for _ in 0..FIRMWARE_POLL_ITERS {
            let status = mmio.read32(MPI_BOOT_COMPLETE)?;
            let code = mmio.read32(MPI_BOOT_CODE)?;
            if code != 0 || status & MPI_BOOT_COMPLETE_BIT != 0 {
                return Ok(());
            }
            awkernel_lib::delay::wait_millisec(1);
        }
        Err(AtlDriverErr::FirmwareTimeout)
    }

    /// Full device initialization sequence (called from `up()`).
    fn init(&mut self) -> Result<(), AtlDriverErr> {
        let mmio = self.mmio;

        // 1. Disable TX and RX while reconfiguring.
        mmio.write32(TPB_TX_BUF_ENABLE, 0)?;
        mmio.write32(RPB_RX_BUF_ENABLE, 0)?;
        mmio.write32(TDM_RING_DISABLE, 1)?;
        mmio.write32(RDM_RING_DISABLE, 1)?;

        // 2. Configure and enable TX ring 0.
        {
            let mut node = MCSNode::new();
            let mut tx = self.tx.lock(&mut node);
            tx.reset(&mmio)?;
        }
        mmio.write32(TDM_RING_ENABLE, 1)?;

        // 3. Configure and enable RX ring 0 (all descriptors pre-filled).
        {
            let mut node = MCSNode::new();
            let mut rx = self.rx.lock(&mut node);
            rx.reset(&mmio)?;
        }
        mmio.write32(RDM_RING_ENABLE, 1)?;

        // 4. Unicast filter 0 = our address; enable it and broadcast.
        let [m0, m1, m2, m3, m4, m5] = self.mac_addr;
        mmio.write32(MAC_FILTER0_ADDRL, u32::from_be_bytes([m2, m3, m4, m5]))?;
        mmio.write32(MAC_FILTER0_ADDRH, u32::from_be_bytes([0, 0, m0, m1]))?;
        mmio.write32(MAC_FILTER_ENABLE, 1)?;
        mmio.write32(MAC_BCAST_ENABLE, 1)?;

        // 5. Re-enable TX and RX packet buffers.
        mmio.write32(TPB_TX_BUF_ENABLE, 1)?;
        mmio.write32(RPB_RX_BUF_ENABLE, 1)?;

        self.flags |= NetFlags::RUNNING;

        // 6. Interrupts (MSI-X → MSI → poll mode).
        self.init_interrupts()?;

        // 7. Initial link status.
        self.update_link_status()
    }

    /// Register an interrupt vector: MSI-X first, then MSI, else poll mode.
    fn init_interrupts(&mut self) -> Result<(), AtlDriverErr> {
        if self.irq.is_some() {
            return Ok(()); // already registered by a previous `up()`
        }
        self.info.disable_legacy_interrupt();

        let seg = usize::from(self.info.get_segment_group());
        let bdf = self.info.get_bdf();
        let irq_name = format!("{DEVICE_SHORT_NAME}-{bdf}");
        let cpu =
            u32::try_from(awkernel_lib::cpu::raw_cpu_id()).map_err(|_| AtlDriverErr::Overflow)?;

        if let Some(msi) = self.info.get_msi_mut() {
            msi.disable();
        }

        let mut registered: Option<IRQ> = None;
        if let Some(msix) = self.info.get_msix_mut() {
            msix.disable();
            match msix.register_handler(
                irq_name.clone().into(),
                Box::new(awkernel_lib::net::net_interrupt),
                seg,
                cpu,
                0, // vector 0 = queue 0
            ) {
                Ok(mut irq) => {
                    irq.enable();
                    msix.enable();
                    registered = Some(irq);
                }
                Err(e) => log::error!("atl {bdf}: MSI-X allocation failed ({e:?}), trying MSI"),
            }
        }

        if registered.is_none() {
            if let Some(msi) = self.info.get_msi_mut() {
                msi.disable();
                match msi.register_handler(
                    irq_name.into(),
                    Box::new(awkernel_lib::net::net_interrupt),
                    seg,
                    cpu,
                ) {
                    Ok(mut irq) => {
                        irq.enable();
                        msi.enable();
                        registered = Some(irq);
                    }
                    Err(e) => {
                        log::warn!("atl {bdf}: MSI allocation failed ({e:?}), using poll mode")
                    }
                }
            }
        }

        match registered {
            Some(irq) => {
                self.irq_num = Some(irq.get_irq());
                self.irq = Some(irq);
                self.is_poll_mode = false;
                self.mmio
                    .write32(ATL2_ITR_MASK_SET, ITR_RX0 | ITR_TX0 | ITR_LSC)?;
            }
            None => self.is_poll_mode = true,
        }
        Ok(())
    }

    /// Mask interrupts and stop TX/RX.
    fn stop(&mut self) -> Result<(), AtlDriverErr> {
        if self.irq.is_some() {
            self.mmio.write32(ATL2_ITR_MASK_CLR, !0u32)?;
        }
        self.mmio.write32(TPB_TX_BUF_ENABLE, 0)?;
        self.mmio.write32(RPB_RX_BUF_ENABLE, 0)?;
        self.flags.remove(NetFlags::RUNNING);
        Ok(())
    }

    /// Update cached link status from the PHY status register.
    fn update_link_status(&mut self) -> Result<(), AtlDriverErr> {
        let phy = self.mmio.read32(PHY_STATUS)?;
        let old = self.link_up;
        self.link_mbps = if phy & PHY_STATUS_10G != 0 {
            10_000
        } else if phy & PHY_STATUS_5G != 0 {
            5_000
        } else if phy & PHY_STATUS_2_5G != 0 {
            2_500
        } else if phy & PHY_STATUS_1G != 0 {
            1_000
        } else if phy & PHY_STATUS_100M != 0 {
            100
        } else {
            0
        };
        self.link_up = self.link_mbps != 0;

        if old != self.link_up {
            if self.link_up {
                log::info!(
                    "atl {}: link up, {} Mbps",
                    self.info.get_bdf(),
                    self.link_mbps
                );
            } else {
                log::info!("atl {}: link down", self.info.get_bdf());
            }
        }
        Ok(())
    }

    /// Acknowledge an interrupt and drain RX.  Link changes are picked up by
    /// the periodic `tick` (it needs `&mut self`).
    ///
    /// WCET: one RX drain (see `Rx::drain`) plus two register accesses.
    fn handle_interrupt(&self) -> Result<(), AtlDriverErr> {
        let status = self.mmio.read32(ATL2_ITR_STATUS)?;
        if status & (ITR_RX0 | ITR_TX0) != 0 {
            self.rx_drain()?;
        }
        // Re-arm the causes we just observed.
        self.mmio.write32(ATL2_ITR_MASK_SET, status)?;
        Ok(())
    }

    fn rx_drain(&self) -> Result<usize, AtlDriverErr> {
        let mut node = MCSNode::new();
        let mut rx = self.rx.lock(&mut node);
        rx.drain(&self.mmio)
    }
}

// ── Public driver struct ──────────────────────────────────────────────────────

/// Aquantia Atlantic Ethernet controller driver.
pub struct Atlantic {
    inner: RwLock<AtlInner>,
}

// ── Public entry points ───────────────────────────────────────────────────────

pub fn match_device(vendor: u16, id: u16) -> bool {
    AQUANTIA_DEVICES.contains(&(vendor, id))
}

pub fn attach(mut info: PCIeInfo) -> Result<Arc<dyn PCIeDevice + Sync + Send>, PCIeDeviceErr> {
    if let Err(e) = info.map_bar() {
        log::warn!("atl: failed to map BAR: {e:?}");
        return Err(PCIeDeviceErr::PageTableFailure);
    }
    info.read_capability();
    info.enable_bus_master();

    let atl = Atlantic::new(info)?;
    let result = Arc::new(atl);
    awkernel_lib::net::add_interface(result.clone(), None);
    Ok(result)
}

impl Atlantic {
    fn new(info: PCIeInfo) -> Result<Self, AtlDriverErr> {
        let mmio = Mmio::from_bar(info.get_bar(0)).map_err(|_| AtlDriverErr::NoBar0)?;

        // Contract: a firmware that never reports "boot complete" is tolerated
        // (the MAC registers are still readable on the boards seen so far);
        // register access failures are not.
        match AtlInner::wait_firmware(&mmio) {
            Ok(()) => {}
            Err(AtlDriverErr::FirmwareTimeout) => log::warn!(
                "atl {}: firmware not ready after {FIRMWARE_POLL_ITERS} ms, continuing",
                info.get_bdf()
            ),
            Err(e) => return Err(e),
        }

        let mac_addr = AtlInner::read_mac_addr(&mmio)?;
        let [m0, m1, m2, m3, m4, m5] = mac_addr;
        log::info!(
            "atl {}: MAC {m0:02x}:{m1:02x}:{m2:02x}:{m3:02x}:{m4:02x}:{m5:02x}",
            info.get_bdf(),
        );

        let tx = Tx {
            sw_tail: 0,
            sw_head: 0,
            desc_ring: DMAPool::new(NUMA_NODE, DESC_RING_PAGES).ok_or(AtlDriverErr::DmaAlloc)?,
            pkt_buf: DMAPool::new(NUMA_NODE, PKT_BUF_PAGES).ok_or(AtlDriverErr::DmaAlloc)?,
        };
        let rx = Rx {
            hw_head: 0,
            sw_tail: RING_MASK,
            desc_ring: DMAPool::new(NUMA_NODE, DESC_RING_PAGES).ok_or(AtlDriverErr::DmaAlloc)?,
            pkt_buf: DMAPool::new(NUMA_NODE, PKT_BUF_PAGES).ok_or(AtlDriverErr::DmaAlloc)?,
            recv_queue: RingQ::new(SW_RX_QUEUE_SIZE),
            dropped: 0,
        };

        let inner = AtlInner {
            info,
            mmio,
            mac_addr,
            flags: NetFlags::BROADCAST | NetFlags::SIMPLEX | NetFlags::MULTICAST,
            capabilities: NetCapabilities::VLAN_MTU,
            irq: None,
            irq_num: None,
            is_poll_mode: true,
            link_up: false,
            link_mbps: 0,
            multicast_addrs: MulticastAddrs::new(),
            tx: Mutex::new(tx),
            rx: Mutex::new(rx),
        };

        Ok(Self {
            inner: RwLock::new(inner),
        })
    }
}

// ── PCIeDevice ────────────────────────────────────────────────────────────────

impl PCIeDevice for Atlantic {
    fn device_name(&self) -> Cow<'static, str> {
        let bdf = self.inner.read().info.get_bdf();
        format!("{bdf}: {DEVICE_NAME}").into()
    }

    fn config_space(&self) -> Option<crate::pcie::config_space::ConfigSpace> {
        let inner = self.inner.read();
        Some(inner.info.config_space.clone())
    }
}

// ── NetDevice ─────────────────────────────────────────────────────────────────

impl NetDevice for Atlantic {
    fn num_queues(&self) -> usize {
        1
    }

    fn flags(&self) -> NetFlags {
        self.inner.read().flags
    }

    fn capabilities(&self) -> NetCapabilities {
        self.inner.read().capabilities
    }

    fn device_short_name(&self) -> Cow<'static, str> {
        let bdf = self.inner.read().info.get_bdf();
        format!("{DEVICE_SHORT_NAME}-{bdf}").into()
    }

    fn link_status(&self) -> LinkStatus {
        if self.inner.read().link_up {
            LinkStatus::UpFullDuplex
        } else {
            LinkStatus::Down
        }
    }

    fn link_speed(&self) -> u64 {
        self.inner.read().link_mbps
    }

    fn mac_address(&self) -> [u8; 6] {
        self.inner.read().mac_addr
    }

    fn can_send(&self) -> bool {
        let inner = self.inner.read();
        inner.flags.contains(NetFlags::RUNNING) && inner.link_up
    }

    // ── UP / DOWN ─────────────────────────────────────────────────────────────

    fn up(&self) -> Result<(), NetDevError> {
        let mut inner = self.inner.write();
        if inner.flags.contains(NetFlags::UP) {
            return Err(NetDevError::AlreadyUp);
        }
        inner.init().map_err(|e| {
            log::error!("atl: init failed: {e}");
            NetDevError::DeviceError
        })?;
        inner.flags.insert(NetFlags::UP);
        Ok(())
    }

    fn down(&self) -> Result<(), NetDevError> {
        let mut inner = self.inner.write();
        if !inner.flags.contains(NetFlags::UP) {
            return Err(NetDevError::AlreadyDown);
        }
        inner.stop().map_err(|e| {
            log::error!("atl: stop failed: {e}");
            NetDevError::DeviceError
        })?;
        inner.flags.remove(NetFlags::UP);
        Ok(())
    }

    // ── TX ────────────────────────────────────────────────────────────────────

    /// WCET: one head-register read when the ring looks full, one copy of at
    /// most `PKT_BUF_SIZE` bytes, one tail-register write.  No allocation, no
    /// logging; the TX mutex is held only for this bounded work.
    fn send(&self, data: EtherFrameRef, que_id: usize) -> Result<(), NetDevError> {
        check_queue(que_id)?;
        let inner = self.inner.read();
        if !inner.flags.contains(NetFlags::RUNNING) || !inner.link_up {
            return Err(NetDevError::DeviceError);
        }

        let mut node = MCSNode::new();
        let mut tx = inner.tx.lock(&mut node);
        if tx.free_slots() == 0 {
            tx.reclaim(&inner.mmio)?;
        }
        tx.post_packet(data.data)?;

        // Descriptor must be visible before the NIC sees the new tail.
        fence(Ordering::Release);
        inner
            .mmio
            .write32(TX_RING0_TAIL, ring_reg(tx.sw_tail)?)
            .map_err(AtlDriverErr::from)?;
        Ok(())
    }

    // ── RX ────────────────────────────────────────────────────────────────────

    /// WCET: one queue pop, or one RX drain (see `Rx::drain`) followed by a pop.
    fn recv(&self, que_id: usize) -> Result<Option<EtherFrameBuf>, NetDevError> {
        check_queue(que_id)?;
        let inner = self.inner.read();

        let mut node = MCSNode::new();
        let mut rx = inner.rx.lock(&mut node);
        if let Some(frame) = rx.recv_queue.pop() {
            return Ok(Some(frame));
        }
        rx.drain(&inner.mmio)?;
        Ok(rx.recv_queue.pop())
    }

    // ── Interrupts ────────────────────────────────────────────────────────────

    fn interrupt(&self, _irq: u16) -> Result<(), NetDevError> {
        self.inner.read().handle_interrupt()?;
        Ok(())
    }

    fn irqs(&self) -> Vec<u16> {
        self.inner.read().irq_num.into_iter().collect()
    }

    fn rx_irq_to_que_id(&self, _irq: u16) -> Option<usize> {
        Some(QUEUE_ID)
    }

    // ── Polling ───────────────────────────────────────────────────────────────

    fn poll_mode(&self) -> bool {
        self.inner.read().is_poll_mode
    }

    /// Returns true when the hardware head has moved past our software head.
    ///
    /// The `bool` API cannot carry an error, so a failed register read reports
    /// `true`: the following `poll_in_service` performs the same access and
    /// returns the error to the caller instead of hiding it.
    fn poll(&self) -> bool {
        let inner = self.inner.read();
        if !inner.flags.contains(NetFlags::RUNNING) {
            return false;
        }
        let Some(hw_head) = inner
            .mmio
            .read32(RX_RING0_HEAD)
            .ok()
            .and_then(|h| usize::try_from(h).ok())
        else {
            return true;
        };
        let mut node = MCSNode::new();
        let rx = inner.rx.lock(&mut node);
        hw_head & RING_MASK != rx.hw_head
    }

    fn poll_in_service(&self) -> Result<(), NetDevError> {
        self.inner.read().rx_drain()?;
        Ok(())
    }

    // ── Periodic tick ─────────────────────────────────────────────────────────

    fn tick_msec(&self) -> Option<u64> {
        Some(TICK_MSEC)
    }

    fn tick(&self) -> Result<(), NetDevError> {
        self.inner.write().update_link_status()?;
        Ok(())
    }

    // ── Multicast ─────────────────────────────────────────────────────────────

    /// The address is recorded only: the hardware multicast filter is not
    /// programmed yet, so reception follows the controller's default filter.
    fn add_multicast_addr(&self, addr: &[u8; 6]) -> Result<(), NetDevError> {
        self.inner.write().multicast_addrs.add_addr(*addr);
        Ok(())
    }

    fn remove_multicast_addr(&self, addr: &[u8; 6]) -> Result<(), NetDevError> {
        if self.inner.write().multicast_addrs.remove_addr(addr) {
            Ok(())
        } else {
            Err(NetDevError::MulticastAddrError)
        }
    }
}
