use awkernel_lib::{addr::Addr, dma_pool::DMAPool};

use super::regs::{trb_type, TRB_CYCLE, TRB_LINK_TC, TRB_TYPE_SHIFT};

/// Number of TRBs in the Command Ring. The last entry is the Link TRB.
/// 256 × 16 bytes = 4096 bytes = 1 page, so DMAPool<CmdRingMem>::new(n, 1) works.
pub const CMD_RING_SIZE: usize = 256;

/// Number of TRBs in the primary Event Ring segment.
/// 256 × 16 bytes = 4096 bytes = 1 page.
pub const EVT_RING_SIZE: usize = 256;

/// Transfer Request Block — 16 bytes. xHCI spec §4.11.1.
///
/// Layout (little-endian):
///   [63:0]  param   — address, data, or command-specific parameter
///   [95:64] status  — length, completion code, …
///   [127:96] ctrl   — cycle bit[0], TRB-specific flags, TRB type[15:10]
#[repr(C)]
#[derive(Default, Copy, Clone)]
pub struct Trb {
    pub param: u64,
    pub status: u32,
    pub ctrl: u32,
}

/// Flat array of TRBs backing the Command Ring (1 page).
pub type CmdRingMem = [Trb; CMD_RING_SIZE];

    /// Completion code of an event TRB (status[31:24]).
    pub fn completion_code(&self) -> u8 {
        let [_, _, _, code] = self.status.to_le_bytes();
        code
    }

    /// TRB Transfer Length of a Transfer Event (status[23:0]): the bytes the
    /// TRB did not transfer (xHCI spec §6.4.2.1).
    pub fn transfer_length(&self) -> u32 {
        self.status & 0x00ff_ffff
    }

    /// Slot ID of a Command Completion event (ctrl[31:24]).
    pub fn slot_id(&self) -> u8 {
        let [_, _, _, slot] = self.ctrl.to_le_bytes();
        slot
    }

    /// Endpoint ID (DCI) of a Transfer Event (ctrl[20:16]).
    pub fn endpoint_id(&self) -> u8 {
        let [_, _, ep, _] = self.ctrl.to_le_bytes();
        ep & 0x1f
    }
}

/// Event Ring Segment Table entry — 16 bytes. xHCI spec §6.5.
///
/// The ERST consists of one or more of these entries.  We use a single-entry
/// table pointing at the sole event ring segment.
#[repr(C)]
#[derive(Default, Copy, Clone)]
pub struct ErstEntry {
    /// Physical base address of the ring segment (must be 64-byte aligned).
    pub ring_segment_base: u64,
    /// bits[15:0] = segment size (number of TRBs), bits[63:16] must be zero.
    pub size_rsvd: u64,
}

pub type CommandRing = ProducerRing;
pub type TransferRing = ProducerRing;

impl ProducerRing {
    /// Allocate a zeroed ring with its Link TRB pointing back to the start.
    pub fn new(numa_id: usize) -> Result<Self, XhciErr> {
        let mem = DMAPool::<RingMem>::new(numa_id, 1).ok_or(XhciErr::DmaAlloc)?;
        let mut ring = Self {
            mem,
            enqueue_idx: 0,
            cycle_bit: trb_flag::CYCLE,
        };
        ring.reset()?;
        Ok(ring)
    }

    /// Clear all TRBs, reinstall the Link TRB and rewind to index 0 (cycle = 1).
    ///
    /// Caller must ensure the controller is not actively fetching from this ring
    /// (e.g. `CRCR.CRR == 0` for the Command Ring).
    pub fn reset(&mut self) -> Result<(), XhciErr> {
        let base = self.phys_base()?;
        for slot in self.mem.as_mut().iter_mut() {
            // SAFETY: `slot` is a valid, aligned &mut Trb inside the DMA page.
            unsafe { write_volatile(slot, Trb::default()) };
        }
        let link = self
            .mem
            .as_mut()
            .get_mut(LINK_IDX)
            .ok_or(XhciErr::Overflow)?;
        let link_trb = Trb {
            param: base,
            status: 0,
            // TC=1 toggles cycle state when the controller wraps; initial cycle = 1.
            ctrl: trb_type::LINK | trb_flag::LINK_TC | trb_flag::CYCLE,
        };
        // SAFETY: `link` is a valid, aligned &mut Trb inside the DMA page.
        unsafe { write_volatile(link, link_trb) };

        self.enqueue_idx = 0;
        self.cycle_bit = trb_flag::CYCLE;
        Ok(())
    }

    /// Write `trb` at the enqueue position, stamping the Producer Cycle State.
    /// Wraps through the Link TRB when the last data slot is used.
    ///
    /// The caller must call [`publish`] and then ring the doorbell.
    ///
    /// Returns the physical address of the written TRB.  The controller reports
    /// this address in the event it generates for the TRB (Command Completion
    /// Event / Transfer Event, xHCI spec §6.4.2), so callers can match the event
    /// against the TRB that produced it.
    pub fn enqueue(&mut self, mut trb: Trb) -> Result<u64, XhciErr> {
        let idx = self.enqueue_idx;
        let pcs = self.cycle_bit;
        trb.ctrl = (trb.ctrl & !trb_flag::CYCLE) | pcs;
        let trb_phys = u64::try_from(idx)
            .ok()
            .and_then(|i| i.checked_mul(TRB_BYTES))
            .and_then(|off| self.phys_base().ok()?.checked_add(off))
            .ok_or(XhciErr::Overflow)?;

        // Stamp cycle bit (preserve all other ctrl bits set by caller).
        trb.ctrl = (trb.ctrl & !TRB_CYCLE) | pcs;

        let next = idx + 1;
        let wraps = next == CMD_RING_SIZE - 1;

        {
            let mem = self.mem.as_mut();
            mem[idx] = trb;
            if wraps {
                // Update Link TRB's cycle bit to match current PCS so the controller
                // recognises it and toggles its internal cycle state (TC=1).
                let ctrl = mem[CMD_RING_SIZE - 1].ctrl;
                mem[CMD_RING_SIZE - 1].ctrl = (ctrl & !TRB_CYCLE) | pcs;
            }
        } // mutable borrow of self.mem ends here

        if wraps {
            self.enqueue_idx = 0;
            self.cycle_bit ^= 1;
        } else {
            self.enqueue_idx = next;
        }
        Ok(trb_phys)
    }

    /// Zero all TRBs and install the Link TRB at the last slot.
    pub fn init(&mut self) {
        // Compute physical base before taking the mutable borrow on mem.
        let phy_base = self.phys_base();

        let trbs = self.mem.as_mut();
        for trb in trbs.iter_mut() {
            *trb = Trb::default();
        }

        let link = &mut trbs[CMD_RING_SIZE - 1];
        link.param = phy_base;
        // TC=1 toggles cycle state when the controller wraps; initial cycle = 1.
        link.ctrl = (trb_type::LINK << TRB_TYPE_SHIFT) | TRB_LINK_TC | TRB_CYCLE;
    }

    /// Physical base address of the ring (written to CRCR during init).
    pub fn phys_base(&self) -> u64 {
        self.mem.get_phy_addr().as_usize() as u64
    }

    /// Physical address of the enqueue position with the DCS bit (bit 0) set to
    /// the current cycle — the value for SET_TR_DEQUEUE_POINTER.
    pub fn enqueue_pointer(&self) -> Result<u64, XhciErr> {
        let offset = u64::try_from(self.enqueue_idx)
            .ok()
            .and_then(|i| i.checked_mul(TRB_BYTES))
            .ok_or(XhciErr::Overflow)?;
        let addr = self
            .phys_base()?
            .checked_add(offset)
            .ok_or(XhciErr::Overflow)?;
        Ok(addr | u64::from(self.cycle_bit))
    }
}

// ---------------------------------------------------------------------------
// Event Ring
// ---------------------------------------------------------------------------

pub struct EventRing {
    /// DMA memory for the ring segment (the actual TRBs).
    pub seg_mem: DMAPool<EvtRingMem>,
    /// DMA memory for the Event Ring Segment Table.
    pub erst_mem: DMAPool<ErstMem>,
    /// Software dequeue index into `seg_mem`.
    pub dequeue_idx: usize,
    /// Current Consumer Cycle State.
    pub cycle_bit: u32,
}

impl EventRing {
    pub fn new(numa_id: usize) -> Option<Self> {
        let seg_mem = DMAPool::<EvtRingMem>::new(numa_id, 1)?;
        let erst_mem = DMAPool::<ErstMem>::new(numa_id, 1)?;
        Some(Self {
            seg_mem,
            erst_mem,
            dequeue_idx: 0,
            cycle_bit: 1,
        })
    }

    /// Return the next event TRB if its cycle bit matches the Consumer Cycle State.
    /// Advances the dequeue index and toggles the cycle state on ring wrap-around.
    pub fn dequeue(&mut self) -> Option<Trb> {
        let idx = self.dequeue_idx;
        let slot = self.seg_mem.as_ref().get(idx).ok_or(XhciErr::Overflow)?;
        // Check the cycle bit before reading the rest: the controller's DMA write
        // of a TRB is not atomic to the CPU, so a whole-TRB read can pair the new
        // `ctrl` with a stale `param` (observed on Intel 00:14.0: every event's
        // TRB pointer read as 0).
        // SAFETY: `slot` is a valid, aligned &Trb inside the DMA page.
        let ctrl = unsafe { read_volatile(&raw const slot.ctrl) };
        if ctrl & trb_flag::CYCLE != self.cycle_bit {
            return Ok(None); // no event ready
        }
        // The rest of this TRB, and everything the controller wrote before it
        // (e.g. a descriptor in a data buffer), must be read after the check.
        fence(Ordering::Acquire);
        // SAFETY: as above.
        let trb = unsafe { read_volatile(slot) };

        let next = idx + 1;
        if next >= EVT_RING_SIZE {
            self.dequeue_idx = 0;
            self.cycle_bit ^= 1;
        } else {
            self.dequeue_idx = next;
        }
        Some(trb)
    }

    /// Physical address of the current dequeue position (written to ERDP after processing).
    pub fn dequeue_phys(&self) -> u64 {
        self.seg_phys() + (self.dequeue_idx * core::mem::size_of::<Trb>()) as u64
    }

    /// Zero the segment and populate the ERST entry.
    pub fn init(&mut self) {
        for trb in self.seg_mem.as_mut().iter_mut() {
            *trb = Trb::default();
        }
        let seg_base = self.seg_phys();
        let erst = self.erst_mem.as_mut();
        erst[0].ring_segment_base = seg_base;
        erst[0].size_rsvd = EVT_RING_SIZE as u64;
    }

    /// Physical address of the ERST (written to ERSTBA during init).
    pub fn erst_phys(&self) -> u64 {
        self.erst_mem.get_phy_addr().as_usize() as u64
    }

    /// Physical base of the event ring segment (written to ERDP during init).
    pub fn seg_phys(&self) -> u64 {
        self.seg_mem.get_phy_addr().as_usize() as u64
    }
}

// ---------------------------------------------------------------------------
// Device Context Base Address Array (DCBAA)
// ---------------------------------------------------------------------------

/// Slot 0 = scratchpad-buffer-array pointer (or 0); slots 1–255 = Device Context pointers.
/// 256 × 8 bytes = 2048 bytes — fits in one DMA page.
pub const DCBAA_SIZE: usize = 256;
pub type DcbaaMem = [u64; DCBAA_SIZE];

pub struct Dcbaa {
    pub mem: DMAPool<DcbaaMem>,
}

impl Dcbaa {
    pub fn new(numa_id: usize) -> Option<Self> {
        let mem = DMAPool::<DcbaaMem>::new(numa_id, 1)?;
        Some(Self { mem })
    }

    pub fn init(&mut self) {
        for slot in self.mem.as_mut().iter_mut() {
            *slot = 0;
        }
    }

    /// Physical base address (written to DCBAAP during init).
    pub fn phys_base(&self) -> u64 {
        self.mem.get_phy_addr().as_usize() as u64
    }
}

// ---------------------------------------------------------------------------
// Transfer Ring  (EP0 default control pipe, and future bulk endpoints)
// ---------------------------------------------------------------------------

pub struct TransferRing {
    pub mem: DMAPool<XferRingMem>,
    /// Software enqueue index (0 … XFER_RING_SIZE-2; skips the Link TRB slot).
    pub enqueue_idx: usize,
    /// Current Producer Cycle State.
    pub cycle_bit: u32,
}

impl TransferRing {
    pub fn new(numa_id: usize) -> Option<Self> {
        let mem = DMAPool::<XferRingMem>::new(numa_id, 1)?;
        Some(Self {
            mem,
            enqueue_idx: 0,
            cycle_bit: 1,
        })
    }

    /// Zero all TRBs and install the Link TRB at the last slot.
    pub fn init(&mut self) {
        let phy_base = self.phys_base();
        let trbs = self.mem.as_mut();
        for trb in trbs.iter_mut() {
            *trb = Trb::default();
        }
        let link = &mut trbs[XFER_RING_SIZE - 1];
        link.param = phy_base;
        link.ctrl = (trb_type::LINK << TRB_TYPE_SHIFT) | TRB_LINK_TC | TRB_CYCLE;
    }

    /// Enqueue a TRB, stamping the current Producer Cycle State onto it.
    pub fn enqueue(&mut self, mut trb: Trb) -> bool {
        if self.enqueue_idx >= XFER_RING_SIZE - 1 {
            return false;
        }
        let idx = self.enqueue_idx;
        let pcs = self.cycle_bit;
        trb.ctrl = (trb.ctrl & !TRB_CYCLE) | pcs;
        let next = idx + 1;
        let wraps = next == XFER_RING_SIZE - 1;
        {
            let mem = self.mem.as_mut();
            mem[idx] = trb;
            if wraps {
                let ctrl = mem[XFER_RING_SIZE - 1].ctrl;
                mem[XFER_RING_SIZE - 1].ctrl = (ctrl & !TRB_CYCLE) | pcs;
            }
        }
        if wraps {
            self.enqueue_idx = 0;
            self.cycle_bit ^= 1;
        } else {
            self.enqueue_idx = next;
        }
        true
    }

    /// Physical base address of the ring (written to EP Context TR Dequeue Pointer).
    pub fn phys_base(&self) -> u64 {
        self.mem.get_phy_addr().as_usize() as u64
    }
}
