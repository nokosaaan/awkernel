pub const STACK_SIZE: usize = 2 * 1024 * 1024; // 2MiB
pub const STACK_START: usize = 256 * 1024 * 1024; // 256MiB

pub const DMA_START: usize = 0x40000000000;

pub const HEAP_START: usize = 0x41000000000;

pub const PREEMPT_IRQ: u16 = 255;
pub const WAKEUP_IRQ: u16 = 254;

pub const DMA_SIZE: usize = 128 * 1024 * 1024; // 128MiB

/// I/O port base for the 16550-compatible serial console
/// (`kernel/src/arch/x86_64/console.rs`). `0x3F8` is the PC/AT convention
/// for COM1, correct on most machines, but not universal: confirmed on the
/// current Windows real-machine target (Lenovo Legion, 2026-09) that the
/// only wired-up/working serial port enumerates as COM3 (standard address
/// `0x3E8`) in Windows' own Device Manager, not COM1 -- awkernel was
/// silently writing to a COM1 register that either doesn't exist or isn't
/// connected to anything on this board, hence zero bytes ever reaching the
/// host's minicom capture despite the framebuffer console, PXE/TFTP
/// transfer, and the cable/adapter itself all working correctly (confirmed
/// separately by sending test bytes to COM3 from Windows PowerShell and
/// seeing them arrive at minicom). Change this to match whichever COM port
/// Device Manager (or `dmesg`/BIOS on Linux targets) actually shows as
/// connected on a given target: `0x3F8` COM1, `0x2F8` COM2, `0x3E8` COM3,
/// `0x2E8` COM4.
pub const SERIAL_PORT_BASE: u16 = 0x3E8;
