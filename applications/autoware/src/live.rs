//! Live inputs from AWSIM over UDP (feature `live`): the closed-loop setup. The host
//! (`awkernel_script/awsim_bridge/awsim_bridge.py --feed`) forwards AWSIM's IMU and vehicle
//! velocity, the Main ECU NDT pose (Autoware's localization on the host), the MRM trigger and the
//! gate events to `LIVE_PORT`, one row per line in the `replay` CSV layouts, prefixed by its kind:
//!
//! | kind | columns after the kind |
//! |---|---|
//! | `imu` | `imu_raw.csv` |
//! | `vel` | `velocity_status.csv` |
//! | `pose` | `pose_with_covariance.csv` |
//! | `trig` | `mrm_trigger.csv` |
//! | `gate` | `gate_events.csv` |
//!
//! Every input is restamped on arrival with the Awkernel uptime (the host's ROS clock is not
//! shared), so input ages measured downstream include the host -> guest network.
//!
//! Only the route (`TRAJECTORY_CSV_PATH`) is embedded at build time.

use alloc::vec::Vec;
use awkernel_lib::sync::mutex::{MCSNode, Mutex};
use core::sync::atomic::{AtomicU32, Ordering};

use ekf_localizer::{PoseWithCovariance, PoseWithCovarianceStamped};
use imu_driver::ImuCsvRow;
use localization_types::{ConstantJerkDecelerationTrigger, Instant};
use vehicle_velocity_converter::VelocityCsvRow;

use crate::awsim_rows::{gate_row, load_trajectory, pose_row, trigger_row};
use crate::nodes::GateInput;
use crate::shared::Latest;
use crate::{
    get_awkernel_uptime_timestamp, imu_row, velocity_row, INTERFACE_ID, TRAJECTORY_CSV_DATA_STR,
};

/// Guest port the host feeds (QEMU: `hostfwd=udp::26200-10.0.2.64:26200`).
pub const LIVE_PORT: u16 = 26200;

static IMU: Latest<(ImuCsvRow, u64)> = Latest::new();
static VELOCITY: Latest<(VelocityCsvRow, u64)> = Latest::new();
/// The newest pose not yet taken by the pose estimator node.
static POSE: Mutex<Option<PoseWithCovarianceStamped>> = Mutex::new(None);
static TRIGGER: Latest<ConstantJerkDecelerationTrigger> = Latest::new();
static GATE: Latest<GateInput> = Latest::new();

static RECEIVED: [AtomicU32; 5] = [const { AtomicU32::new(0) }; 5];
static MALFORMED: AtomicU32 = AtomicU32::new(0);
const KINDS: [&str; 5] = ["imu", "vel", "pose", "trig", "gate"];

/// Set the followed route (the only embedded input).
pub fn load() -> Result<(), &'static str> {
    let points = load_trajectory(TRAJECTORY_CSV_DATA_STR)?;
    log::info!(
        "live: {} trajectory points; inputs on UDP port {}",
        points,
        LIVE_PORT
    );
    Ok(())
}

/// Nothing to align: inputs carry their arrival time.
pub fn start() {}

pub fn imu() -> Option<(ImuCsvRow, u64)> {
    IMU.get()
}

pub fn velocity() -> Option<(VelocityCsvRow, u64)> {
    VELOCITY.get()
}

/// The newest pose received since the previous call.
pub fn next_pose() -> Option<PoseWithCovarianceStamped> {
    let mut node = MCSNode::new();
    let pose = POSE.lock(&mut node).take();
    pose
}

/// The newest trigger message (off until the first one arrives).
pub fn trigger_state() -> ConstantJerkDecelerationTrigger {
    TRIGGER.get().unwrap_or(ConstantJerkDecelerationTrigger {
        stamp: Instant::now(),
        ..Default::default()
    })
}

/// The newest gate state from the host (builtin source, not paused, until the first one).
pub fn gate_state() -> GateInput {
    GATE.get().unwrap_or_default()
}

/// Live inputs never run out.
pub fn finished() -> bool {
    false
}

pub async fn start_receiver() {
    awkernel_async_lib::spawn(
        "live_input_receiver".into(),
        receiver_task(),
        awkernel_async_lib::scheduler::SchedulerType::GEDF(5),
    )
    .await;
}

async fn receiver_task() {
    let config = awkernel_async_lib::net::udp::UdpConfig {
        port: Some(LIVE_PORT),
        ..Default::default()
    };
    let mut socket =
        match awkernel_async_lib::net::udp::UdpSocket::bind_on_interface(INTERFACE_ID, &config) {
            Ok(socket) => socket,
            Err(e) => {
                log::error!("live: failed to bind UDP port {}: {:?}", LIVE_PORT, e);
                return;
            }
        };
    log::info!("live: receiving inputs on UDP port {}", LIVE_PORT);

    let mut buf = alloc::vec![0u8; 8192];
    let mut packets: u64 = 0;
    loop {
        let n = match socket.recv(&mut buf).await {
            Ok((n, _, _)) => n,
            Err(e) => {
                log::warn!("live: receive failed: {:?}", e);
                continue;
            }
        };
        let arrival = get_awkernel_uptime_timestamp();
        match core::str::from_utf8(&buf[..n]) {
            Ok(text) => {
                for line in text.lines().filter(|l| !l.is_empty()) {
                    if let Err(e) = handle_line(line, arrival) {
                        MALFORMED.fetch_add(1, Ordering::Relaxed);
                        log::warn!("live: dropped row ({e}): {line}");
                    }
                }
            }
            Err(_) => {
                MALFORMED.fetch_add(1, Ordering::Relaxed);
            }
        }
        packets += 1;
        if packets == 1 || packets % 1000 == 0 {
            log_counts(packets);
        }
    }
}

fn log_counts(packets: u64) {
    let counts: Vec<u32> = RECEIVED.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    log::info!(
        "live: {} packets; {}={} {}={} {}={} {}={} {}={} malformed={}",
        packets,
        KINDS[0],
        counts[0],
        KINDS[1],
        counts[1],
        KINDS[2],
        counts[2],
        KINDS[3],
        counts[3],
        KINDS[4],
        counts[4],
        MALFORMED.load(Ordering::Relaxed)
    );
}

/// One `kind,col,col,...` row; the recorded stamp is replaced by `arrival`.
fn handle_line(line: &str, arrival: u64) -> Result<(), &'static str> {
    let mut fields: Vec<&str> = line.split(',').collect();
    if fields.is_empty() {
        return Err("empty row");
    }
    let kind = fields.remove(0);
    let index = KINDS
        .iter()
        .position(|k| *k == kind)
        .ok_or("unknown input kind")?;
    match index {
        0 => IMU.set((imu_row(&fields)?, arrival)),
        1 => VELOCITY.set((velocity_row(&fields)?, arrival)),
        2 => {
            let (_, pose, covariance) = pose_row(&fields)?;
            let msg = PoseWithCovarianceStamped {
                header: common_types::Header {
                    frame_id: "map",
                    timestamp: arrival,
                },
                pose: PoseWithCovariance { pose, covariance },
            };
            let mut node = MCSNode::new();
            *POSE.lock(&mut node) = Some(msg);
        }
        3 => {
            let (_, trigger, target_acceleration, target_jerk) = trigger_row(&fields)?;
            TRIGGER.set(ConstantJerkDecelerationTrigger {
                stamp: crate::bridge::ns_to_instant(arrival),
                trigger,
                target_acceleration,
                target_jerk,
            });
        }
        _ => GATE.set(gate_row(&fields)?.1),
    }
    RECEIVED[index].fetch_add(1, Ordering::Relaxed);
    Ok(())
}
