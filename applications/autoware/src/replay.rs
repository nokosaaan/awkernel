//! Replay of recorded inputs (feature `replay`): the open-loop setup where a rosbag of the
//! Autoware run (A) is converted to CSV and embedded at build time
//! (`awkernel_script/awsim_bridge/bag_to_awkernel_csv.py` writes these files).
//!
//! | env var at build time | CSV columns (header row first) |
//! |---|---|
//! | `IMU_CSV_PATH` | existing `imu_raw.csv` layout |
//! | `VELOCITY_CSV_PATH` | existing `velocity_status.csv` layout |
//! | `POSE_CSV_PATH` | `timestamp_sec,timestamp_nanosec,x,y,z,qx,qy,qz,qw,cov_0..cov_35` (Main ECU NDT output) |
//! | `TRAJECTORY_CSV_PATH` | `x,y,z,qx,qy,qz,qw,longitudinal_velocity_mps` (the fixed route) |
//! | `MRM_TRIGGER_CSV_PATH` | `timestamp_sec,timestamp_nanosec,trigger,target_acceleration,target_jerk` |
//!
//! Timeline: bag time `t0` (the earliest IMU / velocity / pose stamp) is mapped to the DAG start
//! on the Awkernel clock. A row becomes due once `stamp - t0` has elapsed, and its header stamp is
//! that mapped time, so message ages and duplicate detection behave as on the original topics.

use alloc::vec::Vec;
use awkernel_lib::sync::mutex::{MCSNode, Mutex};
use core::sync::atomic::{AtomicU64, Ordering};

use ekf_localizer::{Point3D, Pose, PoseWithCovariance, PoseWithCovarianceStamped, Quaternion};
use localization_types::{ConstantJerkDecelerationTrigger, Instant, TrajectoryPoint};

use crate::{
    bridge, get_awkernel_uptime_timestamp, parse_csv_records, parse_f64, parse_timestamp,
    IMU_CSV_DATA, MRM_TRIGGER_CSV_DATA_STR, POSE_CSV_DATA_STR, TRAJECTORY_CSV_DATA_STR,
    VELOCITY_CSV_DATA,
};

#[derive(Clone)]
struct PoseRow {
    stamp_ns: u64,
    pose: Pose,
    covariance: [f64; 36],
}

#[derive(Clone, Copy)]
struct TriggerRow {
    stamp_ns: u64,
    trigger: bool,
    target_acceleration: f32,
    target_jerk: f32,
}

static POSES: Mutex<Vec<PoseRow>> = Mutex::new(Vec::new());
static TRIGGERS: Mutex<Vec<TriggerRow>> = Mutex::new(Vec::new());

/// Bag time of the first sample, and the Awkernel uptime the replay started at.
static BAG_T0_NS: AtomicU64 = AtomicU64::new(0);
static START_UPTIME_NS: AtomicU64 = AtomicU64::new(0);

/// Read cursors: index of the next not-yet-due row of each stream.
pub(crate) static IMU_CURSOR: Mutex<usize> = Mutex::new(0);
pub(crate) static VELOCITY_CURSOR: Mutex<usize> = Mutex::new(0);
static POSE_CURSOR: Mutex<usize> = Mutex::new(0);
static TRIGGER_CURSOR: Mutex<usize> = Mutex::new(0);

/// Parse the replay CSVs, set the followed trajectory, and fix the bag time origin.
/// IMU / velocity rows are parsed by `initialize_csv_data` beforehand.
pub fn load() -> Result<(), &'static str> {
    let poses = parse_pose_csv(POSE_CSV_DATA_STR)?;
    let triggers = parse_trigger_csv(MRM_TRIGGER_CSV_DATA_STR)?;
    let trajectory = parse_trajectory_csv(TRAJECTORY_CSV_DATA_STR)?;
    log::info!(
        "replay: {} poses, {} trigger rows, {} trajectory points",
        poses.len(),
        triggers.len(),
        trajectory.len()
    );

    if !trajectory.is_empty() && !trajectory_follower::set_trajectory(&trajectory) {
        log::error!(
            "replay: trajectory has {} points (max {}); keeping the built-in one",
            trajectory.len(),
            trajectory_follower::MAX_TRAJECTORY_POINTS
        );
    }

    let first_imu = unsafe { IMU_CSV_DATA.as_ref() }.and_then(|v| v.first().map(|r| r.timestamp));
    let first_vel =
        unsafe { VELOCITY_CSV_DATA.as_ref() }.and_then(|v| v.first().map(|r| r.timestamp));
    let first_pose = poses.first().map(|r| r.stamp_ns);
    let t0 = [first_imu, first_vel, first_pose]
        .into_iter()
        .flatten()
        .min()
        .ok_or("replay: no IMU, velocity or pose rows to replay")?;
    BAG_T0_NS.store(t0, Ordering::Relaxed);

    let mut node = MCSNode::new();
    *POSES.lock(&mut node) = poses;
    let mut node = MCSNode::new();
    *TRIGGERS.lock(&mut node) = triggers;
    Ok(())
}

/// Start the replay clock (called when the DAGs start).
pub fn start() {
    START_UPTIME_NS.store(get_awkernel_uptime_timestamp(), Ordering::Relaxed);
}

fn elapsed_ns() -> u64 {
    get_awkernel_uptime_timestamp().saturating_sub(START_UPTIME_NS.load(Ordering::Relaxed))
}

/// Bag stamp -> Awkernel uptime at which it is replayed.
pub fn to_uptime(stamp_ns: u64) -> u64 {
    START_UPTIME_NS
        .load(Ordering::Relaxed)
        .saturating_add(stamp_ns.saturating_sub(BAG_T0_NS.load(Ordering::Relaxed)))
}

/// Advance `cursor` past every due row; return the latest due row's index and whether it was
/// newly reached on this call. `None` before the first row is due.
pub(crate) fn latest_due<T>(
    rows: &[T],
    cursor: &Mutex<usize>,
    stamp_ns: impl Fn(&T) -> u64,
) -> Option<(usize, bool)> {
    let t0 = BAG_T0_NS.load(Ordering::Relaxed);
    let now = elapsed_ns();
    let mut node = MCSNode::new();
    let mut next = cursor.lock(&mut node);
    let before = *next;
    while let Some(row) = rows.get(*next) {
        if stamp_ns(row).saturating_sub(t0) > now {
            break;
        }
        *next += 1;
    }
    next.checked_sub(1).map(|idx| (idx, *next != before))
}

/// The newest pose that became due since the previous call (`/localization/pose_estimator/
/// pose_with_covariance` from the Main ECU), stamped on the Awkernel clock.
pub fn next_pose() -> Option<PoseWithCovarianceStamped> {
    let mut node = MCSNode::new();
    let poses = POSES.lock(&mut node);
    let (idx, is_new) = latest_due(&poses, &POSE_CURSOR, |r| r.stamp_ns)?;
    if !is_new {
        return None;
    }
    let row = poses.get(idx)?;
    Some(PoseWithCovarianceStamped {
        header: common_types::Header {
            frame_id: "map",
            timestamp: to_uptime(row.stamp_ns),
        },
        pose: PoseWithCovariance {
            pose: row.pose,
            covariance: row.covariance,
        },
    })
}

/// The recorded MRM trigger state at the current replay time (off before the first row).
pub fn trigger_state() -> ConstantJerkDecelerationTrigger {
    let mut node = MCSNode::new();
    let triggers = TRIGGERS.lock(&mut node);
    let row = latest_due(&triggers, &TRIGGER_CURSOR, |r| r.stamp_ns)
        .and_then(|(idx, _)| triggers.get(idx).copied());
    match row {
        Some(r) => ConstantJerkDecelerationTrigger {
            stamp: bridge::ns_to_instant(to_uptime(r.stamp_ns)),
            trigger: r.trigger,
            target_acceleration: r.target_acceleration,
            target_jerk: r.target_jerk,
        },
        None => ConstantJerkDecelerationTrigger {
            stamp: Instant::now(),
            ..Default::default()
        },
    }
}

/// Whether every replay stream has run out (logged once by the caller).
pub fn finished() -> bool {
    let imu_len = unsafe { IMU_CSV_DATA.as_ref() }.map_or(0, Vec::len);
    let mut node = MCSNode::new();
    let imu_next = *IMU_CURSOR.lock(&mut node);
    imu_next >= imu_len
}

// ---------------------------------------------------------------------------
// CSV parsing
// ---------------------------------------------------------------------------

fn field<'a>(fields: &[&'a str], i: usize) -> Result<&'a str, &'static str> {
    fields
        .get(i)
        .copied()
        .ok_or("replay CSV has too few columns")
}

fn f64_at(fields: &[&str], i: usize) -> Result<f64, &'static str> {
    parse_f64(field(fields, i)?)
}

fn parse_pose_csv(csv: &str) -> Result<Vec<PoseRow>, &'static str> {
    let mut rows = Vec::new();
    parse_csv_records(csv, |f| {
        let mut covariance = [0.0; 36];
        for (i, c) in covariance.iter_mut().enumerate() {
            *c = f64_at(f, 9 + i)?;
        }
        rows.push(PoseRow {
            stamp_ns: parse_timestamp(field(f, 0)?, field(f, 1)?)?,
            pose: Pose {
                position: Point3D {
                    x: f64_at(f, 2)?,
                    y: f64_at(f, 3)?,
                    z: f64_at(f, 4)?,
                },
                orientation: Quaternion {
                    x: f64_at(f, 5)?,
                    y: f64_at(f, 6)?,
                    z: f64_at(f, 7)?,
                    w: f64_at(f, 8)?,
                },
            },
            covariance,
        });
        Ok(())
    })?;
    Ok(rows)
}

fn parse_trigger_csv(csv: &str) -> Result<Vec<TriggerRow>, &'static str> {
    let mut rows = Vec::new();
    parse_csv_records(csv, |f| {
        rows.push(TriggerRow {
            stamp_ns: parse_timestamp(field(f, 0)?, field(f, 1)?)?,
            trigger: f64_at(f, 2)? != 0.0,
            target_acceleration: f64_at(f, 3)? as f32,
            target_jerk: f64_at(f, 4)? as f32,
        });
        Ok(())
    })?;
    Ok(rows)
}

fn parse_trajectory_csv(csv: &str) -> Result<Vec<TrajectoryPoint>, &'static str> {
    let mut points = Vec::new();
    parse_csv_records(csv, |f| {
        points.push(TrajectoryPoint::new(
            f64_at(f, 0)?,
            f64_at(f, 1)?,
            f64_at(f, 2)?,
            f64_at(f, 6)?, // qw
            f64_at(f, 3)?, // qx
            f64_at(f, 4)?, // qy
            f64_at(f, 5)?, // qz
            f64_at(f, 7)?,
        ));
        Ok(())
    })?;
    Ok(points)
}
