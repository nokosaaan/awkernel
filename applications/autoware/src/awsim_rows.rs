//! Row formats of the AWSIM inputs, shared by `replay` (embedded CSVs, open loop) and `live`
//! (the same rows sent over UDP by the host, closed loop). The column layouts are those
//! `awkernel_script/awsim_bridge/bag_to_awkernel_csv.py` writes.

use alloc::vec::Vec;

use ekf_localizer::{Point3D, Pose, Quaternion};
use localization_types::TrajectoryPoint;

use crate::{nodes::GateInput, parse_csv_records, parse_f64, parse_timestamp};

pub(crate) fn field<'a>(fields: &[&'a str], i: usize) -> Result<&'a str, &'static str> {
    fields
        .get(i)
        .copied()
        .ok_or("AWSIM input row has too few columns")
}

pub(crate) fn f64_at(fields: &[&str], i: usize) -> Result<f64, &'static str> {
    parse_f64(field(fields, i)?)
}

pub(crate) fn stamp_at(fields: &[&str]) -> Result<u64, &'static str> {
    parse_timestamp(field(fields, 0)?, field(fields, 1)?)
}

/// `timestamp_sec,timestamp_nanosec,x,y,z,qx,qy,qz,qw,cov_0..cov_35` (Main ECU NDT output).
pub(crate) fn pose_row(f: &[&str]) -> Result<(u64, Pose, [f64; 36]), &'static str> {
    let mut covariance = [0.0; 36];
    for (i, c) in covariance.iter_mut().enumerate() {
        *c = f64_at(f, 9 + i)?;
    }
    let pose = Pose {
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
    };
    Ok((stamp_at(f)?, pose, covariance))
}

/// `timestamp_sec,timestamp_nanosec,trigger,target_acceleration,target_jerk`.
pub(crate) fn trigger_row(f: &[&str]) -> Result<(u64, bool, f32, f32), &'static str> {
    Ok((
        stamp_at(f)?,
        f64_at(f, 2)? != 0.0,
        f64_at(f, 3)? as f32,
        f64_at(f, 4)? as f32,
    ))
}

/// `timestamp_sec,timestamp_nanosec,engaged,paused`: control_command_gate's selected source is
/// the follower ("main") / the ADAPI pause (`/control/vehicle_cmd_gate/is_paused`).
pub(crate) fn gate_row(f: &[&str]) -> Result<(u64, GateInput), &'static str> {
    Ok((
        stamp_at(f)?,
        GateInput {
            engaged: f64_at(f, 2)? != 0.0,
            paused: f64_at(f, 3)? != 0.0,
        },
    ))
}

/// `x,y,z,qx,qy,qz,qw,longitudinal_velocity_mps` (the fixed route).
pub(crate) fn parse_trajectory_csv(csv: &str) -> Result<Vec<TrajectoryPoint>, &'static str> {
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

/// Set the followed route from the embedded TRAJECTORY CSV.
pub(crate) fn load_trajectory(csv: &str) -> Result<usize, &'static str> {
    let trajectory = parse_trajectory_csv(csv)?;
    if !trajectory.is_empty() && !trajectory_follower::set_trajectory(&trajectory) {
        log::error!(
            "trajectory has {} points (max {}); keeping the built-in one",
            trajectory.len(),
            trajectory_follower::MAX_TRAJECTORY_POINTS
        );
    }
    Ok(trajectory.len())
}
