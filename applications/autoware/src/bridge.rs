//! Conversions between the two message families used in this DAG:
//!
//! - localization front end (`common_types` / `ekf_localizer` / `vehicle_velocity_converter`):
//!   `u64` uptime-nanosecond stamps, `&'static str` frames, `[f64; 36]` row-major covariances.
//! - control back end (`localization_types`): `Instant` stamps, `heapless` frames, nalgebra
//!   0.32 matrices.

use core::time::Duration;
use ekf_localizer::EKFOdometry;
use localization_types::{Header, Instant, KinematicState, PoseWithCovariance, TwistWithCovariance};

pub fn ns_to_instant(ns: u64) -> Instant {
    Instant::zero() + Duration::from_nanos(ns)
}

fn frame(name: &str) -> heapless::String<256> {
    let mut s = heapless::String::new();
    if s.push_str(name).is_err() {
        log::warn!("frame_id longer than 256 bytes is dropped: {name}");
    }
    s
}

fn header(h: &common_types::Header) -> Header {
    Header {
        time_stamp: ns_to_instant(h.timestamp),
        frame_id: frame(h.frame_id),
    }
}

fn covariance(c: &[f64; 36]) -> nalgebra::Matrix6<f64> {
    nalgebra::Matrix6::from_row_slice(c)
}

fn twist(t: &vehicle_velocity_converter::TwistWithCovariance) -> TwistWithCovariance {
    TwistWithCovariance {
        twist: localization_types::Twist {
            linear: nalgebra::Vector3::new(
                t.twist.linear.x,
                t.twist.linear.y,
                t.twist.linear.z,
            ),
            angular: nalgebra::Vector3::new(
                t.twist.angular.x,
                t.twist.angular.y,
                t.twist.angular.z,
            ),
        },
        covariance: covariance(&t.covariance),
    }
}

/// `/localization/pose_twist_fusion_filter/kinematic_state` (nav_msgs/Odometry) ->
/// `KinematicState`. Odometry has no acceleration, so `accel` stays zero.
pub fn odometry_to_kinematic_state(odom: &EKFOdometry) -> KinematicState {
    let p = &odom.pose.pose;
    KinematicState {
        header: header(&odom.header),
        child_frame_id: frame(odom.child_frame_id),
        pose: PoseWithCovariance {
            pose: localization_types::Pose {
                point: nalgebra::Vector3::new(p.position.x, p.position.y, p.position.z),
                orientation: nalgebra::Quaternion::new(
                    p.orientation.w,
                    p.orientation.x,
                    p.orientation.y,
                    p.orientation.z,
                ),
            },
            covariance: covariance(&odom.pose.covariance),
        },
        twist: twist(&odom.twist),
        accel: Default::default(),
    }
}
