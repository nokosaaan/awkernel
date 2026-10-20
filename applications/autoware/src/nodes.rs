//! Node bodies shared by both wirings (`wiring_single` / `wiring_multi`), so that switching
//! between one DAG and per-period DAGs changes only how nodes are connected, never what they
//! compute.

use alloc::{format, string::String, vec::Vec};
#[cfg(not(feature = "replay"))]
use awkernel_lib::delay::wait_microsec;
use awkernel_lib::sync::mutex::{MCSNode, Mutex};
use awkernel_lib::time::Instant;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use core::time::Duration;

use constant_jerk_deceleration_controller::ControllerOutput;
use ekf_localizer::{
    apply_twist_observability_gate, get_or_initialize_default_module, EKFOdometry, EKFParameters,
    Pose, PoseWithCovariance, PoseWithCovarianceStamped,
};
use imu_corrector::{ImuCorrector, ImuWithCovariance, Transform};
use imu_driver::{build_imu_msg_from_csv_row, ImuMsg, TamagawaImuParser};
use localization_types::{
    AccelWithCovarianceStamped, ConstantJerkDecelerationTrigger, Control, GearCommand,
    HazardLightsCommand, KinematicState, TurnIndicatorsCommand,
};
#[cfg(not(feature = "replay"))]
use ndt_scan_matcher::{
    synthetic::World, NdtParams, NdtScanMatcher, PointCloud,
};
use ndt_scan_matcher::{synthetic::PlanarPose, AlignReport};
use vehicle_velocity_converter::{
    build_velocity_report_from_csv_row, reactor_helpers, Twist, TwistWithCovariance,
    TwistWithCovarianceStamped, VehicleVelocityConverter,
};

use crate::shared::Latest;
use crate::{
    bridge, get_awkernel_uptime_timestamp, publish_command_json, publish_odometry_json,
    IMU_CSV_COUNT, IMU_CSV_DATA,
    LOG_ENABLE, VELOCITY_CSV_COUNT, VELOCITY_CSV_DATA,
};

// ---------------------------------------------------------------------------
// Synthetic sensing / scenario constants
// ---------------------------------------------------------------------------

/// pose_initializer stand-in: the `/initialpose` NDT starts from before the EKF has a pose.
#[cfg(not(feature = "replay"))]
pub const INITIAL_POSE: PlanarPose = PlanarPose {
    x: 0.0,
    y: 0.0,
    yaw: 0.0,
};

/// Where the (simulated) vehicle really starts. Offset from `INITIAL_POSE` on purpose, so the
/// first NDT alignment has an initial-pose error to correct.
#[cfg(not(feature = "replay"))]
const GROUND_TRUTH_START: PlanarPose = PlanarPose {
    x: 0.5,
    y: -0.3,
    yaw: 0.03,
};

/// Synthetic LiDAR range and the downsampled cloud size handed to NDT.
#[cfg(not(feature = "replay"))]
const LIDAR_RANGE_M: f64 = 30.0;
#[cfg(not(feature = "replay"))]
const LIDAR_MAX_POINTS: usize = 1_500;

/// MRM trigger stand-in (no `replay`): switch the in-lane stop on this long after the DAGs start.
#[cfg(not(feature = "replay"))]
const MRM_TRIGGER_AFTER: Duration = Duration::from_secs(10);
/// `in_lane_moderate_stop` defaults of autoware_mrm_in_lane_stop_operator (#13332). The TIER IV
/// Pilot.Auto.x2 In Lane Moderate Stop is described as a 0.3G stop.
#[cfg(not(feature = "replay"))]
const MRM_TARGET_ACCELERATION: f32 = -2.5;
#[cfg(not(feature = "replay"))]
const MRM_TARGET_JERK: f32 = -1.5;

/// One pipeline summary line per this interval.
const PIPELINE_LOG_INTERVAL: Duration = Duration::from_secs(1);

// ---------------------------------------------------------------------------
// Node state
// ---------------------------------------------------------------------------

/// Simulated ground truth, dead-reckoned from the same wheel-speed / yaw-rate data the twist
/// estimator sees; the synthetic LiDAR observes the world from here.
static GROUND_TRUTH: Mutex<Option<PlanarPose>> = Mutex::new(None);
#[cfg(not(feature = "replay"))]
static SYNTHETIC_WORLD: Mutex<Option<World>> = Mutex::new(None);

/// `/localization/pose_twist_fusion_filter/biased_pose_with_covariance`. NDT reads it as the
/// initial guess; it is shared state rather than a DAG edge because EKF -> NDT -> EKF is a cycle.
static EKF_BIASED_POSE: Latest<Pose> = Latest::new();
static EKF_INITIALIZED: Mutex<bool> = Mutex::new(false);
/// (`last_predict_time_sec_` [ns], `ekf_dt_` [s]) for `EkfDt::Measured`.
#[cfg(feature = "dag_multi")]
static EKF_TIMING: Mutex<(Option<u64>, f64)> = Mutex::new((None, 0.0));

/// How the EKF prediction step is sized.
#[derive(Clone, Copy)]
pub enum EkfDt {
    /// Advance by a fixed step every cycle. Used by the single DAG, where every cycle carries
    /// exactly one source period's worth of data, so EKF time follows data time.
    #[cfg(not(feature = "dag_multi"))]
    Fixed(f64),
    /// Upstream `timer_callback`: dt is the measured time since the previous cycle (capped at
    /// 10 s, kept unchanged on a backwards jump), and `accumulate_delay_time` runs only once a
    /// previous cycle exists. `nominal` is the step used before the first measured interval.
    #[cfg(feature = "dag_multi")]
    Measured { nominal: f64 },
}

/// Returns `(dt, accumulate)` for this cycle.
fn ekf_step(mode: EkfDt, now_ns: u64) -> (f64, bool) {
    match mode {
        #[cfg(not(feature = "dag_multi"))]
        EkfDt::Fixed(dt) => {
            let _ = now_ns; // only the measured mode reads the clock
            (dt, true)
        }
        #[cfg(feature = "dag_multi")]
        EkfDt::Measured { nominal } => {
            let mut node = MCSNode::new();
            let mut timing = EKF_TIMING.lock(&mut node);
            let (last, prev_dt) = *timing;
            let prev_dt = if prev_dt > 0.0 { prev_dt } else { nominal };
            let step = match last {
                Some(last_ns) if now_ns >= last_ns => {
                    let dt = ((now_ns - last_ns) as f64 / 1.0e9).min(10.0);
                    (dt, true)
                }
                Some(_) => {
                    log::warn!("[EKF] Detected jump back in time");
                    (prev_dt, false)
                }
                None => (prev_dt, false),
            };
            *timing = (Some(now_ns), step.0);
            step
        }
    }
}

#[cfg(not(feature = "replay"))]
static NDT: Mutex<Option<NdtScanMatcher>> = Mutex::new(None);
static NDT_LAST_REPORT: Latest<AlignReport> = Latest::new();
static NDT_PUBLISHED: AtomicU32 = AtomicU32::new(0);
static NDT_SKIPPED: AtomicU32 = AtomicU32::new(0);

/// Execution times [us] of the synthetic scan and of NDT (`exe_time_ms` debug topic upstream).
static SCAN_GEN_TIME_US: AtomicU32 = AtomicU32::new(0);
static NDT_EXE_TIME_US: AtomicU32 = AtomicU32::new(0);
/// Input age [us] (consumer time minus sensor stamp) at the EKF and at the controller.
static EKF_TWIST_AGE_US: AtomicU32 = AtomicU32::new(0);
static CONTROL_INPUT_AGE_US: AtomicU32 = AtomicU32::new(0);

static DAG_START: Mutex<Option<Instant>> = Mutex::new(None);

/// Main ECU localization while the relay is cut: the newest pose that reached the relay. It keeps
/// coming during MRM (the Main ECU NDT keeps running), and is the reference the dead-reckoning
/// error is measured against, as in the MRM v0.7 evaluation.
static REFERENCE_POSE: Latest<Pose> = Latest::new();
static RELAY_CUT_POSES: AtomicU32 = AtomicU32::new(0);
static MRM_STARTED_AT: Mutex<Option<Instant>> = Mutex::new(None);
/// Newest stamps already fused by the EKF; repeated samples (replay ticks without a new row)
/// are skipped so that a measurement is never fused twice.
static LAST_FUSED_TWIST_NS: AtomicU64 = AtomicU64::new(0);
static LAST_FUSED_POSE_NS: AtomicU64 = AtomicU64::new(0);
static COMMAND_SEQ: AtomicU32 = AtomicU32::new(0);
static LAST_PIPELINE_LOG: Mutex<Option<Instant>> = Mutex::new(None);

#[cfg(not(feature = "replay"))]
fn elapsed_us(started: Instant) -> u32 {
    u32::try_from(started.elapsed().as_micros()).unwrap_or(u32::MAX)
}

/// Age of a sensor stamp (uptime ns) at the time of the call, in microseconds.
fn age_us(stamp_ns: u64) -> u32 {
    let age_ns = get_awkernel_uptime_timestamp().saturating_sub(stamp_ns);
    u32::try_from(age_ns / 1_000).unwrap_or(u32::MAX)
}

// ---------------------------------------------------------------------------
// Configuration (node constructors / parameters)
// ---------------------------------------------------------------------------

pub fn configure_nodes() {
    // twist2accel subscribes to the twist estimator output in this pipeline (see the node diagram).
    twist2accel::configure(false);

    // Map loading happens before the pose estimator is activated upstream (pose_initializer
    // triggers NDT only once the map is in); doing it here keeps the first scan off the map
    // build, which would otherwise back up the whole DAG for several periods.
    #[cfg(not(feature = "replay"))]
    {
    match NdtScanMatcher::new(NdtParams::default()) {
        Ok(mut ndt) => {
            if let Err(e) = ndt.update_map(INITIAL_POSE.x, INITIAL_POSE.y) {
                log::error!("ndt_scan_matcher: initial map load failed: {:?}", e);
            }
            log::info!("ndt_scan_matcher: {} map tiles loaded", ndt.loaded_tile_count());
            let mut node = MCSNode::new();
            *NDT.lock(&mut node) = Some(ndt);
        }
        Err(e) => log::error!("ndt_scan_matcher: engine init failed: {:?}", e),
    }
    }

    match nominal_filter_params() {
        Some(params) => command_gate::configure(params, true),
        None => log::error!("control_command_gate: invalid filter table; gate left as pass-through"),
    }
}

/// autoware_control_command_gate default.param.yaml `nominal_filter`;
/// wheel_base is the sample_vehicle value (vehicle_info.param.yaml).
fn nominal_filter_params() -> Option<command_gate::FilterParams> {
    let points = |v: &[f64]| heapless::Vec::<f64, 16>::from_slice(v).ok();
    Some(command_gate::FilterParams {
        wheel_base: 2.79,
        vel_lim: 25.0,
        reference_speed_points: points(&[0.1, 0.3, 20.0, 30.0])?,
        lon_acc_lim_for_lon_vel: points(&[5.0, 5.0, 5.0, 4.0])?,
        lon_jerk_lim_for_lon_acc: points(&[80.0, 5.0, 5.0, 4.0])?,
        lat_acc_lim_for_steer_cmd: points(&[5.0, 5.0, 5.0, 4.0])?,
        lat_jerk_lim_for_steer_cmd: points(&[7.0, 7.0, 7.0, 6.0])?,
        steer_cmd_lim: points(&[1.0, 1.0, 1.0, 0.8])?,
        steer_rate_lim_for_steer_cmd: points(&[1.0, 1.0, 1.0, 0.8])?,
        steer_cmd_diff_lim_from_current_steer: points(&[1.0, 1.0, 1.0, 0.8])?,
        lat_jerk_lim_for_steer_rate: 10.0,
    })
}

/// Mark the start of the scenario (MRM trigger timing is relative to this).
pub fn mark_dag_start() {
    let mut node = MCSNode::new();
    *DAG_START.lock(&mut node) = Some(Instant::now());
    #[cfg(feature = "replay")]
    crate::replay::start();
}

// ---------------------------------------------------------------------------
// Sensing / twist estimator
// ---------------------------------------------------------------------------

pub fn imu_driver() -> ImuMsg {
    let mut node = MCSNode::new();
    let mut count_guard = IMU_CSV_COUNT.lock(&mut node);
    let count = *count_guard;
    let data = unsafe { IMU_CSV_DATA.as_ref() };
    let awkernel_timestamp = get_awkernel_uptime_timestamp();

    // Replay: the newest due row, stamped on the replay timeline. Otherwise the next row.
    #[cfg(feature = "replay")]
    let picked = data.and_then(|csv| {
        let (idx, _) = crate::replay::latest_due(csv, &crate::replay::IMU_CURSOR, |r| r.timestamp)
            .unwrap_or((0, false));
        csv.get(idx).map(|row| (row, crate::replay::to_uptime(row.timestamp)))
    });
    #[cfg(not(feature = "replay"))]
    let picked = data
        .and_then(|csv| csv.get(count % csv.len().max(1)))
        .map(|row| (row, awkernel_timestamp));

    let imu_msg = match picked {
        Some((row, stamp)) => build_imu_msg_from_csv_row(row, "imu_link", stamp),
        None => {
            // Fallback: generate dummy IMU data when no CSV is embedded.
            let mut parser = TamagawaImuParser::new("imu_link");
            let static_dummy_data = parser.generate_static_dummy_data(awkernel_timestamp);
            parser
                .parse_binary_data(&static_dummy_data, awkernel_timestamp)
                .unwrap_or_default()
        }
    };

    *count_guard += 1;
    #[cfg(not(feature = "replay"))]
    if *count_guard >= 5700 {
        *count_guard = 0;
        log::info!("rust_e2e_app: finish csv for IMU");
        loop {
            wait_microsec(1000);
        }
    }
    #[cfg(feature = "replay")]
    if crate::replay::finished() && *count_guard != usize::MAX {
        // Holds the last sample from here on; log the end once.
        *count_guard = usize::MAX;
        log::info!("replay: IMU data finished");
    }

    if LOG_ENABLE {
        log::debug!(
            "IMU data in imu_driver_node,num={}, timestamp={}",
            count,
            imu_msg.header.timestamp
        );
    }
    imu_msg
}

/// Converts the next velocity report and advances the synthetic ground truth by `period_sec`
/// with it (the truth moves at the velocity data's own cadence, independent of the LiDAR rate).
pub fn vehicle_velocity_converter(period_sec: f64) -> TwistWithCovarianceStamped {
    let converter = VehicleVelocityConverter::default();

    let mut node = MCSNode::new();
    let mut count_guard = VELOCITY_CSV_COUNT.lock(&mut node);
    #[cfg(not(feature = "replay"))]
    let count = *count_guard;
    let data = unsafe { VELOCITY_CSV_DATA.as_ref() };
    let awkernel_timestamp = get_awkernel_uptime_timestamp();

    #[cfg(feature = "replay")]
    let picked = data.and_then(|csv| {
        let (idx, _) =
            crate::replay::latest_due(csv, &crate::replay::VELOCITY_CURSOR, |r| r.timestamp)
                .unwrap_or((0, false));
        csv.get(idx).map(|row| (row, crate::replay::to_uptime(row.timestamp)))
    });
    #[cfg(not(feature = "replay"))]
    let picked = data
        .and_then(|csv| csv.get(count % csv.len().max(1)))
        .map(|row| (row, awkernel_timestamp));

    let twist_msg = match picked {
        Some((row, stamp)) => {
            let report = build_velocity_report_from_csv_row(row, "base_link", stamp);
            converter.convert_velocity_report(&report)
        }
        None => {
            // Fallback: constant 1 m/s when no CSV is embedded.
            let dummy_report = vehicle_velocity_converter::VelocityReport {
                header: common_types::Header {
                    frame_id: "base_link",
                    timestamp: awkernel_timestamp,
                },
                longitudinal_velocity: 1.0,
                lateral_velocity: 0.0,
                heading_rate: 0.0,
            };
            converter.convert_velocity_report(&dummy_report)
        }
    };

    // Replay has no simulated truth (AWSIM's ground truth is recorded on the host side instead).
    #[cfg(not(feature = "replay"))]
    advance_ground_truth(
        twist_msg.twist.twist.linear.x,
        twist_msg.twist.twist.angular.z,
        period_sec,
    );
    #[cfg(feature = "replay")]
    let _ = period_sec; // only the synthetic truth integrates by the period

    *count_guard += 1;
    #[cfg(not(feature = "replay"))]
    if *count_guard >= 5700 {
        *count_guard = 0;
        log::info!("rust_e2e_app: finish csv for Velocity");
        loop {
            wait_microsec(1000);
        }
    }

    if LOG_ENABLE {
        log::debug!("Vehicle velocity converter: Converted velocity report to twist - linear.x={:.3}, angular.z={:.3}, awkernel_timestamp={}",
            twist_msg.twist.twist.linear.x,
            twist_msg.twist.twist.angular.z,
            twist_msg.header.timestamp
        );
    }
    twist_msg
}

#[cfg(not(feature = "replay"))]
fn advance_ground_truth(vx: f64, wz: f64, dt: f64) {
    let mut node = MCSNode::new();
    let mut gt = GROUND_TRUTH.lock(&mut node);
    *gt = Some(match *gt {
        None => GROUND_TRUTH_START,
        Some(p) => PlanarPose {
            x: p.x + vx * libm::cos(p.yaw) * dt,
            y: p.y + vx * libm::sin(p.yaw) * dt,
            yaw: p.yaw + wz * dt,
        },
    });
}

pub fn imu_corrector(imu_msg: &ImuMsg) -> ImuWithCovariance {
    ImuCorrector::new().correct_imu_with_covariance(imu_msg, None)
}

pub fn gyro_odometer(
    imu_with_cov: ImuWithCovariance,
    vehicle_twist: TwistWithCovarianceStamped,
) -> TwistWithCovarianceStamped {
    let current_timestamp = imu_with_cov.header.timestamp;
    let current_time = get_awkernel_uptime_timestamp();

    let gyro_odometer = match gyro_odometer::get_or_initialize() {
        Ok(core) => core,
        Err(_) => return reactor_helpers::create_empty_twist(current_timestamp),
    };

    gyro_odometer.add_vehicle_twist(vehicle_twist);
    gyro_odometer.add_imu(imu_with_cov);

    match gyro_odometer.process_and_get_result(current_time) {
        Some(result) => gyro_odometer.process_result(result),
        None => reactor_helpers::create_empty_twist(current_timestamp),
    }
}

pub fn twist2accel(twist: &TwistWithCovarianceStamped) -> AccelWithCovarianceStamped {
    let (header, twist) = bridge::twist_to_localization(twist);
    // `use_odom = false` is set in `configure_nodes`, so `on_twist` always yields.
    twist2accel::on_twist(&header, &twist).unwrap_or_else(|| {
        log::warn!("twist2accel: configured for odometry input; publishing zero accel");
        AccelWithCovarianceStamped {
            header,
            ..Default::default()
        }
    })
}

// ---------------------------------------------------------------------------
// Pose estimator
// ---------------------------------------------------------------------------

/// Synthetic downsampled LiDAR scan taken at the current ground truth.
#[cfg(not(feature = "replay"))]
pub fn pointcloud_dummy_generator() -> PointCloud {
    let truth = {
        let mut node = MCSNode::new();
        let truth = *GROUND_TRUTH.lock(&mut node);
        truth
    }
    .unwrap_or(GROUND_TRUTH_START);

    let started = Instant::now();
    let points = {
        let mut node = MCSNode::new();
        let mut world = SYNTHETIC_WORLD.lock(&mut node);
        world
            .get_or_insert_with(World::new)
            .scan(truth, LIDAR_RANGE_M, LIDAR_MAX_POINTS)
    };
    SCAN_GEN_TIME_US.store(elapsed_us(started), Ordering::Relaxed);

    PointCloud {
        header: common_types::Header {
            frame_id: "base_link",
            timestamp: get_awkernel_uptime_timestamp(),
        },
        points,
    }
}

/// Newest NDT result and the EKF pose of the same scan. The NDT plays the Main ECU, whose own
/// EKF keeps fusing NDT during the MRM; the Sub ECU EKF here is dead reckoning by then and must
/// not steer the reference.
#[cfg(not(feature = "replay"))]
static NDT_GUESS_ANCHOR: Latest<(Pose, Pose)> = Latest::new();

/// NDT initial guess. Normally the EKF biased pose, as upstream. While the relay is cut, the last
/// NDT result moved by the EKF displacement since that scan (about one LiDAR period of odometry),
/// which stands in for the Main ECU EKF that keeps fusing NDT.
#[cfg(not(feature = "replay"))]
fn ndt_initial_guess(ekf_pose: Option<Pose>) -> Option<Pose> {
    let ekf_now = ekf_pose?;
    if !mrm_active() {
        return Some(ekf_now);
    }
    let Some((ndt_then, ekf_then)) = NDT_GUESS_ANCHOR.get() else {
        return Some(ekf_now);
    };
    let mut guess = ndt_then;
    guess.position.x += ekf_now.position.x - ekf_then.position.x;
    guess.position.y += ekf_now.position.y - ekf_then.position.y;
    guess.position.z += ekf_now.position.z - ekf_then.position.z;
    Some(guess)
}

/// One `callback_sensor_points`. `None` when the scan produced no pose (not converged / gated),
/// which is "no message on the topic" upstream.
#[cfg(not(feature = "replay"))]
pub fn ndt_scan_matcher(cloud: &PointCloud) -> Option<PoseWithCovarianceStamped> {
    let ekf_pose = EKF_BIASED_POSE.get();
    let initial_pose = ndt_initial_guess(ekf_pose).unwrap_or_else(|| {
        ndt_scan_matcher::planar_pose(INITIAL_POSE.x, INITIAL_POSE.y, INITIAL_POSE.yaw)
    });

    let started = Instant::now();
    let mut node = MCSNode::new();
    let mut ndt_guard = NDT.lock(&mut node);
    if ndt_guard.is_none() {
        match NdtScanMatcher::new(NdtParams::default()) {
            Ok(m) => *ndt_guard = Some(m),
            Err(e) => {
                log::error!("ndt_scan_matcher: engine init failed: {:?}", e);
                return None;
            }
        }
    }
    let ndt = Option::as_mut(&mut *ndt_guard)?;

    // Dynamic map loading around the current estimate (upstream runs this on a timer; it is a
    // no-op until the vehicle has moved `update_distance`).
    if let Err(e) = ndt.update_map(initial_pose.position.x, initial_pose.position.y) {
        log::warn!("ndt_scan_matcher: map update failed: {:?}", e);
    }

    let result = match ndt.align(cloud, &initial_pose) {
        Ok(Ok(pose)) => {
            NDT_PUBLISHED.fetch_add(1, Ordering::Relaxed);
            if let Some(ekf) = ekf_pose {
                NDT_GUESS_ANCHOR.set((pose.pose.pose, ekf));
            }
            Some(pose)
        }
        Ok(Err(skip)) => {
            NDT_SKIPPED.fetch_add(1, Ordering::Relaxed);
            if LOG_ENABLE {
                log::debug!("ndt_scan_matcher: skipped ({:?})", skip);
            }
            None
        }
        Err(e) => {
            NDT_SKIPPED.fetch_add(1, Ordering::Relaxed);
            log::warn!("ndt_scan_matcher: align failed: {:?}", e);
            None
        }
    };
    let report = ndt.last_report();
    drop(ndt_guard);
    NDT_EXE_TIME_US.store(elapsed_us(started), Ordering::Relaxed);
    if let Some(r) = report {
        NDT_LAST_REPORT.set(r);
    }
    result
}

// ---------------------------------------------------------------------------
// Pose / twist fusion
// ---------------------------------------------------------------------------

/// One EKF timer callback: predict (step sized by `dt_mode`), then fuse every queued pose and
/// twist measurement, in upstream `timer_callback` order.
///
/// The first pose seeds the filter, standing in for upstream's `initialize()` from
/// `/initialpose3d`; later poses feed `measurement_update_pose`. No TF listener is wired in
/// yet, so the transform is the identity (see `EKFModule::initialize`'s doc comment). Until
/// then the filter is not run, as upstream's timer returns early while uninitialized.
pub fn ekf_localizer(
    dt_mode: EkfDt,
    poses: &[PoseWithCovarianceStamped],
    twists: &mut [TwistWithCovarianceStamped],
) -> EKFOdometry {
    let ekf = get_or_initialize_default_module();
    let current_time = get_awkernel_uptime_timestamp();

    let (initialized, seeded_from) = {
        let mut node = MCSNode::new();
        let mut initialized = EKF_INITIALIZED.lock(&mut node);
        let mut seeded_from = None;
        if !*initialized {
            if let Some(first) = poses.first() {
                ekf.initialize(first, &Transform::identity());
                LAST_FUSED_POSE_NS.store(first.header.timestamp, Ordering::Relaxed);
                *initialized = true;
                seeded_from = Some(0usize);
            }
        }
        (*initialized, seeded_from)
    };

    if initialized {
        // Both calls are required every predict tick -- `predict_with_delay` does not age the
        // delay-time buffer itself (see its doc comment).
        let (dt, accumulate) = ekf_step(dt_mode, current_time);
        if accumulate {
            ekf.accumulate_delay_time(dt);
        }
        ekf.predict_with_delay(dt);

        for (i, pose) in poses.iter().enumerate() {
            if seeded_from != Some(i) && take_new(&LAST_FUSED_POSE_NS, pose.header.timestamp) {
                ekf.measurement_update_pose(pose, current_time);
            }
        }

        // Below `threshold_observable_velocity_mps` (0.0 by default, matching upstream's
        // shipped config), vx is inflated as unreliable before fusing; wiring the gate
        // unconditionally here matches upstream's `callback_twist_with_covariance`, even
        // though the default threshold makes it a no-op today.
        let threshold = EKFParameters::default().threshold_observable_velocity_mps;
        for twist in twists.iter_mut() {
            if !take_new(&LAST_FUSED_TWIST_NS, twist.header.timestamp) {
                continue;
            }
            apply_twist_observability_gate(twist, threshold);
            ekf.measurement_update_twist(twist, current_time);
        }

        EKF_BIASED_POSE.set(ekf.get_current_pose(true, current_time).pose);
    }

    let stamp = twists
        .last()
        .map_or(current_time, |t| t.header.timestamp);
    if let Some(t) = twists.last() {
        EKF_TWIST_AGE_US.store(age_us(t.header.timestamp), Ordering::Relaxed);
    }

    let ekf_pose = ekf.get_current_pose(false, current_time);
    let ekf_twist = ekf.get_current_twist(current_time);
    EKFOdometry {
        header: common_types::Header {
            frame_id: "map",
            timestamp: stamp,
        },
        child_frame_id: "base_link",
        pose: PoseWithCovariance {
            pose: ekf_pose.pose,
            covariance: ekf.get_current_pose_covariance(),
        },
        twist: TwistWithCovariance {
            twist: Twist {
                linear: common_types::Vector3::new(
                    ekf_twist.twist.linear.x,
                    ekf_twist.twist.linear.y,
                    ekf_twist.twist.linear.z,
                ),
                angular: common_types::Vector3::new(
                    ekf_twist.twist.angular.x,
                    ekf_twist.twist.angular.y,
                    ekf_twist.twist.angular.z,
                ),
            },
            covariance: ekf.get_current_twist_covariance(),
        },
    }
}

/// `true` (and record `stamp_ns`) when it is newer than everything fused so far.
fn take_new(last: &AtomicU64, stamp_ns: u64) -> bool {
    if stamp_ns > last.load(Ordering::Relaxed) {
        last.store(stamp_ns, Ordering::Relaxed);
        true
    } else {
        false
    }
}

pub fn stop_filter(odom: &EKFOdometry) -> KinematicState {
    stop_filter::apply(
        bridge::odometry_to_kinematic_state(odom),
        stop_filter::DEFAULT_VX_THRESHOLD,
        stop_filter::DEFAULT_WZ_THRESHOLD,
    )
}

// ---------------------------------------------------------------------------
// Control
// ---------------------------------------------------------------------------

pub fn trajectory_follower(ks: &KinematicState, accel: &AccelWithCovarianceStamped) -> Control {
    let stamp_ns = u64::try_from(ks.header.time_stamp.uptime().as_nanos()).unwrap_or(u64::MAX);
    CONTROL_INPUT_AGE_US.store(age_us(stamp_ns), Ordering::Relaxed);
    trajectory_follower::follow(ks, accel)
}

/// MRM trigger source: the recorded trigger with `replay`, otherwise a dummy that fires
/// `MRM_TRIGGER_AFTER` after start (stands in for SubEcuInLaneModerateStopSwitcher /
/// mrm_in_lane_stop_operator).
pub fn mrm_trigger() -> ConstantJerkDecelerationTrigger {
    #[cfg(feature = "replay")]
    let trigger = crate::replay::trigger_state();
    #[cfg(not(feature = "replay"))]
    let trigger = {
        let started = {
            let mut node = MCSNode::new();
            let started = *DAG_START.lock(&mut node);
            started
        };
        ConstantJerkDecelerationTrigger {
            stamp: Instant::now(),
            trigger: started.is_some_and(|s| s.elapsed() >= MRM_TRIGGER_AFTER),
            target_acceleration: MRM_TARGET_ACCELERATION,
            target_jerk: MRM_TARGET_JERK,
        }
    };
    if trigger.trigger {
        let mut node = MCSNode::new();
        let mut started = MRM_STARTED_AT.lock(&mut node);
        if started.is_none() {
            *started = Some(Instant::now());
            log::info!("MRM: in-lane stop triggered; pose relay cut (dead reckoning starts)");
        }
    }
    trigger
}

fn mrm_active() -> bool {
    mrm_trigger().trigger
}

/// topic_relay_controller for `pose_with_covariance` (Main ECU -> MRM side). While the MRM is
/// active the relay is cut, so the EKF predicts from twist only: that is the dead reckoning
/// (MRM構成説明資料 "ekf_localizerノードにpose_with_covarianceのinputを途絶させる"). The pose
/// that arrives is still kept as the Main ECU reference for the dead-reckoning error.
pub fn pose_relay(pose: Option<PoseWithCovarianceStamped>) -> Option<PoseWithCovarianceStamped> {
    if let Some(p) = &pose {
        REFERENCE_POSE.set(p.pose.pose);
    }
    if mrm_active() {
        if pose.is_some() {
            RELAY_CUT_POSES.fetch_add(1, Ordering::Relaxed);
        }
        return None;
    }
    pose
}

/// Replay pose source: the recorded Main ECU NDT output, in place of the synthetic LiDAR + NDT.
#[cfg(feature = "replay")]
pub fn pose_replay() -> Option<PoseWithCovarianceStamped> {
    crate::replay::next_pose()
}

/// Upstream has two independent callbacks; here both inputs arrive per period, and the trigger
/// is applied first so it acts on this period's control.
pub fn constant_jerk_deceleration_controller(
    control: &Control,
    trigger: &ConstantJerkDecelerationTrigger,
) -> ControllerOutput {
    constant_jerk_deceleration_controller::on_trigger(trigger);
    constant_jerk_deceleration_controller::on_control(control)
}

/// With a single command source the CommandSelector is a pass-through, so only the Control
/// goes through the limit filter; gear / lights are forwarded as selected.
pub fn control_command_gate(control: &Control, ks: &KinematicState) -> Control {
    command_gate::filter(control, ks)
}

/// Vehicle interface stand-in: periodic summary log plus the EKF odometry JSON for UDP export.
pub fn vehicle_cmd_sink(
    mode: &str,
    control: &Control,
    gear: &GearCommand,
    hazard_lights: &HazardLightsCommand,
    turn_indicators: &TurnIndicatorsCommand,
    ekf_odom: Option<&EKFOdometry>,
    queue_drops: Option<(u32, u32)>,
) {
    let due = {
        let mut node = MCSNode::new();
        let mut last = LAST_PIPELINE_LOG.lock(&mut node);
        let due = last.is_none_or(|t| t.elapsed() >= PIPELINE_LOG_INTERVAL);
        if due {
            *last = Some(Instant::now());
        }
        due
    };
    if due {
        log_pipeline(mode, control, gear, hazard_lights, turn_indicators, ekf_odom, queue_drops);
    }
    publish_command_json(command_json(control, gear, hazard_lights, turn_indicators));
    if let Some(odom) = ekf_odom {
        publish_odometry_json(odometry_json(odom));
    }
}

/// One command packet (UDP port `UDP_CMD_DST_PORT`) for the AWSIM bridge: the four
/// `/control/command/*` messages of one sink period. The bridge restamps them on the ROS clock.
fn command_json(
    control: &Control,
    gear: &GearCommand,
    hazard_lights: &HazardLightsCommand,
    turn_indicators: &TurnIndicatorsCommand,
) -> String {
    let stamp_ns = |t: Instant| u64::try_from(t.uptime().as_nanos()).unwrap_or(u64::MAX);
    format!(
        r#"{{"type":"command","seq":{},"stamp_ns":{},"mrm":{},"control":{{"velocity":{:.6},"acceleration":{:.6},"jerk":{:.6},"is_defined_acceleration":{},"is_defined_jerk":{},"steering_tire_angle":{:.6},"steering_tire_rotation_rate":{:.6},"is_defined_steering_tire_rotation_rate":{}}},"gear":{},"hazard_lights":{},"turn_indicators":{}}}"#,
        COMMAND_SEQ.fetch_add(1, Ordering::Relaxed),
        stamp_ns(control.stamp),
        mrm_active(),
        control.longitudinal.velocity,
        control.longitudinal.acceleration,
        control.longitudinal.jerk,
        control.longitudinal.is_defined_acceleration,
        control.longitudinal.is_defined_jerk,
        control.lateral.steering_tire_angle,
        control.lateral.steering_tire_rotation_rate,
        control.lateral.is_defined_steering_tire_rotation_rate,
        gear.command,
        hazard_lights.command,
        turn_indicators.command,
    )
}

fn odometry_json(ekf_odom: &EKFOdometry) -> String {
    format!(
        r#"{{"header":{{"frame_id":"{}","timestamp":{}}},"child_frame_id":"{}","pose":{{"pose":{{"position":{{"x":{:.6},"y":{:.6},"z":{:.6}}},"orientation":{{"x":{:.6},"y":{:.6},"z":{:.6},"w":{:.6}}}}},"covariance":[{}]}},"twist":{{"twist":{{"linear":{{"x":{:.6},"y":{:.6},"z":{:.6}}},"angular":{{"x":{:.6},"y":{:.6},"z":{:.6}}}}},"covariance":[{}]}}}}"#,
        ekf_odom.header.frame_id,
        ekf_odom.header.timestamp,
        ekf_odom.child_frame_id,
        ekf_odom.pose.pose.position.x,
        ekf_odom.pose.pose.position.y,
        ekf_odom.pose.pose.position.z,
        ekf_odom.pose.pose.orientation.x,
        ekf_odom.pose.pose.orientation.y,
        ekf_odom.pose.pose.orientation.z,
        ekf_odom.pose.pose.orientation.w,
        ekf_odom.pose.covariance.iter().map(|&x| format!("{:.6}", x)).collect::<Vec<_>>().join(","),
        ekf_odom.twist.twist.linear.x,
        ekf_odom.twist.twist.linear.y,
        ekf_odom.twist.twist.linear.z,
        ekf_odom.twist.twist.angular.x,
        ekf_odom.twist.twist.angular.y,
        ekf_odom.twist.twist.angular.z,
        ekf_odom.twist.covariance.iter().map(|&x| format!("{:.6}", x)).collect::<Vec<_>>().join(",")
    )
}

fn yaw_from_quaternion(x: f64, y: f64, z: f64, w: f64) -> f64 {
    libm::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
}

/// One summary line of the whole pipeline (truth / NDT / EKF / commands) for serial-log checks.
fn log_pipeline(
    mode: &str,
    control: &Control,
    gear: &GearCommand,
    hazard_lights: &HazardLightsCommand,
    turn_indicators: &TurnIndicatorsCommand,
    ekf_odom: Option<&EKFOdometry>,
    queue_drops: Option<(u32, u32)>,
) {
    let truth = {
        let mut node = MCSNode::new();
        let truth = *GROUND_TRUTH.lock(&mut node);
        truth
    };
    let (tx, ty, tyaw) = truth.map_or((f64::NAN, f64::NAN, f64::NAN), |t| (t.x, t.y, t.yaw));
    let (iters, nvtl, converged) = NDT_LAST_REPORT.get().map_or((-1, f32::NAN, false), |r| {
        (r.iteration_num, r.nearest_voxel_likelihood, r.converged)
    });
    // Dead-reckoning error against the Main ECU reference (Euclidean, as in MRM v0.7).
    let mrm_s = {
        let mut node = MCSNode::new();
        let started = *MRM_STARTED_AT.lock(&mut node);
        started
    }
    .map_or(f64::NAN, |t| t.elapsed().as_secs_f64());
    let dr_ref_err = match (ekf_odom, REFERENCE_POSE.get()) {
        (Some(o), Some(r)) => {
            let dx = o.pose.pose.position.x - r.position.x;
            let dy = o.pose.pose.position.y - r.position.y;
            libm::sqrt(dx * dx + dy * dy)
        }
        _ => f64::NAN,
    };
    let (ex, ey, eyaw) = ekf_odom.map_or((f64::NAN, f64::NAN, f64::NAN), |o| {
        let p = &o.pose.pose;
        (
            p.position.x,
            p.position.y,
            yaw_from_quaternion(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w),
        )
    });
    log::info!(
        "[pipeline:{}] t={}ms truth=({:.3},{:.3},{:.4}) scan_us={} ndt(iter={},nvtl={:.3},conv={},pub={},skip={},us={}) ekf=({:.3},{:.3},{:.4}) err=({:.3},{:.3}) age_us(twist->ekf={},sensor->ctrl={}) cmd(v={:.3},a={:.3},j={:.3},steer={:.4}) gear={} hazard={} turn={} mrm={} mrm_s={:.2} relay_cut={} ref_err={:.3} drops(twist,pose)={:?}",
        mode,
        get_awkernel_uptime_timestamp() / 1_000_000,
        tx,
        ty,
        tyaw,
        SCAN_GEN_TIME_US.load(Ordering::Relaxed),
        iters,
        nvtl,
        converged,
        NDT_PUBLISHED.load(Ordering::Relaxed),
        NDT_SKIPPED.load(Ordering::Relaxed),
        NDT_EXE_TIME_US.load(Ordering::Relaxed),
        ex,
        ey,
        eyaw,
        ex - tx,
        ey - ty,
        EKF_TWIST_AGE_US.load(Ordering::Relaxed),
        CONTROL_INPUT_AGE_US.load(Ordering::Relaxed),
        control.longitudinal.velocity,
        control.longitudinal.acceleration,
        control.longitudinal.jerk,
        control.lateral.steering_tire_angle,
        gear.command,
        hazard_lights.command,
        turn_indicators.command,
        mrm_active(),
        mrm_s,
        RELAY_CUT_POSES.load(Ordering::Relaxed),
        dr_ref_err,
        queue_drops,
    );
}
