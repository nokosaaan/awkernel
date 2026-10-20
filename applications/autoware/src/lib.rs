#![no_std]
#![allow(static_mut_refs)]
extern crate alloc;

mod bridge;

use alloc::{borrow::Cow, format, string::String, vec, vec::Vec};
use awkernel_async_lib::dag::{create_dag, finish_create_dags};
use awkernel_async_lib::net::IpAddr;
use awkernel_async_lib::scheduler::SchedulerType;
use awkernel_lib::delay::wait_microsec;
use awkernel_lib::sync::mutex::{MCSNode, Mutex};
use core::net::Ipv4Addr;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use core::time::Duration;
use csv_core::{ReadRecordResult, Reader};

pub use common_types::Header;
use constant_jerk_deceleration_controller::ControllerOutput;
use core::slice;
use ekf_localizer::{
    apply_twist_observability_gate, get_or_initialize_default_module, EKFOdometry, EKFParameters,
    Pose, PoseWithCovariance, PoseWithCovarianceStamped,
};
use imu_corrector::{ImuCorrector, ImuWithCovariance, Transform};
use imu_driver::{build_imu_msg_from_csv_row, ImuCsvRow, ImuMsg, TamagawaImuParser};
use localization_types::{
    AccelWithCovarianceStamped, ConstantJerkDecelerationTrigger, Control, GearCommand,
    HazardLightsCommand, KinematicState, TurnIndicatorsCommand,
};
use ndt_scan_matcher::{
    synthetic::{PlanarPose, World},
    AlignReport, NdtParams, NdtScanMatcher, PointCloud,
};
use vehicle_velocity_converter::{
    build_velocity_report_from_csv_row, reactor_helpers, Twist, TwistWithCovariance,
    TwistWithCovarianceStamped, VehicleVelocityConverter, VelocityCsvRow,
};

const LOG_ENABLE: bool = false;

/// Print one pipeline summary line every N source periods (20 x 50 ms = 1 s).
const PIPELINE_LOG_EVERY: u32 = 20;

const INTERFACE_ID: u64 = 0;
const INTERFACE_ADDR: Ipv4Addr = Ipv4Addr::new(10, 0, 2, 64);
const UDP_TCP_DST_ADDR: Ipv4Addr = Ipv4Addr::new(10, 0, 2, 2);
const UDP_DST_PORT: u16 = 26099;

static mut LATEST_JSON_DATA: Option<String> = None;
static JSON_DATA_READY: AtomicBool = AtomicBool::new(false);
static JSON_DATA_LENGTH: AtomicUsize = AtomicUsize::new(0);

include!(concat!(env!("OUT_DIR"), "/csv_data.rs"));

static mut IMU_CSV_DATA: Option<Vec<ImuCsvRow>> = None;
static mut VELOCITY_CSV_DATA: Option<Vec<VelocityCsvRow>> = None;
static IMU_CSV_COUNT: Mutex<usize> = Mutex::new(0);
static VELOCITY_CSV_COUNT: Mutex<usize> = Mutex::new(0);

// ---------------------------------------------------------------------------
// Synthetic sensing / feedback state
// ---------------------------------------------------------------------------

/// Source period; the EKF predict step and the synthetic ground truth both advance by this.
const PERIOD_SEC: f64 = 0.05;

/// pose_initializer stand-in: the `/initialpose` NDT starts from before the EKF has a pose.
const INITIAL_POSE: PlanarPose = PlanarPose {
    x: 0.0,
    y: 0.0,
    yaw: 0.0,
};

/// Where the (simulated) vehicle really starts. Offset from `INITIAL_POSE` on purpose, so the
/// first NDT alignment has an initial-pose error to correct.
const GROUND_TRUTH_START: PlanarPose = PlanarPose {
    x: 0.5,
    y: -0.3,
    yaw: 0.03,
};

/// Synthetic LiDAR range and the downsampled cloud size handed to NDT.
const LIDAR_RANGE_M: f64 = 30.0;
const LIDAR_MAX_POINTS: usize = 1_500;

/// MRM trigger stand-in: switch the in-lane stop on after this many source periods (10 s).
const MRM_TRIGGER_ON_TICK: u32 = 200;
const MRM_TARGET_ACCELERATION: f32 = -1.0;
const MRM_TARGET_JERK: f32 = -0.5;

/// Latest vehicle twist (vx, wz) from `vehicle_velocity_converter`, used to dead-reckon the
/// synthetic ground truth that the LiDAR dummy observes.
static LATEST_VEHICLE_TWIST: Mutex<(f64, f64)> = Mutex::new((0.0, 0.0));
static GROUND_TRUTH: Mutex<Option<PlanarPose>> = Mutex::new(None);
static SYNTHETIC_WORLD: Mutex<Option<World>> = Mutex::new(None);

/// `/localization/pose_twist_fusion_filter/biased_pose_with_covariance`. NDT reads it as the
/// initial guess; it is shared state rather than a DAG edge because EKF -> NDT -> EKF is a cycle.
static EKF_BIASED_POSE: Mutex<Option<Pose>> = Mutex::new(None);

static NDT: Mutex<Option<NdtScanMatcher>> = Mutex::new(None);
static NDT_LAST_REPORT: Mutex<Option<AlignReport>> = Mutex::new(None);
static NDT_PUBLISHED: AtomicU32 = AtomicU32::new(0);
static NDT_SKIPPED: AtomicU32 = AtomicU32::new(0);

static SOURCE_TICK: AtomicU32 = AtomicU32::new(0);
/// Last execution times [us] of the synthetic scan and of NDT (`exe_time_ms` debug topic upstream).
static SCAN_GEN_TIME_US: AtomicU32 = AtomicU32::new(0);
static NDT_EXE_TIME_US: AtomicU32 = AtomicU32::new(0);
static SINK_TICK: AtomicU32 = AtomicU32::new(0);

fn configure_nodes() {
    // twist2accel subscribes to the twist estimator output in this pipeline (see the node diagram).
    twist2accel::configure(false);

    // Map loading happens before the pose estimator is activated upstream (pose_initializer
    // triggers NDT only once the map is in); doing it here keeps the first scan off the map
    // build, which would otherwise back up the whole DAG for several periods.
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

pub async fn run() {
    wait_microsec(1000000);

    if let Err(e) = initialize_csv_data() {
        log::warn!("Failed to initialize CSV data: {}", e);
    }

    configure_nodes();

    log::info!("Starting Autoware test application with simplified TCP networking");

    let dag = create_dag();
    let _dag_id = dag.get_id();

    dag.register_periodic_reactor::<_, (i32, i32, i32, i32)>(
        "start_dummy_data".into(),
        move || -> (i32, i32, i32, i32) {
            SOURCE_TICK.fetch_add(1, Ordering::Relaxed);
            (1, 2, 3, 4)
        },
        vec![
            Cow::from("start_imu"),
            Cow::from("start_vel"),
            Cow::from("start_lidar"),
            Cow::from("start_mrm"),
        ],
        SchedulerType::GEDF(5),
        Duration::from_millis(50),
    )
    .await;

    dag.register_reactor::<_, (i32,), (ImuMsg,)>(
        "imu_driver".into(),
        move |(_start_msg,): (i32,)| -> (ImuMsg,) {
            let mut node = MCSNode::new();
            let mut count_guard = IMU_CSV_COUNT.lock(&mut node);
            let count = *count_guard;
            let data = unsafe { IMU_CSV_DATA.as_ref() };
            let awkernel_timestamp = get_awkernel_uptime_timestamp();

            let imu_msg = if let Some(csv_data) = data {
                if csv_data.is_empty() {
                    // Fallback: generate dummy IMU data
                    let mut parser = TamagawaImuParser::new("imu_link");
                    let static_dummy_data = parser.generate_static_dummy_data(awkernel_timestamp);
                    parser
                        .parse_binary_data(&static_dummy_data, awkernel_timestamp)
                        .unwrap_or_default()
                } else {
                    let idx = count % csv_data.len();
                    let row = &csv_data[idx];
                    build_imu_msg_from_csv_row(row, "imu_link", awkernel_timestamp)
                }
            } else {
                // Fallback: generate dummy IMU data if data is not initialized
                let mut parser = TamagawaImuParser::new("imu_link");
                let static_dummy_data = parser.generate_static_dummy_data(awkernel_timestamp);
                parser
                    .parse_binary_data(&static_dummy_data, awkernel_timestamp)
                    .unwrap_or_default()
            };

            *count_guard += 1;
            if *count_guard >= 5700 {
                *count_guard = 0;
                log::info!("rust_e2e_app: finish csv for IMU");
                loop {
                    wait_microsec(1000);
                }
            }

            if LOG_ENABLE {
                log::debug!(
                    "IMU data in imu_driver_node,num={}, timestamp={}",
                    count,
                    imu_msg.header.timestamp
                );
            }

            (imu_msg,)
        },
        vec![Cow::from("start_imu")],
        vec![Cow::from("imu_data")],
        SchedulerType::GEDF(5),
    )
    .await;

    dag.register_reactor::<_, (i32,), (TwistWithCovarianceStamped,)>(
        "vehicle_velocity_converter".into(),
        move |(_start_msg,): (i32,)| -> (TwistWithCovarianceStamped,) {
            let converter = VehicleVelocityConverter::default();

            let mut node = MCSNode::new();
            let mut count_guard = VELOCITY_CSV_COUNT.lock(&mut node);
            let count = *count_guard;
            let data = unsafe { VELOCITY_CSV_DATA.as_ref() };
            let awkernel_timestamp = get_awkernel_uptime_timestamp();

            let twist_msg = if let Some(csv_data) = data {
                if csv_data.is_empty() {
                    // Fallback: generate dummy velocity report
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
                } else {
                    let idx = count % csv_data.len();
                    let row = &csv_data[idx];
                    let velocity_report =
                        build_velocity_report_from_csv_row(row, "base_link", awkernel_timestamp);
                    converter.convert_velocity_report(&velocity_report)
                }
            } else {
                // Fallback: generate dummy velocity report if data is not initialized
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
            };

            {
                let mut node = MCSNode::new();
                *LATEST_VEHICLE_TWIST.lock(&mut node) =
                    (twist_msg.twist.twist.linear.x, twist_msg.twist.twist.angular.z);
            }

            *count_guard += 1;
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

            (twist_msg,)
        },
        vec![Cow::from("start_vel")],
        vec![Cow::from("velocity_twist")],
        SchedulerType::GEDF(5),
    )
    .await;

    dag.register_reactor::<_, (ImuMsg,), (ImuWithCovariance,)>(
        "imu_corrector".into(),
        |(imu_msg,): (ImuMsg,)| -> (ImuWithCovariance,) {
            let corrector = ImuCorrector::new();
            let corrected = corrector.correct_imu_with_covariance(&imu_msg, None);
            (corrected,)
        },
        vec![Cow::from("imu_data")],
        vec![Cow::from("corrected_imu_data")],
        SchedulerType::GEDF(5),
    )
    .await;

    dag.register_reactor::<_, (
        ImuWithCovariance,
        TwistWithCovarianceStamped,
    ), (TwistWithCovarianceStamped,)>(
        "gyro_odometer".into(),
        |(imu_with_cov, vehicle_twist): (
            ImuWithCovariance,
            TwistWithCovarianceStamped,
        )|
         -> (TwistWithCovarianceStamped,) {
            let current_timestamp = imu_with_cov.header.timestamp;
            let current_time = get_awkernel_uptime_timestamp();

            let gyro_odometer = match gyro_odometer::get_or_initialize() {
                Ok(core) => core,
                Err(_) => {
                    return (reactor_helpers::create_empty_twist(current_timestamp),);
                }
            };

            gyro_odometer.add_vehicle_twist(vehicle_twist);
            gyro_odometer.add_imu(imu_with_cov);

            match gyro_odometer.process_and_get_result(current_time) {
                Some(result) => (gyro_odometer.process_result(result),),
                None => (reactor_helpers::create_empty_twist(current_timestamp),),
            }
        },
        vec![Cow::from("corrected_imu_data"), Cow::from("velocity_twist")],
        vec![Cow::from("twist_with_covariance")],
        SchedulerType::GEDF(5),
    )
    .await;

    // -----------------------------------------------------------------------
    // Pose estimator: synthetic LiDAR -> NDT scan matcher
    // -----------------------------------------------------------------------

    dag.register_reactor::<_, (i32,), (PointCloud,)>(
        "pointcloud_dummy_generator".into(),
        move |(_start_msg,): (i32,)| -> (PointCloud,) {
            let (vx, wz) = {
                let mut node = MCSNode::new();
                let twist = *LATEST_VEHICLE_TWIST.lock(&mut node);
        twist
            };
            // Dead-reckon the simulated vehicle with the same wheel-speed / yaw-rate data the
            // twist estimator sees; NDT must recover this pose from the scan alone.
            let truth = {
                let mut node = MCSNode::new();
                let mut gt = GROUND_TRUTH.lock(&mut node);
                let next = match *gt {
                    None => GROUND_TRUTH_START,
                    Some(p) => PlanarPose {
                        x: p.x + vx * libm::cos(p.yaw) * PERIOD_SEC,
                        y: p.y + vx * libm::sin(p.yaw) * PERIOD_SEC,
                        yaw: p.yaw + wz * PERIOD_SEC,
                    },
                };
                *gt = Some(next);
                next
            };
            let started = awkernel_lib::time::Instant::now();
            let cloud = PointCloud {
                header: common_types::Header {
                    frame_id: "base_link",
                    timestamp: get_awkernel_uptime_timestamp(),
                },
                points: {
                    let mut node = MCSNode::new();
                    let mut world = SYNTHETIC_WORLD.lock(&mut node);
                    world
                        .get_or_insert_with(World::new)
                        .scan(truth, LIDAR_RANGE_M, LIDAR_MAX_POINTS)
                },
            };
            SCAN_GEN_TIME_US.store(elapsed_us(started), Ordering::Relaxed);
            (cloud,)
        },
        vec![Cow::from("start_lidar")],
        vec![Cow::from("/localization/util/downsample/pointcloud")],
        SchedulerType::GEDF(5),
    )
    .await;

    dag.register_reactor::<_, (PointCloud,), (Option<PoseWithCovarianceStamped>,)>(
        "ndt_scan_matcher".into(),
        |(cloud,): (PointCloud,)| -> (Option<PoseWithCovarianceStamped>,) {
            let initial_pose = {
                let mut node = MCSNode::new();
                let biased = *EKF_BIASED_POSE.lock(&mut node);
                biased.unwrap_or_else(|| {
                    ndt_scan_matcher::planar_pose(INITIAL_POSE.x, INITIAL_POSE.y, INITIAL_POSE.yaw)
                })
            };

            let started = awkernel_lib::time::Instant::now();
            let mut node = MCSNode::new();
            let mut ndt_guard = NDT.lock(&mut node);
            if ndt_guard.is_none() {
                match NdtScanMatcher::new(NdtParams::default()) {
                    Ok(m) => *ndt_guard = Some(m),
                    Err(e) => {
                        log::error!("ndt_scan_matcher: engine init failed: {:?}", e);
                        return (None,);
                    }
                }
            }
            let Some(ndt) = ndt_guard.as_mut() else {
                return (None,);
            };

            // Dynamic map loading around the current estimate (upstream runs this on a timer;
            // it is a no-op until the vehicle has moved `update_distance`).
            if let Err(e) = ndt.update_map(initial_pose.position.x, initial_pose.position.y) {
                log::warn!("ndt_scan_matcher: map update failed: {:?}", e);
            }

            let result = match ndt.align(&cloud, &initial_pose) {
                Ok(Ok(pose)) => {
                    NDT_PUBLISHED.fetch_add(1, Ordering::Relaxed);
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
            let mut node = MCSNode::new();
            *NDT_LAST_REPORT.lock(&mut node) = report;
            (result,)
        },
        vec![Cow::from("/localization/util/downsample/pointcloud")],
        vec![Cow::from("/localization/pose_estimator/pose_with_covariance")],
        SchedulerType::GEDF(5),
    )
    .await;

    // -----------------------------------------------------------------------
    // Pose / twist fusion: EKF
    // -----------------------------------------------------------------------

    dag.register_reactor::<_, (Option<PoseWithCovarianceStamped>, TwistWithCovarianceStamped), (EKFOdometry,)>(
        "ekf_localizer".into(),
        |(ndt_pose, mut twist): (Option<PoseWithCovarianceStamped>, TwistWithCovarianceStamped)| -> (EKFOdometry,) {
            let ekf = get_or_initialize_default_module();
            let current_time = get_awkernel_uptime_timestamp();

            // Seeded once from the first NDT pose, standing in for upstream's `initialize()` from
            // `/initialpose3d`; every later NDT pose feeds `measurement_update_pose`. No TF
            // listener is wired in yet, so the transform is the identity (see
            // `EKFModule::initialize`'s doc comment). Until then the filter is not run, as
            // upstream's timer returns early while uninitialized.
            static mut INITIALIZED: bool = false;
            let just_initialized = unsafe {
                match (&ndt_pose, INITIALIZED) {
                    (Some(pose), false) => {
                        ekf.initialize(pose, &Transform::identity());
                        INITIALIZED = true;
                        true
                    }
                    _ => false,
                }
            };
            let initialized = unsafe { INITIALIZED };

            if initialized {
                // Fixed timestep matching the periodic "start_dummy_data" trigger cadence.
                // Both calls are required every predict tick -- `predict_with_delay` does not
                // age the delay-time buffer itself (see its doc comment).
                ekf.accumulate_delay_time(PERIOD_SEC);
                ekf.predict_with_delay(PERIOD_SEC);

                // A scan that NDT did not publish (not converged / gated) is simply absent,
                // as on the ROS topic.
                if let (false, Some(pose)) = (just_initialized, &ndt_pose) {
                    ekf.measurement_update_pose(pose, current_time);
                }

                // Below `threshold_observable_velocity_mps` (0.0 by default, matching upstream's
                // shipped config), vx is inflated as unreliable before fusing; wiring the gate
                // unconditionally here matches upstream's `callback_twist_with_covariance`, even
                // though the default threshold makes it a no-op today.
                apply_twist_observability_gate(
                    &mut twist,
                    EKFParameters::default().threshold_observable_velocity_mps,
                );
                ekf.measurement_update_twist(&twist, current_time);

                let biased = ekf.get_current_pose(true, current_time).pose;
                let mut node = MCSNode::new();
                *EKF_BIASED_POSE.lock(&mut node) = Some(biased);
            }

            let ekf_pose = ekf.get_current_pose(false, current_time);
            let pose_covariance = ekf.get_current_pose_covariance();
            let twist_covariance = ekf.get_current_twist_covariance();
            let ekf_twist = ekf.get_current_twist(current_time);

            let odometry = EKFOdometry {
                header: common_types::Header {
                    frame_id: "map",
                    timestamp: twist.header.timestamp,
                },
                child_frame_id: "base_link",
                pose: PoseWithCovariance {
                    pose: ekf_pose.pose,
                    covariance: pose_covariance,
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
                    covariance: twist_covariance,
                },
            };

            (odometry,)
        },
        vec![
            Cow::from("/localization/pose_estimator/pose_with_covariance"),
            Cow::from("twist_with_covariance"),
        ],
        vec![Cow::from("/localization/pose_twist_fusion_filter/kinematic_state")],
        SchedulerType::GEDF(5),
    )
    .await;

    dag.register_reactor::<_, (EKFOdometry,), (KinematicState,)>(
        "stop_filter".into(),
        |(odom,): (EKFOdometry,)| -> (KinematicState,) {
            let ks = bridge::odometry_to_kinematic_state(&odom);
            (stop_filter::apply(
                ks,
                stop_filter::DEFAULT_VX_THRESHOLD,
                stop_filter::DEFAULT_WZ_THRESHOLD,
            ),)
        },
        vec![Cow::from("/localization/pose_twist_fusion_filter/kinematic_state")],
        vec![Cow::from("/localization/kinematic_state")],
        SchedulerType::GEDF(5),
    )
    .await;

    dag.register_reactor::<_, (TwistWithCovarianceStamped,), (AccelWithCovarianceStamped,)>(
        "twist2accel".into(),
        |(twist,): (TwistWithCovarianceStamped,)| -> (AccelWithCovarianceStamped,) {
            let (header, twist) = bridge::twist_to_localization(&twist);
            // `use_odom = false` is set in `configure_nodes`, so `on_twist` always yields.
            let accel = twist2accel::on_twist(&header, &twist).unwrap_or_else(|| {
                log::warn!("twist2accel: configured for odometry input; publishing zero accel");
                AccelWithCovarianceStamped {
                    header,
                    ..Default::default()
                }
            });
            (accel,)
        },
        vec![Cow::from("twist_with_covariance")],
        vec![Cow::from("/localization/acceleration")],
        SchedulerType::GEDF(5),
    )
    .await;

    // -----------------------------------------------------------------------
    // Control: trajectory follower -> constant-jerk deceleration -> command gate
    // -----------------------------------------------------------------------

    dag.register_reactor::<_, (KinematicState, AccelWithCovarianceStamped), (Control,)>(
        "trajectory_follower".into(),
        |(ks, accel): (KinematicState, AccelWithCovarianceStamped)| -> (Control,) {
            (trajectory_follower::follow(&ks, &accel),)
        },
        vec![
            Cow::from("/localization/kinematic_state"),
            Cow::from("/localization/acceleration"),
        ],
        vec![Cow::from("/control/trajectory_follower/control_cmd")],
        SchedulerType::GEDF(5),
    )
    .await;

    dag.register_reactor::<_, (i32,), (ConstantJerkDecelerationTrigger,)>(
        "mrm_trigger_dummy".into(),
        |(_start_msg,): (i32,)| -> (ConstantJerkDecelerationTrigger,) {
            let tick = SOURCE_TICK.load(Ordering::Relaxed);
            (ConstantJerkDecelerationTrigger {
                stamp: localization_types::Instant::now(),
                trigger: tick >= MRM_TRIGGER_ON_TICK,
                target_acceleration: MRM_TARGET_ACCELERATION,
                target_jerk: MRM_TARGET_JERK,
            },)
        },
        vec![Cow::from("start_mrm")],
        vec![Cow::from("/control/constant_jerk_deceleration_trigger")],
        SchedulerType::GEDF(5),
    )
    .await;

    dag.register_reactor::<_, (Control, ConstantJerkDecelerationTrigger), (
        Control,
        GearCommand,
        HazardLightsCommand,
        TurnIndicatorsCommand,
    )>(
        "constant_jerk_deceleration_controller".into(),
        |(control, trigger): (Control, ConstantJerkDecelerationTrigger)| -> (
            Control,
            GearCommand,
            HazardLightsCommand,
            TurnIndicatorsCommand,
        ) {
            // Upstream has two independent callbacks; in the DAG both inputs arrive per period,
            // and the trigger is applied first so it acts on this period's control.
            constant_jerk_deceleration_controller::on_trigger(&trigger);
            let ControllerOutput {
                control,
                gear,
                hazard_lights,
                turn_indicators,
            } = constant_jerk_deceleration_controller::on_control(&control);
            (control, gear, hazard_lights, turn_indicators)
        },
        vec![
            Cow::from("/control/trajectory_follower/control_cmd"),
            Cow::from("/control/constant_jerk_deceleration_trigger"),
        ],
        vec![
            Cow::from("/control/control_command_gate/inputs/in_lane_stop/control"),
            Cow::from("/control/control_command_gate/inputs/in_lane_stop/gear"),
            Cow::from("/control/control_command_gate/inputs/in_lane_stop/hazard_lights"),
            Cow::from("/control/control_command_gate/inputs/in_lane_stop/turn_indicators"),
        ],
        SchedulerType::GEDF(5),
    )
    .await;

    dag.register_reactor::<_, (
        Control,
        GearCommand,
        HazardLightsCommand,
        TurnIndicatorsCommand,
        KinematicState,
    ), (Control, GearCommand, HazardLightsCommand, TurnIndicatorsCommand)>(
        "control_command_gate".into(),
        |(control, gear, hazard_lights, turn_indicators, ks): (
            Control,
            GearCommand,
            HazardLightsCommand,
            TurnIndicatorsCommand,
            KinematicState,
        )| -> (Control, GearCommand, HazardLightsCommand, TurnIndicatorsCommand) {
            // With a single command source the CommandSelector is a pass-through, so only the
            // Control goes through the limit filter; gear / lights are forwarded as selected.
            (
                command_gate::filter(&control, &ks),
                gear,
                hazard_lights,
                turn_indicators,
            )
        },
        vec![
            Cow::from("/control/control_command_gate/inputs/in_lane_stop/control"),
            Cow::from("/control/control_command_gate/inputs/in_lane_stop/gear"),
            Cow::from("/control/control_command_gate/inputs/in_lane_stop/hazard_lights"),
            Cow::from("/control/control_command_gate/inputs/in_lane_stop/turn_indicators"),
            Cow::from("/localization/kinematic_state"),
        ],
        vec![
            Cow::from("/control/command/control_cmd"),
            Cow::from("/control/command/gear_cmd"),
            Cow::from("/control/command/hazard_lights_cmd"),
            Cow::from("/control/command/turn_indicators_cmd"),
        ],
        SchedulerType::GEDF(5),
    )
    .await;

    // -----------------------------------------------------------------------
    // Sink: vehicle interface stand-in (+ EKF odometry export over UDP)
    // -----------------------------------------------------------------------

    dag.register_sink_reactor::<_, (
        Control,
        GearCommand,
        HazardLightsCommand,
        TurnIndicatorsCommand,
        EKFOdometry,
    )>(
        "vehicle_cmd_sink".into(),
        move |(control, gear, hazard_lights, turn_indicators, ekf_odom): (
            Control,
            GearCommand,
            HazardLightsCommand,
            TurnIndicatorsCommand,
            EKFOdometry,
        )| {
            let tick = SINK_TICK.fetch_add(1, Ordering::Relaxed);
            if tick.is_multiple_of(PIPELINE_LOG_EVERY) {
                log_pipeline(&control, &gear, &hazard_lights, &turn_indicators, &ekf_odom);
            }

            let json_data = format!(
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
            );

            save_json_data_to_global(json_data);
        },
        vec![
            Cow::from("/control/command/control_cmd"),
            Cow::from("/control/command/gear_cmd"),
            Cow::from("/control/command/hazard_lights_cmd"),
            Cow::from("/control/command/turn_indicators_cmd"),
            Cow::from("/localization/pose_twist_fusion_filter/kinematic_state"),
        ],
        SchedulerType::GEDF(5),
        Duration::from_millis(50),
    )
    .await;

    let result = finish_create_dags(slice::from_ref(&dag)).await;

    match result {
        Ok(_) => {
            log::info!("Autoware test application DAGs created successfully");
        }
        Err(errors) => {
            log::error!("Failed to create Autoware test application DAGs");
            for error in errors {
                log::error!("- {error}");
            }
        }
    }

    log::info!("Autoware test application DAG completed");

    log::info!("=== Network test start ===");
    log::info!("Interface ID: {}", INTERFACE_ID);
    log::info!("Interface IP: {}", INTERFACE_ADDR);
    log::info!("Destination IP: {}", UDP_TCP_DST_ADDR);
    awkernel_lib::net::add_ipv4_addr(INTERFACE_ID, INTERFACE_ADDR, 24);
    log::info!(
        "Configured IPv4 address {} on interface {}",
        INTERFACE_ADDR,
        INTERFACE_ID
    );

    log::info!("Waiting for network stack initialization...");
    awkernel_async_lib::sleep(Duration::from_secs(2)).await;

    log::info!("Starting periodic UDP sender task");
    start_periodic_udp_sender().await;

    log::info!("Waiting for JSON data to become ready...");
    let mut wait_count = 0;
    const MAX_WAIT_COUNT: u32 = 3;

    while !JSON_DATA_READY.load(Ordering::Relaxed) && wait_count < MAX_WAIT_COUNT {
        awkernel_async_lib::sleep(Duration::from_secs(1)).await;
        wait_count += 1;
    }

    if JSON_DATA_READY.load(Ordering::Relaxed) {
    } else {
        log::warn!("JSON data was not ready. Periodic UDP sender task remains waiting");
    }

    log::info!("Autoware test application completed");
}

fn initialize_csv_data() -> Result<(), &'static str> {
    unsafe {
        if IMU_CSV_DATA.is_none() {
            let imu_data = parse_imu_csv(IMU_CSV_DATA_STR)?;
            log::info!("Loaded IMU CSV data: {} rows", imu_data.len());
            IMU_CSV_DATA = Some(imu_data);
        }

        if VELOCITY_CSV_DATA.is_none() {
            let velocity_data = parse_velocity_csv(VELOCITY_CSV_DATA_STR)?;
            log::info!("Loaded velocity CSV data: {} rows", velocity_data.len());
            VELOCITY_CSV_DATA = Some(velocity_data);
        }
    }

    Ok(())
}

fn parse_imu_csv(csv: &str) -> Result<Vec<ImuCsvRow>, &'static str> {
    let mut rows = Vec::new();

    parse_csv_records(csv, |fields| {
        if fields.len() < 12 {
            return Err("IMU CSV has insufficient columns");
        }

        let timestamp = parse_timestamp(fields[0], fields[1])?;
        let angular_velocity = common_types::Vector3::new(
            parse_f64(fields[6])?,
            parse_f64(fields[7])?,
            parse_f64(fields[8])?,
        );
        let linear_acceleration = common_types::Vector3::new(
            parse_f64(fields[9])?,
            parse_f64(fields[10])?,
            parse_f64(fields[11])?,
        );

        rows.push(ImuCsvRow {
            timestamp,
            angular_velocity,
            linear_acceleration,
        });
        Ok(())
    })?;

    Ok(rows)
}

fn parse_velocity_csv(csv: &str) -> Result<Vec<VelocityCsvRow>, &'static str> {
    let mut rows = Vec::new();

    parse_csv_records(csv, |fields| {
        if fields.len() < 5 {
            return Err("Velocity CSV has insufficient columns");
        }

        let timestamp = parse_timestamp(fields[0], fields[1])?;
        let longitudinal_velocity = parse_f64(fields[2])?;
        let lateral_velocity = parse_f64(fields[3])?;
        let heading_rate = parse_f64(fields[4])?;

        rows.push(VelocityCsvRow {
            timestamp,
            longitudinal_velocity,
            lateral_velocity,
            heading_rate,
        });
        Ok(())
    })?;

    Ok(rows)
}

fn parse_csv_records<F>(csv: &str, mut on_record: F) -> Result<(), &'static str>
where
    F: FnMut(&[&str]) -> Result<(), &'static str>,
{
    let mut reader = Reader::new();
    let mut input = csv.as_bytes();
    let mut output = vec![0u8; 4096];
    let mut ends = vec![0usize; 32];
    let mut header_skipped = false;

    loop {
        let (result, in_read, _out_written, num_fields) =
            reader.read_record(input, &mut output, &mut ends);
        input = &input[in_read..];

        if matches!(result, ReadRecordResult::OutputFull) {
            return Err("CSV output buffer is too small");
        }

        if num_fields == 0 {
            if matches!(result, ReadRecordResult::InputEmpty | ReadRecordResult::End) {
                break;
            }
            continue;
        }

        let mut fields: Vec<&str> = Vec::with_capacity(num_fields);
        let mut start = 0usize;
        for &end in ends.iter().take(num_fields) {
            let slice = &output[start..end];
            let field = core::str::from_utf8(slice).map_err(|_| "Failed to decode CSV UTF-8")?;
            fields.push(field);
            start = end;
        }

        if !header_skipped {
            header_skipped = true;
        } else {
            on_record(&fields)?;
        }

        if matches!(result, ReadRecordResult::End) {
            break;
        }
    }

    Ok(())
}

fn parse_timestamp(sec: &str, nsec: &str) -> Result<u64, &'static str> {
    let sec_val = parse_u64(sec)?;
    let nsec_val = parse_u64(nsec)?;
    let ts = sec_val
        .checked_mul(1_000_000_000)
        .and_then(|v| v.checked_add(nsec_val))
        .ok_or("Timestamp calculation overflowed")?;
    Ok(ts)
}

fn parse_u64(field: &str) -> Result<u64, &'static str> {
    let trimmed = field.trim();
    if trimmed.is_empty() {
        return Ok(0);
    }
    trimmed.parse::<u64>().map_err(|_| "Failed to parse u64")
}

fn parse_f64(field: &str) -> Result<f64, &'static str> {
    let trimmed = field.trim();
    if trimmed.is_empty() {
        return Ok(0.0);
    }
    trimmed.parse::<f64>().map_err(|_| "Failed to parse f64")
}

fn get_awkernel_uptime_timestamp() -> u64 {
    let uptime_nanos = awkernel_lib::delay::uptime_nano();
    if uptime_nanos > u64::MAX as u128 {
        u64::MAX
    } else {
        uptime_nanos as u64
    }
}

pub async fn start_periodic_udp_sender() {
    awkernel_async_lib::spawn(
        "periodic_udp_sender".into(),
        periodic_udp_sender_task(),
        awkernel_async_lib::scheduler::SchedulerType::GEDF(5),
    )
    .await;
}

async fn periodic_udp_sender_task() {
    let socket_result = awkernel_async_lib::net::udp::UdpSocket::bind_on_interface(
        INTERFACE_ID,
        &Default::default(),
    );

    let mut socket = match socket_result {
        Ok(socket) => socket,
        Err(e) => {
            log::error!(
                "Periodic UDP sender task: failed to create UDP socket: {:?}",
                e
            );
            return;
        }
    };

    let dst_addr = IpAddr::new_v4(UDP_TCP_DST_ADDR);
    let mut counter = 0;

    loop {
        if !JSON_DATA_READY.load(Ordering::Relaxed) {
            awkernel_async_lib::sleep(Duration::from_secs(1)).await;
            continue;
        }

        let json_data = unsafe { LATEST_JSON_DATA.clone() };

        if let Some(data) = json_data {
            match socket.send(data.as_bytes(), &dst_addr, UDP_DST_PORT).await {
                Ok(_) => {
                    counter += 1;
                    log::info!(
                        "Periodic UDP sender task: send success #{} ({} bytes)",
                        counter,
                        data.len()
                    );

                    let mut buf = [0u8; 1024];
                    if let Some(Ok((n, src_addr, src_port))) = awkernel_async_lib::timeout(
                        Duration::from_millis(100),
                        socket.recv(&mut buf),
                    )
                    .await
                    {
                        if let Ok(response) = core::str::from_utf8(&buf[..n]) {
                            log::debug!(
                                "Periodic UDP sender task: response received: {}:{} - {}",
                                src_addr.get_addr(),
                                src_port,
                                response
                            );
                        }
                    }
                }
                Err(e) => {
                    log::warn!("Periodic UDP sender task: send failed: {:?}", e);
                }
            }
        } else {
            log::warn!("Periodic UDP sender task: failed to get JSON data");
        }

        awkernel_async_lib::sleep(Duration::from_millis(5)).await;
    }
}

fn save_json_data_to_global(json_data: String) {
    unsafe {
        LATEST_JSON_DATA = Some(json_data.clone());
    }
    JSON_DATA_READY.store(true, Ordering::Relaxed);
    JSON_DATA_LENGTH.store(json_data.len(), Ordering::Relaxed);
}

fn elapsed_us(started: awkernel_lib::time::Instant) -> u32 {
    u32::try_from(started.elapsed().as_micros()).unwrap_or(u32::MAX)
}

fn yaw_from_quaternion(x: f64, y: f64, z: f64, w: f64) -> f64 {
    libm::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
}

/// One summary line of the whole pipeline (truth / NDT / EKF / commands) for serial-log checks.
fn log_pipeline(
    control: &Control,
    gear: &GearCommand,
    hazard_lights: &HazardLightsCommand,
    turn_indicators: &TurnIndicatorsCommand,
    ekf_odom: &EKFOdometry,
) {
    let truth = {
        let mut node = MCSNode::new();
        let truth = *GROUND_TRUTH.lock(&mut node);
        truth
    };
    let report = {
        let mut node = MCSNode::new();
        let report = *NDT_LAST_REPORT.lock(&mut node);
        report
    };
    let p = &ekf_odom.pose.pose;
    let ekf_yaw = yaw_from_quaternion(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w);
    let (tx, ty, tyaw) = truth.map_or((f64::NAN, f64::NAN, f64::NAN), |t| (t.x, t.y, t.yaw));
    let (iters, nvtl, converged) =
        report.map_or((-1, f32::NAN, false), |r| (r.iteration_num, r.nearest_voxel_likelihood, r.converged));
    log::info!(
        "[pipeline] tick={} truth=({:.3},{:.3},{:.4}) scan_us={} ndt(iter={},nvtl={:.3},conv={},pub={},skip={},us={}) ekf=({:.3},{:.3},{:.4}) err=({:.3},{:.3}) cmd(v={:.3},a={:.3},j={:.3},steer={:.4}) gear={} hazard={} turn={} mrm={}",
        SOURCE_TICK.load(Ordering::Relaxed),
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
        p.position.x,
        p.position.y,
        ekf_yaw,
        p.position.x - tx,
        p.position.y - ty,
        control.longitudinal.velocity,
        control.longitudinal.acceleration,
        control.longitudinal.jerk,
        control.lateral.steering_tire_angle,
        gear.command,
        hazard_lights.command,
        turn_indicators.command,
        SOURCE_TICK.load(Ordering::Relaxed) >= MRM_TRIGGER_ON_TICK,
    );
}
