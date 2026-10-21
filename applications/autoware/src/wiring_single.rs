//! Single-DAG wiring: one periodic source fans out to every sensor branch (the RTNS MRM
//! approach of merging several sources into one). Everything runs at the source period;
//! the LiDAR branch only does work every `LIDAR_DIVIDER`-th tick, and the EKF predicts by
//! the source period instead of its own 20 ms timer.
//!
//! A node has at most five input topics, so the EKF odometry reaches the sink (for the UDP
//! odometry export) through `LATEST_ODOMETRY` instead of a sixth sink input.

use alloc::{borrow::Cow, sync::Arc, vec, vec::Vec};
use awkernel_async_lib::dag::{create_dag, Dag};
use awkernel_async_lib::scheduler::SchedulerType;
use core::time::Duration;

use ekf_localizer::{EKFOdometry, PoseWithCovarianceStamped};
use imu_corrector::ImuWithCovariance;
use imu_driver::ImuMsg;
use localization_types::{
    AccelWithCovarianceStamped, ConstantJerkDecelerationTrigger, Control, GearCommand,
    HazardLightsCommand, KinematicState, TurnIndicatorsCommand,
};
#[cfg(not(feature = "replay"))]
use ndt_scan_matcher::PointCloud;
use vehicle_velocity_converter::TwistWithCovarianceStamped;

use crate::nodes;
use crate::shared::Latest;

/// `/localization/pose_twist_fusion_filter/kinematic_state` for the sink's odometry export.
static LATEST_ODOMETRY: Latest<EKFOdometry> = Latest::new();

/// Source period: the IMU / vehicle status rate (33 ms in the RTNS node diagram).
const PERIOD: Duration = Duration::from_millis(33);
/// LiDAR runs at 100 ms, i.e. every third 33 ms tick.
#[cfg(not(feature = "replay"))]
const LIDAR_DIVIDER: u32 = 3;

pub async fn build() -> Vec<Arc<Dag>> {
    let period_sec = PERIOD.as_secs_f64();
    let dag = create_dag();

    dag.register_periodic_reactor::<_, (u32, u32, u32, u32)>(
        "start_dummy_data".into(),
        {
            let tick = core::sync::atomic::AtomicU32::new(0);
            move || -> (u32, u32, u32, u32) {
                let t = tick.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                (t, t, t, t)
            }
        },
        vec![
            Cow::from("start_imu"),
            Cow::from("start_vel"),
            Cow::from("start_lidar"),
            Cow::from("start_mrm"),
        ],
        SchedulerType::GEDF(5),
        PERIOD,
    )
    .await;

    dag.register_reactor::<_, (u32,), (ImuMsg,)>(
        "imu_driver".into(),
        |(_tick,): (u32,)| -> (ImuMsg,) { (nodes::imu_driver(),) },
        vec![Cow::from("start_imu")],
        vec![Cow::from("imu_data")],
        SchedulerType::GEDF(5),
    )
    .await;

    dag.register_reactor::<_, (u32,), (TwistWithCovarianceStamped,)>(
        "vehicle_velocity_converter".into(),
        move |(_tick,): (u32,)| -> (TwistWithCovarianceStamped,) {
            (nodes::vehicle_velocity_converter(period_sec),)
        },
        vec![Cow::from("start_vel")],
        vec![Cow::from("velocity_twist")],
        SchedulerType::GEDF(5),
    )
    .await;

    dag.register_reactor::<_, (ImuMsg,), (ImuWithCovariance,)>(
        "imu_corrector".into(),
        |(imu,): (ImuMsg,)| -> (ImuWithCovariance,) { (nodes::imu_corrector(&imu),) },
        vec![Cow::from("imu_data")],
        vec![Cow::from("corrected_imu_data")],
        SchedulerType::GEDF(5),
    )
    .await;

    dag.register_reactor::<_, (ImuWithCovariance, TwistWithCovarianceStamped), (TwistWithCovarianceStamped,)>(
        "gyro_odometer".into(),
        |(imu, twist): (ImuWithCovariance, TwistWithCovarianceStamped)| -> (TwistWithCovarianceStamped,) {
            (nodes::gyro_odometer(imu, twist),)
        },
        vec![Cow::from("corrected_imu_data"), Cow::from("velocity_twist")],
        vec![Cow::from("/localization/twist_estimator/twist_with_covariance")],
        SchedulerType::GEDF(5),
    )
    .await;

    // Pose estimator: synthetic LiDAR + NDT, or the recorded Main ECU pose with `replay`.
    #[cfg(not(feature = "replay"))]
    {
        dag.register_reactor::<_, (u32,), (Option<PointCloud>,)>(
            "pointcloud_dummy_generator".into(),
            |(tick,): (u32,)| -> (Option<PointCloud>,) {
                (tick
                    .is_multiple_of(LIDAR_DIVIDER)
                    .then(nodes::pointcloud_dummy_generator),)
            },
            vec![Cow::from("start_lidar")],
            vec![Cow::from("/localization/util/downsample/pointcloud")],
            SchedulerType::GEDF(5),
        )
        .await;

        dag.register_reactor::<_, (Option<PointCloud>,), (Option<PoseWithCovarianceStamped>,)>(
            "ndt_scan_matcher".into(),
            |(cloud,): (Option<PointCloud>,)| -> (Option<PoseWithCovarianceStamped>,) {
                (cloud.as_ref().and_then(nodes::ndt_scan_matcher),)
            },
            vec![Cow::from("/localization/util/downsample/pointcloud")],
            vec![Cow::from("/localization/pose_estimator/pose_with_covariance")],
            SchedulerType::GEDF(5),
        )
        .await;
    }
    #[cfg(feature = "replay")]
    dag.register_reactor::<_, (u32,), (Option<PoseWithCovarianceStamped>,)>(
        "pose_replay".into(),
        |(_tick,): (u32,)| -> (Option<PoseWithCovarianceStamped>,) { (nodes::pose_replay(),) },
        vec![Cow::from("start_lidar")],
        vec![Cow::from("/localization/pose_estimator/pose_with_covariance")],
        SchedulerType::GEDF(5),
    )
    .await;

    dag.register_reactor::<_, (Option<PoseWithCovarianceStamped>,), (Option<PoseWithCovarianceStamped>,)>(
        "topic_relay_controller_pose".into(),
        |(pose,): (Option<PoseWithCovarianceStamped>,)| -> (Option<PoseWithCovarianceStamped>,) {
            (nodes::pose_relay(pose),)
        },
        vec![Cow::from("/localization/pose_estimator/pose_with_covariance")],
        vec![Cow::from("/main/localization/pose_with_covariance")],
        SchedulerType::GEDF(5),
    )
    .await;

    dag.register_reactor::<_, (Option<PoseWithCovarianceStamped>, TwistWithCovarianceStamped), (EKFOdometry,)>(
        "ekf_localizer".into(),
        move |(pose, twist): (Option<PoseWithCovarianceStamped>, TwistWithCovarianceStamped)| -> (EKFOdometry,) {
            let poses: Vec<_> = pose.into_iter().collect();
            let mut twists = [twist];
            let odom = nodes::ekf_localizer(nodes::EkfDt::Fixed(period_sec), &poses, &mut twists);
            LATEST_ODOMETRY.set(odom.clone());
            (odom,)
        },
        vec![
            Cow::from("/main/localization/pose_with_covariance"),
            Cow::from("/localization/twist_estimator/twist_with_covariance"),
        ],
        vec![Cow::from("/localization/pose_twist_fusion_filter/kinematic_state")],
        SchedulerType::GEDF(5),
    )
    .await;

    dag.register_reactor::<_, (EKFOdometry,), (KinematicState,)>(
        "stop_filter".into(),
        |(odom,): (EKFOdometry,)| -> (KinematicState,) { (nodes::stop_filter(&odom),) },
        vec![Cow::from("/localization/pose_twist_fusion_filter/kinematic_state")],
        vec![Cow::from("/localization/kinematic_state")],
        SchedulerType::GEDF(5),
    )
    .await;

    dag.register_reactor::<_, (KinematicState,), (AccelWithCovarianceStamped,)>(
        "twist2accel".into(),
        |(ks,): (KinematicState,)| -> (AccelWithCovarianceStamped,) {
            (nodes::twist2accel(&ks),)
        },
        vec![Cow::from("/localization/kinematic_state")],
        vec![Cow::from("/localization/acceleration")],
        SchedulerType::GEDF(5),
    )
    .await;

    dag.register_reactor::<_, (KinematicState,), (Control,)>(
        "trajectory_follower".into(),
        |(ks,): (KinematicState,)| -> (Control,) { (nodes::trajectory_follower(&ks),) },
        vec![Cow::from("/localization/kinematic_state")],
        vec![Cow::from("/control/trajectory_follower/control_cmd")],
        SchedulerType::GEDF(5),
    )
    .await;

    dag.register_reactor::<_, (u32,), (ConstantJerkDecelerationTrigger,)>(
        "mrm_trigger_dummy".into(),
        |(_tick,): (u32,)| -> (ConstantJerkDecelerationTrigger,) { (nodes::mrm_trigger(),) },
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
            let out = nodes::constant_jerk_deceleration_controller(&control, &trigger);
            (out.control, out.gear, out.hazard_lights, out.turn_indicators)
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
            (
                nodes::control_command_gate(&control, &ks),
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

    dag.register_sink_reactor::<_, (
        Control,
        GearCommand,
        HazardLightsCommand,
        TurnIndicatorsCommand,
        AccelWithCovarianceStamped,
    )>(
        "vehicle_cmd_sink".into(),
        // `/localization/acceleration` has no consumer in this pipeline (the follower does not
        // use it); it ends here, as an unsubscribed topic does upstream.
        |(control, gear, hazard_lights, turn_indicators, _accel): (
            Control,
            GearCommand,
            HazardLightsCommand,
            TurnIndicatorsCommand,
            AccelWithCovarianceStamped,
        )| {
            let odom = LATEST_ODOMETRY.get();
            nodes::vehicle_cmd_sink(
                "single",
                &control,
                &gear,
                &hazard_lights,
                &turn_indicators,
                odom.as_ref(),
                None,
            );
        },
        vec![
            Cow::from("/control/command/control_cmd"),
            Cow::from("/control/command/gear_cmd"),
            Cow::from("/control/command/hazard_lights_cmd"),
            Cow::from("/control/command/turn_indicators_cmd"),
            Cow::from("/localization/acceleration"),
        ],
        SchedulerType::GEDF(5),
        PERIOD,
    )
    .await;

    vec![dag]
}
