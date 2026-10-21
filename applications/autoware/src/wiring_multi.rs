//! Per-period wiring: one DAG per rate in the RTNS node diagram, each with a single periodic
//! source and a sink whose relative deadline is its period. Data crosses DAG boundaries through
//! `shared::{Latest, Queue}` (topics cannot cross DAGs).
//!
//! | DAG | period | source | nodes | sink -> shared |
//! |---|---|---|---|---|
//! | twist estimator | 33 ms | sensor start | imu_driver, vehicle_velocity_converter, imu_corrector, gyro_odometer | twist queue |
//! | pose estimator | 100 ms | pointcloud_dummy_generator (or pose_input) | ndt_scan_matcher | pose queue |
//! | localization | 20 ms | ekf_localizer (its own timer) | stop_filter, twist2accel | latest kinematic state / odometry / accel |
//! | control | 30 ms | trajectory_follower (its own timer) | constant_jerk_deceleration_controller, control_command_gate | UDP export |
//!
//! twist2accel follows stop_filter because it differentiates `/localization/kinematic_state`
//! (`use_odom: true`); upstream it is a callback on that topic, i.e. at the EKF rate.

use alloc::{borrow::Cow, sync::Arc, vec, vec::Vec};
use awkernel_async_lib::dag::{create_dag, Dag};
use awkernel_async_lib::scheduler::SchedulerType;
use core::time::Duration;

use ekf_localizer::{EKFOdometry, PoseWithCovarianceStamped};
use imu_corrector::ImuWithCovariance;
use imu_driver::ImuMsg;
use localization_types::{
    AccelWithCovarianceStamped, Control, GearCommand, HazardLightsCommand, KinematicState,
    TurnIndicatorsCommand,
};
#[cfg(not(awsim_input))]
use ndt_scan_matcher::PointCloud;
use vehicle_velocity_converter::TwistWithCovarianceStamped;

use crate::nodes::{self, Commands};
use crate::shared::{Latest, Queue};

const TWIST_PERIOD: Duration = Duration::from_millis(33);
const POSE_PERIOD: Duration = Duration::from_millis(100);
/// ekf_localizer `predict_frequency: 50.0`.
const EKF_PERIOD: Duration = Duration::from_millis(20);
/// simple_trajectory_follower `create_timer(30ms)`.
const CONTROL_PERIOD: Duration = Duration::from_millis(30);

/// Enough for a few EKF periods of backlog; older measurements are dropped first.
const MEASUREMENT_QUEUE: usize = 8;

static TWIST_QUEUE: Queue<TwistWithCovarianceStamped, MEASUREMENT_QUEUE> = Queue::new();
static POSE_QUEUE: Queue<PoseWithCovarianceStamped, MEASUREMENT_QUEUE> = Queue::new();
/// `/localization/acceleration`: no consumer in this pipeline (the follower does not use it).
static LATEST_ACCEL: Latest<AccelWithCovarianceStamped> = Latest::new();
static LATEST_KINEMATIC_STATE: Latest<KinematicState> = Latest::new();
static LATEST_ODOMETRY: Latest<EKFOdometry> = Latest::new();

/// Commands of one control period; `None` when the topic was not published this period.
type MaybeCommands = (
    Option<Control>,
    Option<GearCommand>,
    Option<HazardLightsCommand>,
    Option<TurnIndicatorsCommand>,
);

fn some_commands((control, gear, hazard_lights, turn_indicators): Commands) -> MaybeCommands {
    (
        Some(control),
        Some(gear),
        Some(hazard_lights),
        Some(turn_indicators),
    )
}

pub async fn build() -> Vec<Arc<Dag>> {
    vec![
        twist_estimator_dag().await,
        pose_estimator_dag().await,
        localization_dag().await,
        control_dag().await,
    ]
}

async fn twist_estimator_dag() -> Arc<Dag> {
    let period_sec = TWIST_PERIOD.as_secs_f64();
    let dag = create_dag();

    dag.register_periodic_reactor::<_, (u32, u32)>(
        "sensor_start".into(),
        || -> (u32, u32) { (0, 0) },
        vec![Cow::from("start_imu"), Cow::from("start_vel")],
        SchedulerType::GEDF(5),
        TWIST_PERIOD,
    )
    .await;

    dag.register_reactor::<_, (u32,), (ImuMsg,)>(
        "imu_driver".into(),
        |(_,): (u32,)| -> (ImuMsg,) { (nodes::imu_driver(),) },
        vec![Cow::from("start_imu")],
        vec![Cow::from("imu_data")],
        SchedulerType::GEDF(5),
    )
    .await;

    dag.register_reactor::<_, (u32,), (TwistWithCovarianceStamped,)>(
        "vehicle_velocity_converter".into(),
        move |(_,): (u32,)| -> (TwistWithCovarianceStamped,) {
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

    dag.register_sink_reactor::<_, (TwistWithCovarianceStamped,)>(
        "twist_sink".into(),
        |(twist,): (TwistWithCovarianceStamped,)| {
            TWIST_QUEUE.push(twist);
        },
        vec![Cow::from("/localization/twist_estimator/twist_with_covariance")],
        SchedulerType::GEDF(5),
        TWIST_PERIOD,
    )
    .await;

    dag
}

async fn pose_estimator_dag() -> Arc<Dag> {
    let dag = create_dag();

    // Source: synthetic LiDAR (then NDT), or the Main ECU pose from AWSIM (recorded / live).
    #[cfg(not(awsim_input))]
    {
        dag.register_periodic_reactor::<_, (PointCloud,)>(
            "pointcloud_dummy_generator".into(),
            || -> (PointCloud,) { (nodes::pointcloud_dummy_generator(),) },
            vec![Cow::from("/localization/util/downsample/pointcloud")],
            SchedulerType::GEDF(5),
            POSE_PERIOD,
        )
        .await;

        dag.register_reactor::<_, (PointCloud,), (Option<PoseWithCovarianceStamped>,)>(
            "ndt_scan_matcher".into(),
            |(cloud,): (PointCloud,)| -> (Option<PoseWithCovarianceStamped>,) {
                (nodes::ndt_scan_matcher(&cloud),)
            },
            vec![Cow::from("/localization/util/downsample/pointcloud")],
            vec![Cow::from("/localization/pose_estimator/pose_with_covariance")],
            SchedulerType::GEDF(5),
        )
        .await;
    }
    #[cfg(awsim_input)]
    dag.register_periodic_reactor::<_, (Option<PoseWithCovarianceStamped>,)>(
        "pose_input".into(),
        || -> (Option<PoseWithCovarianceStamped>,) { (nodes::pose_input(),) },
        vec![Cow::from("/localization/pose_estimator/pose_with_covariance")],
        SchedulerType::GEDF(5),
        POSE_PERIOD,
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

    dag.register_sink_reactor::<_, (Option<PoseWithCovarianceStamped>,)>(
        "pose_sink".into(),
        |(pose,): (Option<PoseWithCovarianceStamped>,)| {
            if let Some(pose) = pose {
                POSE_QUEUE.push(pose);
            }
        },
        vec![Cow::from("/main/localization/pose_with_covariance")],
        SchedulerType::GEDF(5),
        POSE_PERIOD,
    )
    .await;

    dag
}

async fn localization_dag() -> Arc<Dag> {
    let period_sec = EKF_PERIOD.as_secs_f64();
    let dag = create_dag();

    dag.register_periodic_reactor::<_, (EKFOdometry,)>(
        "ekf_localizer".into(),
        move || -> (EKFOdometry,) {
            let poses = POSE_QUEUE.drain();
            let mut twists = TWIST_QUEUE.drain();
            (nodes::ekf_localizer(
                nodes::EkfDt::Measured {
                    nominal: period_sec,
                },
                &poses,
                &mut twists,
            ),)
        },
        vec![Cow::from("/localization/pose_twist_fusion_filter/kinematic_state")],
        SchedulerType::GEDF(5),
        EKF_PERIOD,
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

    dag.register_sink_reactor::<_, (KinematicState, EKFOdometry, AccelWithCovarianceStamped)>(
        "kinematic_sink".into(),
        |(ks, odom, accel): (KinematicState, EKFOdometry, AccelWithCovarianceStamped)| {
            LATEST_KINEMATIC_STATE.set(ks);
            LATEST_ODOMETRY.set(odom);
            LATEST_ACCEL.set(accel);
        },
        vec![
            Cow::from("/localization/kinematic_state"),
            Cow::from("/localization/pose_twist_fusion_filter/kinematic_state"),
            Cow::from("/localization/acceleration"),
        ],
        SchedulerType::GEDF(5),
        EKF_PERIOD,
    )
    .await;

    dag
}

async fn control_dag() -> Arc<Dag> {
    let dag = create_dag();

    // Upstream's onTimer returns without publishing until odometry has arrived; `None` is
    // that "no message" case and is carried down the chain.
    dag.register_periodic_reactor::<_, (Option<Control>,)>(
        "trajectory_follower".into(),
        || -> (Option<Control>,) {
            let control = LATEST_KINEMATIC_STATE
                .get()
                .map(|ks| nodes::trajectory_follower(&ks));
            (control,)
        },
        vec![Cow::from("/control/trajectory_follower/control_cmd")],
        SchedulerType::GEDF(5),
        CONTROL_PERIOD,
    )
    .await;

    // The trigger is an event input upstream; here it is read inside the node (no extra source).
    dag.register_reactor::<_, (Option<Control>,), MaybeCommands>(
        "constant_jerk_deceleration_controller".into(),
        |(control,): (Option<Control>,)| -> MaybeCommands {
            match control {
                Some(control) => {
                    let out = nodes::constant_jerk_deceleration_controller(
                        &control,
                        &nodes::mrm_trigger(),
                    );
                    some_commands((out.control, out.gear, out.hazard_lights, out.turn_indicators))
                }
                None => (None, None, None, None),
            }
        },
        vec![Cow::from("/control/trajectory_follower/control_cmd")],
        vec![
            Cow::from("/control/control_command_gate/inputs/in_lane_stop/control"),
            Cow::from("/control/control_command_gate/inputs/in_lane_stop/gear"),
            Cow::from("/control/control_command_gate/inputs/in_lane_stop/hazard_lights"),
            Cow::from("/control/control_command_gate/inputs/in_lane_stop/turn_indicators"),
        ],
        SchedulerType::GEDF(5),
    )
    .await;

    // Without a "main" command this period the gate outputs its builtin source, as upstream
    // where the builtin source is selected until "main" is.
    dag.register_reactor::<_, MaybeCommands, MaybeCommands>(
        "control_command_gate".into(),
        |(control, gear, hazard_lights, turn_indicators): MaybeCommands| -> MaybeCommands {
            let main = match (control, gear, hazard_lights, turn_indicators) {
                (Some(c), Some(g), Some(h), Some(t)) => Some((c, g, h, t)),
                _ => None,
            };
            let ks = LATEST_KINEMATIC_STATE.get().unwrap_or_default();
            some_commands(nodes::control_command_gate(main, &ks))
        },
        vec![
            Cow::from("/control/control_command_gate/inputs/in_lane_stop/control"),
            Cow::from("/control/control_command_gate/inputs/in_lane_stop/gear"),
            Cow::from("/control/control_command_gate/inputs/in_lane_stop/hazard_lights"),
            Cow::from("/control/control_command_gate/inputs/in_lane_stop/turn_indicators"),
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

    dag.register_sink_reactor::<_, MaybeCommands>(
        "vehicle_cmd_sink".into(),
        |(control, gear, hazard_lights, turn_indicators): MaybeCommands| {
            if let (Some(c), Some(g), Some(h), Some(t)) =
                (control, gear, hazard_lights, turn_indicators)
            {
                let odom = LATEST_ODOMETRY.get();
                nodes::vehicle_cmd_sink(
                    "multi",
                    &c,
                    &g,
                    &h,
                    &t,
                    odom.as_ref(),
                    Some((TWIST_QUEUE.dropped(), POSE_QUEUE.dropped())),
                );
            }
        },
        vec![
            Cow::from("/control/command/control_cmd"),
            Cow::from("/control/command/gear_cmd"),
            Cow::from("/control/command/hazard_lights_cmd"),
            Cow::from("/control/command/turn_indicators_cmd"),
        ],
        SchedulerType::GEDF(5),
        CONTROL_PERIOD,
    )
    .await;

    dag
}

