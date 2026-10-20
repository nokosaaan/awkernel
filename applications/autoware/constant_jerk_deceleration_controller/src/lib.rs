//! Constant-jerk deceleration controller: MRM in-lane-stop helper that sits between
//! the trajectory follower and the control command gate.
//!
//! Ported from autoware_universe#12771
//! (control/autoware_constant_jerk_deceleration_controller, constant_jerk_deceleration_controller.cpp).
//!
//! - trigger off: the upstream `Control` is passed through unchanged.
//! - trigger on: velocity / acceleration / jerk are overridden by a constant-jerk ramp
//!   down to `target_acceleration`.
//! - every control message also emits gear = DRIVE, hazard lights = ENABLE and
//!   turn indicators = DISABLE.
//!
//! ## Faithfulness notes
//! - The ramp integrates with `prev_control_.longitudinal.jerk`, as the C++ code does (the
//!   upstream README says `target_jerk`; the code wins). On the first override step the
//!   acceleration therefore moves by the upstream jerk, and `target_jerk` only acts from
//!   step 2.
//! - `Longitudinal` fields are `float32` in ROS but `f64` in `localization_types`, so the
//!   arithmetic re-creates the C++ promotions/truncations explicitly (`as_f32`).
//! - `std::max<float>(a, b)` returns `a` unless `a < b`; this is reproduced with an explicit
//!   comparison rather than `f32::max`, whose NaN handling differs.
//! - The C++ node has no parameters, no timer and no dt guard; none are added here.
#![no_std]

use awkernel_lib::sync::{mcs::MCSNode, mutex::Mutex};
use awkernel_lib::time::Instant;
use localization_types::{
    ConstantJerkDecelerationTrigger, Control, GearCommand, HazardLightsCommand,
    TurnIndicatorsCommand,
};

/// Everything one `onControl` callback publishes, in the C++ publish order.
#[derive(Clone)]
pub struct ControllerOutput {
    pub control: Control,
    pub gear: GearCommand,
    pub hazard_lights: HazardLightsCommand,
    pub turn_indicators: TurnIndicatorsCommand,
}

pub struct ConstantJerkDecelerationController {
    trigger: ConstantJerkDecelerationTrigger,
    prev_control: Control,
}

impl Default for ConstantJerkDecelerationController {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstantJerkDecelerationController {
    /// Mirrors the constructor's `// State` block: trigger false, targets 0, prev v/a/j 0,
    /// prev stamp 0.
    pub fn new() -> Self {
        Self {
            trigger: ConstantJerkDecelerationTrigger::default(),
            prev_control: Control::default(),
        }
    }

    /// Mirrors `onTrigger`: the whole message replaces the stored trigger (its stamp is unused).
    pub fn on_trigger(&mut self, msg: &ConstantJerkDecelerationTrigger) {
        self.trigger = *msg;
    }

    /// Mirrors `onControl`. `now` stands in for `this->now()`, used to stamp the
    /// gear / hazard lights / turn indicators commands.
    pub fn on_control(&mut self, msg: &Control, now: Instant) -> ControllerOutput {
        let control = if !self.trigger.trigger {
            self.prev_control = msg.clone();
            msg.clone()
        } else {
            let dt = signed_seconds(msg.stamp, self.prev_control.stamp);
            let prev = &self.prev_control.longitudinal;
            // Stored as float32 in the C++ message; promoting f32 -> f64 is exact.
            let prev_v = f64::from(as_f32(prev.velocity));
            let prev_a = f64::from(as_f32(prev.acceleration));
            let prev_j = f64::from(as_f32(prev.jerk));
            let target_acceleration = self.trigger.target_acceleration;

            let mut control_msg = msg.clone();
            let velocity = as_f32(max_f64(prev_v + prev_a * dt, 0.0));
            let acceleration = std_max_f32(as_f32(prev_a + prev_j * dt), target_acceleration);
            let jerk = if acceleration == target_acceleration {
                0.0
            } else {
                self.trigger.target_jerk
            };
            control_msg.longitudinal.velocity = f64::from(velocity);
            control_msg.longitudinal.acceleration = f64::from(acceleration);
            control_msg.longitudinal.jerk = f64::from(jerk);

            self.prev_control = control_msg.clone();
            control_msg
        };

        ControllerOutput {
            control,
            gear: GearCommand {
                stamp: now,
                command: GearCommand::DRIVE,
            },
            hazard_lights: HazardLightsCommand {
                stamp: now,
                command: HazardLightsCommand::ENABLE,
            },
            turn_indicators: TurnIndicatorsCommand {
                stamp: now,
                command: TurnIndicatorsCommand::DISABLE,
            },
        }
    }
}

/// `(rclcpp::Time(a) - rclcpp::Time(b)).seconds()`: signed nanosecond difference / 1e9.
fn signed_seconds(a: Instant, b: Instant) -> f64 {
    if a >= b {
        a.saturating_duration_since(b).as_nanos() as f64 / 1.0e9
    } else {
        -(b.saturating_duration_since(a).as_nanos() as f64 / 1.0e9)
    }
}

/// `static_cast<float>(double)`: round to nearest f32 (the truncation is the point).
#[inline]
fn as_f32(x: f64) -> f32 {
    x as f32
}

/// `std::max<double>(a, b)`: `a` unless `a < b`.
#[inline]
fn max_f64(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}

/// `std::max<float>(a, b)`: `a` unless `a < b` (keeps NaN in `a`, ignores NaN in `b`).
#[inline]
fn std_max_f32(a: f32, b: f32) -> f32 {
    if a < b {
        b
    } else {
        a
    }
}

// ---------------------------------------------------------------------------
// Node-level singleton (same pattern as the other ported nodes)
// ---------------------------------------------------------------------------

static STATE: Mutex<Option<ConstantJerkDecelerationController>> = Mutex::new(None);

/// Node entry for the trigger subscription.
pub fn on_trigger(msg: &ConstantJerkDecelerationTrigger) {
    let mut node = MCSNode::new();
    let mut st = STATE.lock(&mut node);
    st.get_or_insert_with(ConstantJerkDecelerationController::new)
        .on_trigger(msg);
}

/// Node entry for the control subscription.
pub fn on_control(msg: &Control) -> ControllerOutput {
    let now = Instant::now();
    let mut node = MCSNode::new();
    let mut st = STATE.lock(&mut node);
    st.get_or_insert_with(ConstantJerkDecelerationController::new)
        .on_control(msg, now)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::time::Duration;

    fn at_ms(ms: u64) -> Instant {
        Instant::zero() + Duration::from_millis(ms)
    }

    fn control(ms: u64, v: f64, a: f64, j: f64) -> Control {
        let mut c = Control::default();
        c.stamp = at_ms(ms);
        c.lateral.steering_tire_angle = 0.1;
        c.longitudinal.velocity = v;
        c.longitudinal.acceleration = a;
        c.longitudinal.jerk = j;
        c
    }

    fn trigger(on: bool, target_acceleration: f32, target_jerk: f32) -> ConstantJerkDecelerationTrigger {
        ConstantJerkDecelerationTrigger {
            stamp: Instant::zero(),
            trigger: on,
            target_acceleration,
            target_jerk,
        }
    }

    #[test]
    fn pass_through_copies_input_and_emits_fixed_commands() {
        let mut c = ConstantJerkDecelerationController::new();
        let now = at_ms(7);
        let out = c.on_control(&control(100, 5.123456789, 0.3, 0.01), now);
        // Untouched: no f32 rounding on pass-through.
        assert_eq!(out.control.longitudinal.velocity, 5.123456789);
        assert_eq!(out.control.longitudinal.acceleration, 0.3);
        assert_eq!(out.control.lateral.steering_tire_angle, 0.1);
        assert_eq!(out.gear, GearCommand { stamp: now, command: 2 });
        assert_eq!(out.hazard_lights, HazardLightsCommand { stamp: now, command: 2 });
        assert_eq!(out.turn_indicators, TurnIndicatorsCommand { stamp: now, command: 1 });
    }

    #[test]
    fn first_override_step_integrates_with_upstream_jerk_not_target_jerk() {
        let mut c = ConstantJerkDecelerationController::new();
        c.on_control(&control(0, 10.0, 0.0, 0.0), at_ms(0));
        c.on_trigger(&trigger(true, -2.0, -1.0));

        // dt = 0.1 s, prev a = 0, prev j = 0: velocity and acceleration unchanged.
        let out = c.on_control(&control(100, 99.0, 9.0, 9.0), at_ms(100));
        assert_eq!(out.control.longitudinal.velocity, 10.0);
        assert_eq!(out.control.longitudinal.acceleration, 0.0);
        assert_eq!(out.control.longitudinal.jerk, -1.0);
        // Non-longitudinal fields come from the incoming message.
        assert_eq!(out.control.stamp, at_ms(100));

        // Step 2: target_jerk now acts. a = 0 + (-1) * 0.1.
        let out = c.on_control(&control(200, 99.0, 9.0, 9.0), at_ms(200));
        assert_eq!(out.control.longitudinal.velocity, 10.0);
        assert_eq!(
            out.control.longitudinal.acceleration,
            f64::from((0.0f64 + -1.0 * 0.1) as f32)
        );
        assert_eq!(out.control.longitudinal.jerk, -1.0);
    }

    #[test]
    fn ramp_clamps_at_target_and_jerk_becomes_zero() {
        let mut c = ConstantJerkDecelerationController::new();
        c.on_control(&control(0, 10.0, 0.0, 0.0), at_ms(0));
        c.on_trigger(&trigger(true, -1.0, -2.0));
        // Step 1 keeps a = 0 (prev jerk 0); step 2 reaches the clamp exactly: 0 + -2 * 0.5.
        let out = c.on_control(&control(500, 0.0, 0.0, 0.0), at_ms(500));
        assert_eq!(out.control.longitudinal.acceleration, 0.0);
        assert_eq!(out.control.longitudinal.jerk, -2.0);
        let out = c.on_control(&control(1000, 0.0, 0.0, 0.0), at_ms(1000));
        assert_eq!(out.control.longitudinal.acceleration, -1.0);
        assert_eq!(out.control.longitudinal.jerk, 0.0);
        // Further steps hold the target; a -1 + 0 * dt stays at -1.
        let out = c.on_control(&control(1500, 0.0, 0.0, 0.0), at_ms(1500));
        assert_eq!(out.control.longitudinal.acceleration, -1.0);
        assert_eq!(out.control.longitudinal.jerk, 0.0);
    }

    #[test]
    fn upstream_harder_braking_snaps_up_to_target() {
        let mut c = ConstantJerkDecelerationController::new();
        c.on_control(&control(0, 10.0, -5.0, 0.0), at_ms(0));
        c.on_trigger(&trigger(true, -2.0, -1.0));
        let out = c.on_control(&control(100, 0.0, 0.0, 0.0), at_ms(100));
        assert_eq!(out.control.longitudinal.acceleration, -2.0);
        assert_eq!(out.control.longitudinal.jerk, 0.0);
        assert_eq!(out.control.longitudinal.velocity, f64::from((10.0f64 - 0.5) as f32));
    }

    #[test]
    fn velocity_clamps_at_zero_and_acceleration_is_kept() {
        let mut c = ConstantJerkDecelerationController::new();
        c.on_control(&control(0, 0.1, -2.0, 0.0), at_ms(0));
        c.on_trigger(&trigger(true, -2.0, -1.0));
        let out = c.on_control(&control(1000, 0.0, 0.0, 0.0), at_ms(1000));
        assert_eq!(out.control.longitudinal.velocity, 0.0);
        assert_eq!(out.control.longitudinal.acceleration, -2.0);
    }

    #[test]
    fn trigger_before_any_control_uses_dt_from_zero_stamp() {
        let mut c = ConstantJerkDecelerationController::new();
        c.on_trigger(&trigger(true, -2.0, -1.0));
        let out = c.on_control(&control(3000, 8.0, 1.0, 0.0), at_ms(3000));
        // prev v = a = j = 0 -> v = 0, a = max(0, -2) = 0, jerk = target_jerk.
        assert_eq!(out.control.longitudinal.velocity, 0.0);
        assert_eq!(out.control.longitudinal.acceleration, 0.0);
        assert_eq!(out.control.longitudinal.jerk, -1.0);
        // Next step: a = 0 + (-1) * 0.5.
        let out = c.on_control(&control(3500, 8.0, 1.0, 0.0), at_ms(3500));
        assert_eq!(out.control.longitudinal.acceleration, -0.5);
    }

    #[test]
    fn trigger_off_resumes_pass_through_and_on_again_restarts_from_latest_upstream() {
        let mut c = ConstantJerkDecelerationController::new();
        c.on_control(&control(0, 10.0, 0.0, 0.0), at_ms(0));
        c.on_trigger(&trigger(true, -2.0, -1.0));
        c.on_control(&control(100, 0.0, 0.0, 0.0), at_ms(100));
        c.on_control(&control(200, 0.0, 0.0, 0.0), at_ms(200));

        c.on_trigger(&trigger(false, -2.0, -1.0));
        let out = c.on_control(&control(300, 6.0, 0.5, 0.0), at_ms(300));
        assert_eq!(out.control.longitudinal.velocity, 6.0);
        assert_eq!(out.control.longitudinal.acceleration, 0.5);

        c.on_trigger(&trigger(true, -2.0, -1.0));
        let out = c.on_control(&control(400, 0.0, 0.0, 0.0), at_ms(400));
        // Integrates from the pass-through command at t = 300 ms.
        assert_eq!(out.control.longitudinal.velocity, f64::from((6.0f64 + 0.5 * 0.1) as f32));
        assert_eq!(out.control.longitudinal.acceleration, 0.5);
        assert_eq!(out.control.longitudinal.jerk, -1.0);
    }

    #[test]
    fn negative_dt_integrates_backwards_without_guard() {
        let mut c = ConstantJerkDecelerationController::new();
        c.on_control(&control(1000, 10.0, -1.0, 0.0), at_ms(1000));
        c.on_trigger(&trigger(true, -3.0, -1.0));
        let out = c.on_control(&control(500, 0.0, 0.0, 0.0), at_ms(500));
        assert_eq!(out.control.longitudinal.velocity, 10.5);
    }

    #[test]
    fn std_max_semantics_on_nan() {
        assert!(std_max_f32(f32::NAN, -1.0).is_nan());
        assert_eq!(std_max_f32(-0.5, f32::NAN), -0.5);
    }
}
