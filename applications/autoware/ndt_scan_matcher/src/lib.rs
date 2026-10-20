//! NDT scan matcher node for the awkernel Autoware DAG.
//!
//! The numeric engine is `tier4/realtime_core::realtime_ndt_scan_matcher` (the ROS-free
//! no_std core behind autoware_core#1236). This crate is the thin node layer that the ROS
//! node (`ndt_scan_matcher_core.cpp` / the PR's `node.rs`) provides around it:
//!
//! | Autoware node step (callback_sensor_points) | here |
//! |---|---|
//! | sensor points empty / max distance < `required_distance` -> skip | [`NdtScanMatcher::align`] gates |
//! | initial pose from EKF `biased_pose_with_covariance` (SmartPoseBuffer interpolation) | latest EKF pose, no interpolation |
//! | `ndt_ptr_->align(...)` + convergence verdict | `NdtEngine::align_outcome` |
//! | not converged -> do not publish the pose | returns `Ok(None)` |
//! | covariance (`covariance_estimation_type`) | `NdtEngine::estimate_covariance` (FIXED_VALUE) |
//! | dynamic map loading (`update_distance` / `map_radius`) | [`NdtScanMatcher::update_map`] on synthetic tiles |
//!
//! Not ported: TF lookup (the scan is already in base_link), regularization (disabled by
//! default), the TPE `ndt_align_srv`, the `trigger_node` service, and diagnostics.
#![no_std]
extern crate alloc;

pub mod synthetic;

use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use ekf_localizer::{Point3D, Pose, PoseWithCovariance, PoseWithCovarianceStamped, Quaternion};
use realtime_ndt_scan_matcher::engine::{MatchScratch, NdtEngine};
use realtime_ndt_scan_matcher::nalgebra as na;
pub use realtime_ndt_scan_matcher::ndt::AlignError;

/// Downsampled sensor point cloud in base_link (`sensor_msgs/PointCloud2`, xyz only).
#[derive(Clone, Debug)]
pub struct PointCloud {
    pub header: common_types::Header,
    pub points: Vec<[f32; 3]>,
}

/// Node parameters. Defaults are `ndt_scan_matcher.param.yaml` from autoware_core
/// (`num_threads` forced to 1: the no_std build is serial).
#[derive(Clone, Debug)]
pub struct NdtParams {
    pub trans_epsilon: f64,
    pub step_size: f64,
    pub resolution: f64,
    pub max_iterations: i32,
    /// Not in the yaml; the pclomp / C++ `NdtParams` default.
    pub outlier_ratio: f64,
    pub converged_param_type: i32,
    pub converged_param_transform_probability: f64,
    pub converged_param_nearest_voxel_transformation_likelihood: f64,
    pub required_distance: f64,
    pub output_pose_covariance: [f64; 36],
    pub update_distance: f64,
    pub map_radius: f64,
    /// Work envelope of the engine (`Pmax` / `Lmax`).
    pub max_source_points: usize,
    pub max_active_leaves: usize,
}

impl Default for NdtParams {
    fn default() -> Self {
        let mut cov = [0.0; 36];
        for (i, v) in [0.0225, 0.0225, 0.0225, 0.000625, 0.000625, 0.000625]
            .into_iter()
            .enumerate()
        {
            if let Some(slot) = cov.get_mut(i * 7) {
                *slot = v;
            }
        }
        Self {
            trans_epsilon: 0.01,
            step_size: 0.1,
            resolution: 2.0,
            max_iterations: 30,
            outlier_ratio: 0.55,
            converged_param_type: 1,
            converged_param_transform_probability: 3.0,
            converged_param_nearest_voxel_transformation_likelihood: 2.3,
            required_distance: 10.0,
            output_pose_covariance: cov,
            update_distance: 20.0,
            // 150 m upstream; the synthetic world only needs the scan range plus a margin.
            map_radius: 60.0,
            max_source_points: 2_000,
            max_active_leaves: 50_000,
        }
    }
}

/// Why a scan produced no pose (mirrors the early returns / non-published cases upstream).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Skip {
    EmptyScan,
    ScanTooShort,
    NoMap,
    NotConverged,
}

/// Per-scan result kept for logging.
#[derive(Clone, Copy, Debug)]
pub struct AlignReport {
    pub iteration_num: i32,
    pub transform_probability: f32,
    pub nearest_voxel_likelihood: f32,
    pub converged: bool,
}

pub struct NdtScanMatcher {
    params: NdtParams,
    engine: NdtEngine,
    scratch: MatchScratch,
    loaded_tiles: BTreeSet<(i32, i32)>,
    last_update_position: Option<[f64; 2]>,
    last_report: Option<AlignReport>,
}

impl NdtScanMatcher {
    pub fn new(params: NdtParams) -> Result<Self, AlignError> {
        let engine = NdtEngine::new(
            params.resolution,
            6,
            0.01,
            params.max_source_points,
            params.max_active_leaves,
            params.max_iterations,
        )?;
        engine.set_params(
            params.trans_epsilon,
            params.step_size,
            params.resolution,
            params.max_iterations,
            params.outlier_ratio,
            1,
        );
        engine.set_convergence_params(
            params.converged_param_type,
            params.converged_param_transform_probability,
            params.converged_param_nearest_voxel_transformation_likelihood,
        );
        // covariance_estimation_type 0 (FIXED_VALUE): offsets / temperature unused.
        engine.set_covariance_config(0, 1.0, 0.05, params.output_pose_covariance, &[], &[]);
        let scratch = MatchScratch::try_for_limits(engine.limits())?;
        Ok(Self {
            params,
            engine,
            scratch,
            loaded_tiles: BTreeSet::new(),
            last_update_position: None,
            last_report: None,
        })
    }

    pub fn last_report(&self) -> Option<AlignReport> {
        self.last_report
    }

    pub fn loaded_tile_count(&self) -> usize {
        self.loaded_tiles.len()
    }

    /// Differential map update around `(x, y)`, run only after moving `update_distance`
    /// (upstream `callback_timer` -> `update_ndt`). Returns whether the map changed.
    pub fn update_map(&mut self, x: f64, y: f64) -> Result<bool, AlignError> {
        if let Some([px, py]) = self.last_update_position {
            let (dx, dy) = (x - px, y - py);
            if libm::sqrt(dx * dx + dy * dy) < self.params.update_distance {
                return Ok(false);
            }
        }
        let wanted: BTreeSet<(i32, i32)> =
            synthetic::tiles_within(x, y, self.params.map_radius).into_iter().collect();
        for &(ix, iy) in self.loaded_tiles.difference(&wanted) {
            self.engine.remove_target(&synthetic::tile_id(ix, iy));
        }
        for &(ix, iy) in wanted.difference(&self.loaded_tiles) {
            self.engine
                .add_target(&synthetic::tile_points(ix, iy), &synthetic::tile_id(ix, iy));
        }
        self.engine.create_kdtree()?;
        self.loaded_tiles = wanted;
        self.last_update_position = Some([x, y]);
        Ok(true)
    }

    /// One `callback_sensor_points`: align `scan` from `initial_pose` and return the pose
    /// to publish on `/localization/pose_estimator/pose_with_covariance`, or why there is none.
    pub fn align(
        &mut self,
        scan: &PointCloud,
        initial_pose: &Pose,
    ) -> Result<Result<PoseWithCovarianceStamped, Skip>, AlignError> {
        if scan.points.is_empty() {
            return Ok(Err(Skip::EmptyScan));
        }
        let max_distance = scan
            .points
            .iter()
            .map(|p| libm::sqrtf(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]))
            .fold(0.0f32, f32::max);
        if f64::from(max_distance) < self.params.required_distance {
            return Ok(Err(Skip::ScanTooShort));
        }
        if !self.engine.has_target() {
            return Ok(Err(Skip::NoMap));
        }

        let guess = pose_to_matrix(initial_pose);
        let outcome = self
            .engine
            .align_outcome(&guess, &scan.points, &mut self.scratch)?;
        self.last_report = Some(AlignReport {
            iteration_num: outcome.iteration_num,
            transform_probability: outcome.transform_probability,
            nearest_voxel_likelihood: outcome.nearest_voxel_likelihood,
            converged: outcome.verdict.is_converged,
        });
        if !outcome.verdict.is_converged {
            return Ok(Err(Skip::NotConverged));
        }

        let hessian = self.scratch.result_ref().hessian;
        let cov = self.engine.estimate_covariance(
            &outcome.pose,
            &hessian,
            &guess,
            &scan.points,
            outcome.nearest_voxel_likelihood,
            &mut self.scratch,
        )?;
        Ok(Ok(PoseWithCovarianceStamped {
            header: common_types::Header {
                frame_id: "map",
                timestamp: scan.header.timestamp,
            },
            pose: PoseWithCovariance {
                pose: matrix_to_pose(&outcome.pose),
                covariance: cov.ndt_covariance,
            },
        }))
    }
}

/// map -> base_link pose as the engine's homogeneous `Matrix4<f32>`.
pub fn pose_to_matrix(p: &Pose) -> na::Matrix4<f32> {
    let q = na::UnitQuaternion::from_quaternion(na::Quaternion::new(
        p.orientation.w,
        p.orientation.x,
        p.orientation.y,
        p.orientation.z,
    ));
    let iso = na::Isometry3::from_parts(
        na::Translation3::new(p.position.x, p.position.y, p.position.z),
        q,
    );
    iso.to_homogeneous().cast::<f32>()
}

/// Inverse of [`pose_to_matrix`] (re-orthonormalises the f32 rotation block).
pub fn matrix_to_pose(m: &na::Matrix4<f32>) -> Pose {
    let m = m.cast::<f64>();
    let rot: na::Matrix3<f64> = m.fixed_view::<3, 3>(0, 0).into_owned();
    let t: na::Vector3<f64> = m.fixed_view::<3, 1>(0, 3).into_owned();
    let q = na::UnitQuaternion::from_rotation_matrix(&na::Rotation3::from_matrix(&rot));
    Pose {
        position: Point3D {
            x: t.x,
            y: t.y,
            z: t.z,
        },
        orientation: Quaternion {
            x: q.i,
            y: q.j,
            z: q.k,
            w: q.w,
        },
    }
}

/// Yaw of a pose's orientation (z-up), for logging and the synthetic sensor.
pub fn yaw_of(p: &Pose) -> f64 {
    let q = &p.orientation;
    libm::atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )
}

/// Planar pose with the given yaw (roll = pitch = 0, z = 0).
pub fn planar_pose(x: f64, y: f64, yaw: f64) -> Pose {
    Pose {
        position: Point3D { x, y, z: 0.0 },
        orientation: Quaternion {
            x: 0.0,
            y: 0.0,
            z: libm::sin(yaw / 2.0),
            w: libm::cos(yaw / 2.0),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use synthetic::{generate_scan, PlanarPose};

    fn scan_at(x: f64, y: f64, yaw: f64) -> PointCloud {
        PointCloud {
            header: common_types::Header {
                frame_id: "base_link",
                timestamp: 42,
            },
            points: generate_scan(PlanarPose { x, y, yaw }, 30.0, 2_000),
        }
    }

    #[test]
    fn pose_matrix_round_trip() {
        let p = planar_pose(3.0, -2.0, 0.7);
        let back = matrix_to_pose(&pose_to_matrix(&p));
        assert!((back.position.x - 3.0).abs() < 1e-5);
        assert!((back.position.y + 2.0).abs() < 1e-5);
        assert!((yaw_of(&back) - 0.7).abs() < 1e-5);
    }

    #[test]
    fn cached_world_scan_matches_uncached_and_evicts() {
        let mut w = synthetic::World::new();
        let a = PlanarPose { x: 3.0, y: -1.0, yaw: 0.4 };
        assert_eq!(w.scan(a, 30.0, 1_500), generate_scan(a, 30.0, 1_500));
        let near = w.cached_tile_count();
        let far = PlanarPose { x: 500.0, y: 0.0, yaw: 0.0 };
        assert_eq!(w.scan(far, 30.0, 1_500), generate_scan(far, 30.0, 1_500));
        // Tiles around the first pose were evicted, so the cache did not simply grow.
        assert!(w.cached_tile_count() <= near + 1);
    }

    #[test]
    fn tile_ids_are_unique_and_signed() {
        assert_ne!(synthetic::tile_id(-1, 0), synthetic::tile_id(0, -1));
        assert_eq!(synthetic::tile_index(-0.1, 19.9), (-1, 0));
    }

    #[test]
    fn corrects_offset_initial_guess_to_truth() {
        let mut m = NdtScanMatcher::new(NdtParams::default()).expect("engine");
        assert!(m.update_map(5.0, 1.0).expect("map"));
        let (tx, ty, tyaw) = (5.0, 1.0, 0.15);
        let scan = scan_at(tx, ty, tyaw);
        assert!(scan.points.len() <= 2_000 && scan.points.len() > 500);

        let guess = planar_pose(tx + 0.4, ty - 0.3, tyaw - 0.03);
        let out = m.align(&scan, &guess).expect("align").expect("converged");
        let p = &out.pose.pose;
        assert!((p.position.x - tx).abs() < 0.05, "x = {}", p.position.x);
        assert!((p.position.y - ty).abs() < 0.05, "y = {}", p.position.y);
        assert!((yaw_of(p) - tyaw).abs() < 0.01, "yaw = {}", yaw_of(p));
        assert_eq!(out.header.timestamp, 42);
        // FIXED_VALUE covariance with a yaw rotation keeps the isotropic xy diagonal.
        assert!((out.pose.covariance[0] - 0.0225).abs() < 1e-9);
        assert!(m.last_report().is_some_and(|r| r.converged));
    }

    #[test]
    fn map_update_waits_for_update_distance() {
        let mut m = NdtScanMatcher::new(NdtParams::default()).expect("engine");
        assert!(m.update_map(0.0, 0.0).expect("map"));
        let n = m.loaded_tile_count();
        assert!(!m.update_map(10.0, 0.0).expect("map"));
        assert!(m.update_map(25.0, 0.0).expect("map"));
        assert!(m.loaded_tile_count() >= n.saturating_sub(4));
    }

    #[test]
    fn gates_mirror_upstream_skips() {
        let mut m = NdtScanMatcher::new(NdtParams::default()).expect("engine");
        let empty = PointCloud {
            header: common_types::Header {
                frame_id: "base_link",
                timestamp: 0,
            },
            points: Vec::new(),
        };
        let origin = planar_pose(0.0, 0.0, 0.0);
        assert_eq!(m.align(&empty, &origin).expect("ok").err(), Some(Skip::EmptyScan));
        let short = PointCloud {
            points: alloc::vec![[1.0, 0.0, 0.0]],
            ..empty.clone()
        };
        assert_eq!(m.align(&short, &origin).expect("ok").err(), Some(Skip::ScanTooShort));
        assert_eq!(
            m.align(&scan_at(0.0, 0.0, 0.0), &origin).expect("ok").err(),
            Some(Skip::NoMap)
        );
    }
}

