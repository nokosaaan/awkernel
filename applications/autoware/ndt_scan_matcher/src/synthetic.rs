//! Procedural point-cloud world: stand-in for the PCD map (`pointcloud_map_loader`) and the
//! downsampled LiDAR scan (`/localization/util/downsample/pointcloud`) until real data is embedded.
//!
//! The world is an infinite grid of `TILE_SIZE` square tiles. Each tile holds a flat ground grid
//! plus 4x4 box pillars whose size and height vary per pillar (hash of the indices), so the
//! scan is not exactly periodic and NDT has x / y / yaw structure to lock onto.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// Edge length of one map tile [m] (Autoware's sample maps also use 20 m grid tiles).
pub const TILE_SIZE: f64 = 20.0;

const GROUND_STEP: f64 = 0.5;
const PILLAR_PITCH: f64 = 5.0;
const PILLARS_PER_AXIS: i32 = 4;
const FACE_STEP: f64 = 0.1;
const HEIGHT_STEP: f64 = 0.2;

/// Index of the tile containing `(x, y)`.
pub fn tile_index(x: f64, y: f64) -> (i32, i32) {
    (
        libm::floor(x / TILE_SIZE) as i32,
        libm::floor(y / TILE_SIZE) as i32,
    )
}

/// Byte key for `NdtEngine::add_target` / `remove_target`.
pub fn tile_id(ix: i32, iy: i32) -> [u8; 8] {
    let mut id = [0u8; 8];
    let (lo, hi) = id.split_at_mut(4);
    lo.copy_from_slice(&ix.to_le_bytes());
    hi.copy_from_slice(&iy.to_le_bytes());
    id
}

/// Tiles whose center lies within `radius` of `(cx, cy)` (the differential map loader's query),
/// sorted by `(ix, iy)`.
pub fn tiles_within(cx: f64, cy: f64, radius: f64) -> Vec<(i32, i32)> {
    let (min_x, min_y) = tile_index(cx - radius, cy - radius);
    let (max_x, max_y) = tile_index(cx + radius, cy + radius);
    let mut out = Vec::new();
    for ix in min_x..=max_x {
        for iy in min_y..=max_y {
            let tx = (f64::from(ix) + 0.5) * TILE_SIZE - cx;
            let ty = (f64::from(iy) + 0.5) * TILE_SIZE - cy;
            if libm::sqrt(tx * tx + ty * ty) <= radius {
                out.push((ix, iy));
            }
        }
    }
    out
}

/// Small deterministic hash in [0, 1) used to vary pillar shapes.
fn hash01(a: i32, b: i32, c: i32) -> f64 {
    let mut h = (a as u32).wrapping_mul(0x9E37_79B1)
        ^ (b as u32).wrapping_mul(0x85EB_CA77)
        ^ (c as u32).wrapping_mul(0xC2B2_AE3D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    f64::from(h & 0xFFFF) / 65536.0
}

/// Map-frame points of one tile.
pub fn tile_points(ix: i32, iy: i32) -> Vec<[f32; 3]> {
    let x0 = f64::from(ix) * TILE_SIZE;
    let y0 = f64::from(iy) * TILE_SIZE;
    let mut pts = Vec::new();

    let n_ground = (TILE_SIZE / GROUND_STEP) as i32;
    for gx in 0..n_ground {
        for gy in 0..n_ground {
            pts.push([
                (x0 + f64::from(gx) * GROUND_STEP) as f32,
                (y0 + f64::from(gy) * GROUND_STEP) as f32,
                0.0,
            ]);
        }
    }

    for i in 0..PILLARS_PER_AXIS {
        for j in 0..PILLARS_PER_AXIS {
            let cx = x0 + (f64::from(i) + 0.5) * PILLAR_PITCH;
            let cy = y0 + (f64::from(j) + 0.5) * PILLAR_PITCH;
            let half = 0.2 + 0.3 * hash01(ix * 4 + i, iy * 4 + j, 1);
            let height = 1.0 + 2.0 * hash01(ix * 4 + i, iy * 4 + j, 2);
            push_box_faces(&mut pts, cx, cy, half, height);
        }
    }
    pts
}

fn push_box_faces(pts: &mut Vec<[f32; 3]>, cx: f64, cy: f64, half: f64, height: f64) {
    let n_face = libm::ceil(2.0 * half / FACE_STEP) as i32;
    let n_height = libm::ceil(height / HEIGHT_STEP) as i32;
    for k in 0..=n_face {
        let s = -half + 2.0 * half * f64::from(k) / f64::from(n_face.max(1));
        for h in 1..=n_height {
            let z = (f64::from(h) * HEIGHT_STEP).min(height) as f32;
            pts.push([(cx + s) as f32, (cy - half) as f32, z]);
            pts.push([(cx + s) as f32, (cy + half) as f32, z]);
            pts.push([(cx - half) as f32, (cy + s) as f32, z]);
            pts.push([(cx + half) as f32, (cy + s) as f32, z]);
        }
    }
}

/// Planar sensor pose used to synthesise a scan: map -> base_link.
#[derive(Clone, Copy, Debug)]
pub struct PlanarPose {
    pub x: f64,
    pub y: f64,
    pub yaw: f64,
}

/// Simulated downsampled LiDAR scan, in base_link, taken at `truth` (uncached; see [`World`]).
pub fn generate_scan(truth: PlanarPose, range: f64, max_points: usize) -> Vec<[f32; 3]> {
    World::new().scan(truth, range, max_points)
}

/// Tile cache for the synthetic sensor, so a scan does not regenerate every tile per frame.
#[derive(Default)]
pub struct World {
    tiles: BTreeMap<(i32, i32), Vec<[f32; 3]>>,
}

impl World {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cached_tile_count(&self) -> usize {
        self.tiles.len()
    }

    /// All map points within `range` (2-D distance) of `truth`, transformed into base_link and
    /// decimated with a fixed stride to at most `max_points` (a crude stand-in for the
    /// voxel-grid downsample). Tiles far from `truth` are evicted from the cache.
    pub fn scan(&mut self, truth: PlanarPose, range: f64, max_points: usize) -> Vec<[f32; 3]> {
        // A tile intersects the disc iff its center is within range + half its diagonal.
        let reach = range + TILE_SIZE * core::f64::consts::FRAC_1_SQRT_2;
        let needed = tiles_within(truth.x, truth.y, reach);
        self.tiles
            .retain(|&(ix, iy), _| needed.binary_search(&(ix, iy)).is_ok());

        let (c, s) = (libm::cos(truth.yaw), libm::sin(truth.yaw));
        let mut in_range = Vec::new();
        for &(ix, iy) in &needed {
            let points = self
                .tiles
                .entry((ix, iy))
                .or_insert_with(|| tile_points(ix, iy));
            for p in points.iter() {
                let dx = f64::from(p[0]) - truth.x;
                let dy = f64::from(p[1]) - truth.y;
                if dx * dx + dy * dy > range * range {
                    continue;
                }
                // R(yaw)^T * (p - t)
                in_range.push([(c * dx + s * dy) as f32, (-s * dx + c * dy) as f32, p[2]]);
            }
        }
        if max_points == 0 || in_range.len() <= max_points {
            return in_range;
        }
        let stride = in_range.len().div_ceil(max_points);
        in_range.into_iter().step_by(stride).collect()
    }
}
