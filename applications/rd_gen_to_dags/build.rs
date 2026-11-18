//! Generates `$OUT_DIR/dag_files.rs`, a `DAG_FILES: &[&str]` constant with one
//! `include_str!` entry per `dag_<N>.yaml` found under [`DAGS_DIR`], instead
//! of requiring one hand-written `include_str!` constant per file in
//! `lib.rs`. Runs on the host with `std` regardless of the crate's own
//! `#![no_std]` (Cargo always builds `build.rs` as a normal host binary).
//!
//! For DAG-Fluid's time-driven dispatch (feature `dagfluid` without
//! `dagfluid_event`) it also generates `$OUT_DIR/fluid_table.rs`: it admits
//! the DAGs exactly as the kernel will (`src/fluid_offline.rs`, shared with
//! the kernel build) and plans one hyperperiod of DP-Wrap switch points
//! (`awkernel_async_lib::dag_sched::fluid_table::build`, the kernel's own
//! planning code) for `RD_GEN_FLUID_POOL_CPUS` pool CPUs.

extern crate alloc;

#[allow(dead_code)]
#[path = "src/dag_stats.rs"]
mod dag_stats;
#[path = "src/fluid_offline.rs"]
mod fluid_offline;
#[allow(dead_code)]
#[path = "src/parse_yaml.rs"]
mod parse_yaml;

use std::{
    env, fs,
    path::{Path, PathBuf},
    time::Duration,
};

/// Number of DAG-pool CPUs the offline table is planned for when
/// `RD_GEN_FLUID_POOL_CPUS` is unset: 16 CPUs (the evaluation machine with
/// `smt_disable`, and `make qemu-x86_64`) minus the primary CPU and the
/// regular-pool CPU. The kernel refuses a table planned for a different
/// pool size.
const DEFAULT_FLUID_POOL_CPUS: usize = 14;

/// Seconds per unit of the YAML's times, from the crate's time-unit feature
/// (milliseconds by default, as `src/time_unit.rs`).
fn units_per_sec() -> f64 {
    let has = |feature: &str| env::var_os(format!("CARGO_FEATURE_{feature}")).is_some();
    if has("SECONDS") {
        1.0
    } else if has("MICROSECONDS") {
        1_000_000.0
    } else if has("NANOSECONDS") {
        1_000_000_000.0
    } else {
        1_000.0
    }
}

/// Write `$OUT_DIR/fluid_table.rs` for the DAG files `contents` (in
/// `DAG_FILES` order).
fn generate_fluid_table(contents: &[String], out_dir: &Path) {
    use awkernel_async_lib::dag_sched::fluid_table;

    println!("cargo:rerun-if-env-changed=RD_GEN_FLUID_POOL_CPUS");
    let pool_cpus = env::var("RD_GEN_FLUID_POOL_CPUS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_FLUID_POOL_CPUS);
    let per_node = env::var_os("CARGO_FEATURE_DAGFLUID_NODE").is_some();
    let units = units_per_sec();
    // Same conversion as `time_unit::convert_duration_f64`.
    let unit = move |v: f64| Duration::try_from_secs_f64(v / units).ok();

    let module = "awkernel_async_lib::dag_sched::fluid_table";
    let result_type =
        format!("Result<(&'static {module}::FluidTable, &'static [usize]), &'static str>");
    let files: Vec<&str> = contents.iter().map(String::as_str).collect();
    let generated = match parse_yaml::parse_dags(&files) {
        Err(e) => format!(
            "pub(crate) static FLUID_TABLE_RESULT: {result_type} = Err({:?});\n",
            format!("cannot parse the DAG files: {e}")
        ),
        Ok(dags) => {
            let (admitted, offline) = fluid_offline::admit_offline(&dags, pool_cpus, unit);
            match fluid_table::build(&offline, pool_cpus, per_node) {
                Ok(table) => {
                    let entries: usize = table
                        .spans
                        .iter()
                        .flat_map(|span| span.cpus.iter())
                        .map(Vec::len)
                        .sum();
                    println!(
                        "cargo:warning=DAG-Fluid offline table: DAG files {admitted:?} on {pool_cpus} CPUs, hyperperiod {} ns, {} spans, {entries} entries{}",
                        table.hyperperiod_ns,
                        table.spans.len(),
                        if per_node { " (per node)" } else { "" }
                    );
                    let mut code = table.to_rust(module, "FLUID_TABLE");
                    code.push_str(&format!(
                        "pub(crate) static FLUID_TABLE_RESULT: {result_type} = Ok((&FLUID_TABLE, &{admitted:?}));\n"
                    ));
                    code
                }
                Err(e) => {
                    let reason = format!(
                        "DAG files {admitted:?} on {pool_cpus} CPUs rejected by the offline table: {e:?}"
                    );
                    println!("cargo:warning=DAG-Fluid: {reason}");
                    format!(
                        "pub(crate) static FLUID_TABLE_RESULT: {result_type} = Err({reason:?});\n"
                    )
                }
            }
        }
    };
    fs::write(out_dir.join("fluid_table.rs"), generated)
        .expect("failed to write generated fluid_table.rs");
}

/// Folder to load every `dag_<N>.yaml` file from (searched recursively).
/// Overridable via the `RD_GEN_DAGS_DIR` environment variable at build time
/// (e.g. `RD_GEN_DAGS_DIR=/path/to/generated/DAGs cargo build ...`) so an
/// automation pipeline can point the kernel at a freshly-generated RD-Gen
/// output directory without editing this file; falls back to this fixed
/// path — the staging directory `scripts/real_machine_trial.py` fills with
/// the DAGs of one offline-tested trial (a `trials.jsonl` record) — when
/// unset, so a plain `make x86_64` boots the last staged task set.
const DEFAULT_DAGS_DIR: &str = "/home/nokosan/azumi-lab/RD-Gen/test/awkernel_staged";

/// True for `dag_<N>.yaml`/`.yml` files. RD-Gen writes sibling files in the
/// same directory that are not DAG definitions (e.g. `combination_log.yaml`,
/// `dag_<N>.png`/`.eps` renders); `parse_yaml::parse_dags` fails the entire
/// batch on the first file that isn't a single-document DAG YAML, so those
/// must be filtered out here rather than left for it to reject at runtime.
fn is_dag_yaml(path: &Path) -> bool {
    let is_yaml = path
        .extension()
        .is_some_and(|ext| ext == "yaml" || ext == "yml");
    let is_dag_named = path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with("dag_"));
    is_yaml && is_dag_named
}

/// The `<N>` in `dag_<N>.yaml`, used to sort files in the natural order
/// RD-Gen generates them in (0, 1, 2, ..., 10, ...) rather than the
/// lexicographic order their filenames sort in (0, 1, 10, 11, ..., 2, ...).
/// Falls back to `usize::MAX` (sorted last) for a name that doesn't fit the
/// pattern, rather than failing the build over a cosmetic ordering detail.
fn dag_index(path: &Path) -> usize {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .and_then(|stem| stem.strip_prefix("dag_"))
        .and_then(|n| n.parse().ok())
        .unwrap_or(usize::MAX)
}

fn collect_dag_yaml_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_dag_yaml_files(&path, out);
        } else if is_dag_yaml(&path) {
            out.push(path);
        }
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=RD_GEN_DAGS_DIR");
    let dags_dir = env::var("RD_GEN_DAGS_DIR").unwrap_or_else(|_| DEFAULT_DAGS_DIR.to_string());
    println!("cargo:rerun-if-changed={dags_dir}");

    let mut files = Vec::new();
    collect_dag_yaml_files(Path::new(&dags_dir), &mut files);
    files.sort_by_key(|path| dag_index(path));

    assert!(
        !files.is_empty(),
        "no dag_<N>.yaml files found under DAGS_DIR ({dags_dir})"
    );

    let mut generated = String::from(
        "// Auto-generated by build.rs from DAGS_DIR -- do not edit by hand.\n\
         pub(crate) static DAG_FILES: &[&str] = &[\n",
    );
    let mut contents: Vec<String> = Vec::with_capacity(files.len());
    for file in &files {
        // A trailing newline is required by the yaml_peg specification, as
        // in `DAG_FILES` below.
        let text = fs::read_to_string(file).expect("failed to read a DAG file");
        contents.push(format!("{text}\n"));
        println!("cargo:rerun-if-changed={}", file.display());
        // A trailing newline is required by the yaml_peg specification.
        generated.push_str(&format!("    concat!(include_str!({file:?}), \"\\n\"),\n"));
    }
    generated.push_str("];\n");

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set by cargo");
    fs::write(Path::new(&out_dir).join("dag_files.rs"), generated)
        .expect("failed to write generated dag_files.rs");

    let has = |feature: &str| env::var_os(format!("CARGO_FEATURE_{feature}")).is_some();
    if has("DAGFLUID") && !has("DAGFLUID_EVENT") {
        generate_fluid_table(&contents, Path::new(&out_dir));
    }
}
