use std::{env, fs, path::PathBuf};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is not set"));
    let csv_data_path = out_dir.join("csv_data.rs");

    let replay = env::var_os("CARGO_FEATURE_REPLAY").is_some();
    let live = env::var_os("CARGO_FEATURE_LIVE").is_some();
    if replay && live {
        panic!("features `replay` (open loop) and `live` (closed loop) are exclusive");
    }
    // `awsim_input`: the inputs come from AWSIM (recorded or live) instead of the synthetic
    // sensors, so the AWSIM sensor kit's frames apply and the MRM trigger is external.
    println!("cargo::rustc-check-cfg=cfg(awsim_input)");
    if replay || live {
        println!("cargo:rustc-cfg=awsim_input");
    }

    // Every embedded CSV, as (environment variable, generated constant, needed by).
    // The constants are always generated (empty when the variable is unset).
    const CSV_INPUTS: [(&str, &str, Need); 6] = [
        ("IMU_CSV_PATH", "IMU_CSV_DATA_STR", Need::Replay),
        ("VELOCITY_CSV_PATH", "VELOCITY_CSV_DATA_STR", Need::Replay),
        ("POSE_CSV_PATH", "POSE_CSV_DATA_STR", Need::Replay),
        (
            "TRAJECTORY_CSV_PATH",
            "TRAJECTORY_CSV_DATA_STR",
            Need::ReplayAndLive,
        ),
        (
            "MRM_TRIGGER_CSV_PATH",
            "MRM_TRIGGER_CSV_DATA_STR",
            Need::Replay,
        ),
        // Gate events of the recorded run (source selection / ADAPI pause); optional: without
        // them the replayed gate is engaged from the start.
        ("GATE_CSV_PATH", "GATE_CSV_DATA_STR", Need::Optional),
    ];

    println!("cargo:rerun-if-changed=build.rs");
    let mut generated = String::new();
    for (var, constant, need) in CSV_INPUTS {
        // Rebuild when the variable or the file it points to changes.
        println!("cargo:rerun-if-env-changed={var}");
        if let Ok(path) = env::var(var) {
            println!("cargo:rerun-if-changed={path}");
        }
        let contents = read_csv_from_env(var);
        // A build without its data runs on all-zero inputs; flag it at build time.
        let needed = match need {
            Need::Replay => replay,
            Need::ReplayAndLive => replay || live,
            Need::Optional => false,
        };
        if needed && contents.is_empty() {
            let mode = if replay { "replay" } else { "live" };
            println!(
                "cargo:warning={mode} build: {var} is unset or empty (source replay_env.sh first)"
            );
        }
        generated.push_str(&format!("pub const {constant}: &str = {contents:?};\n"));
    }

    fs::write(&csv_data_path, generated).expect("failed to write generated csv constants");
}

#[derive(Clone, Copy)]
enum Need {
    Replay,
    ReplayAndLive,
    Optional,
}

fn read_csv_from_env(var_name: &str) -> String {
    let Ok(path) = env::var(var_name) else {
        return String::new();
    };

    match fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(err) => {
            println!("cargo:warning=failed to read {var_name} from {path}: {err}");
            String::new()
        }
    }
}
