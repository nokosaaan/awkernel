use std::{env, fs, path::PathBuf};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is not set"));
    let csv_data_path = out_dir.join("csv_data.rs");

    // Every embedded CSV, as (environment variable, generated constant).
    // The replay set (POSE / TRAJECTORY / MRM_TRIGGER) is only read with the `replay` feature,
    // but the constants are always generated (empty when the variable is unset).
    const CSV_INPUTS: [(&str, &str); 5] = [
        ("IMU_CSV_PATH", "IMU_CSV_DATA_STR"),
        ("VELOCITY_CSV_PATH", "VELOCITY_CSV_DATA_STR"),
        ("POSE_CSV_PATH", "POSE_CSV_DATA_STR"),
        ("TRAJECTORY_CSV_PATH", "TRAJECTORY_CSV_DATA_STR"),
        ("MRM_TRIGGER_CSV_PATH", "MRM_TRIGGER_CSV_DATA_STR"),
    ];

    println!("cargo:rerun-if-changed=build.rs");
    let mut generated = String::new();
    for (var, constant) in CSV_INPUTS {
        // Rebuild when the variable or the file it points to changes.
        println!("cargo:rerun-if-env-changed={var}");
        if let Ok(path) = env::var(var) {
            println!("cargo:rerun-if-changed={path}");
        }
        let contents = read_csv_from_env(var);
        generated.push_str(&format!("pub const {constant}: &str = {contents:?};\n"));
    }

    fs::write(&csv_data_path, generated).expect("failed to write generated csv constants");
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
