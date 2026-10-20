#![no_std]
#![allow(static_mut_refs)]
extern crate alloc;

mod bridge;
mod nodes;
#[cfg(feature = "replay")]
mod replay;
mod shared;
#[cfg(not(feature = "dag_multi"))]
mod wiring_single;
#[cfg(feature = "dag_multi")]
mod wiring_multi;

use alloc::{string::String, vec, vec::Vec};
use awkernel_async_lib::dag::finish_create_dags;
use awkernel_async_lib::net::IpAddr;
use awkernel_lib::delay::wait_microsec;
use awkernel_lib::sync::mutex::Mutex;
use core::net::Ipv4Addr;
use core::sync::atomic::{AtomicU32, Ordering};
use shared::{Latest, Queue};
use core::time::Duration;
use csv_core::{ReadRecordResult, Reader};

pub use common_types::Header;
use imu_driver::ImuCsvRow;
use vehicle_velocity_converter::VelocityCsvRow;

pub(crate) const LOG_ENABLE: bool = false;

const INTERFACE_ID: u64 = 0;
const INTERFACE_ADDR: Ipv4Addr = Ipv4Addr::new(10, 0, 2, 64);
const UDP_TCP_DST_ADDR: Ipv4Addr = Ipv4Addr::new(10, 0, 2, 2);
/// EKF odometry JSON (read by `awkernel_script/ekf_view/run_ekf_monitoring.py`).
const UDP_DST_PORT: u16 = 26099;
/// `/control/command/*` JSON, one packet per sink period (read by `awsim_bridge`).
const UDP_CMD_DST_PORT: u16 = 26100;

/// Newest odometry JSON with a sequence number, so the sender only sends new ones.
static ODOMETRY_JSON: Latest<(u32, String)> = Latest::new();
static ODOMETRY_SEQ: AtomicU32 = AtomicU32::new(0);
/// Command packets waiting for the sender task (oldest dropped when full, counted).
static COMMAND_JSON: Queue<String, 16> = Queue::new();

include!(concat!(env!("OUT_DIR"), "/csv_data.rs"));

pub(crate) static mut IMU_CSV_DATA: Option<Vec<ImuCsvRow>> = None;
pub(crate) static mut VELOCITY_CSV_DATA: Option<Vec<VelocityCsvRow>> = None;
pub(crate) static IMU_CSV_COUNT: Mutex<usize> = Mutex::new(0);
pub(crate) static VELOCITY_CSV_COUNT: Mutex<usize> = Mutex::new(0);

/// DAG wiring, selected at build time:
/// - default: one DAG, every node at the 33 ms source period (`wiring_single`)
/// - feature `dag_multi`: one DAG per period of the node diagram (`wiring_multi`)
#[cfg(not(feature = "dag_multi"))]
use wiring_single::build as build_dags;
#[cfg(feature = "dag_multi")]
use wiring_multi::build as build_dags;

pub async fn run() {
    wait_microsec(1000000);

    if let Err(e) = initialize_csv_data() {
        log::warn!("Failed to initialize CSV data: {}", e);
    }
    #[cfg(feature = "replay")]
    if let Err(e) = replay::load() {
        log::error!("{}", e);
    }

    // The network and the UDP sender come up before the DAGs, so the first vehicle command (and
    // the start of the replay timeline) already reaches the host instead of being dropped.
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

    nodes::configure_nodes();

    log::info!("Starting Autoware test application with simplified TCP networking");

    let dags = build_dags().await;
    log::info!("Autoware DAG wiring: {} DAG(s)", dags.len());

    nodes::mark_dag_start();
    let result = finish_create_dags(&dags).await;

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

pub(crate) fn parse_csv_records<F>(csv: &str, mut on_record: F) -> Result<(), &'static str>
where
    F: FnMut(&[&str]) -> Result<(), &'static str>,
{
    let mut reader = Reader::new();
    let mut input = csv.as_bytes();
    let mut output = vec![0u8; 4096];
    // Field end offsets of one record; the pose CSV is the widest (9 + 36 covariance columns).
    let mut ends = vec![0usize; 64];
    let mut header_skipped = false;

    loop {
        let (result, in_read, _out_written, num_fields) =
            reader.read_record(input, &mut output, &mut ends);
        input = &input[in_read..];

        if matches!(result, ReadRecordResult::OutputFull) {
            return Err("CSV output buffer is too small");
        }
        // Without this check the rest of the record would be read as a new, misaligned record.
        if matches!(result, ReadRecordResult::OutputEndsFull) {
            return Err("CSV record has too many fields");
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

pub(crate) fn parse_timestamp(sec: &str, nsec: &str) -> Result<u64, &'static str> {
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

pub(crate) fn parse_f64(field: &str) -> Result<f64, &'static str> {
    let trimmed = field.trim();
    if trimmed.is_empty() {
        return Ok(0.0);
    }
    trimmed.parse::<f64>().map_err(|_| "Failed to parse f64")
}

pub(crate) fn get_awkernel_uptime_timestamp() -> u64 {
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
    let mut socket = match awkernel_async_lib::net::udp::UdpSocket::bind_on_interface(
        INTERFACE_ID,
        &Default::default(),
    ) {
        Ok(socket) => socket,
        Err(e) => {
            log::error!("UDP sender: failed to create UDP socket: {:?}", e);
            return;
        }
    };

    let dst_addr = IpAddr::new_v4(UDP_TCP_DST_ADDR);
    let mut last_odometry_seq = None;
    let mut sent_commands: u64 = 0;

    // Each command is sent exactly once, as soon as the sink produced it; the odometry is sent
    // when a new one exists. No reply is awaited (the old 100 ms receive wait capped the send
    // rate at about 9 Hz).
    loop {
        for json in COMMAND_JSON.drain() {
            match socket.send(json.as_bytes(), &dst_addr, UDP_CMD_DST_PORT).await {
                Ok(_) => {
                    sent_commands += 1;
                    if sent_commands % 200 == 1 {
                        log::info!(
                            "UDP sender: {} command packets sent ({} dropped before send)",
                            sent_commands,
                            COMMAND_JSON.dropped()
                        );
                    }
                }
                Err(e) => log::warn!("UDP sender: command send failed: {:?}", e),
            }
        }

        if let Some((seq, json)) = ODOMETRY_JSON.get() {
            if last_odometry_seq != Some(seq) {
                last_odometry_seq = Some(seq);
                if let Err(e) = socket.send(json.as_bytes(), &dst_addr, UDP_DST_PORT).await {
                    log::warn!("UDP sender: odometry send failed: {:?}", e);
                }
            }
        }

        awkernel_async_lib::sleep(Duration::from_millis(5)).await;
    }
}

pub(crate) fn publish_odometry_json(json: String) {
    let seq = ODOMETRY_SEQ.fetch_add(1, Ordering::Relaxed);
    ODOMETRY_JSON.set((seq, json));
}

pub(crate) fn publish_command_json(json: String) {
    COMMAND_JSON.push(json);
}
