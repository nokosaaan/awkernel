#![no_std]
#![allow(static_mut_refs)]
extern crate alloc;

mod bridge;
mod nodes;
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
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use core::time::Duration;
use csv_core::{ReadRecordResult, Reader};

pub use common_types::Header;
use imu_driver::ImuCsvRow;
use vehicle_velocity_converter::VelocityCsvRow;

pub(crate) const LOG_ENABLE: bool = false;

const INTERFACE_ID: u64 = 0;
const INTERFACE_ADDR: Ipv4Addr = Ipv4Addr::new(10, 0, 2, 64);
const UDP_TCP_DST_ADDR: Ipv4Addr = Ipv4Addr::new(10, 0, 2, 2);
const UDP_DST_PORT: u16 = 26099;

static mut LATEST_JSON_DATA: Option<String> = None;
static JSON_DATA_READY: AtomicBool = AtomicBool::new(false);
static JSON_DATA_LENGTH: AtomicUsize = AtomicUsize::new(0);

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

pub(crate) fn save_json_data_to_global(json_data: String) {
    unsafe {
        LATEST_JSON_DATA = Some(json_data.clone());
    }
    JSON_DATA_READY.store(true, Ordering::Relaxed);
    JSON_DATA_LENGTH.store(json_data.len(), Ordering::Relaxed);
}
