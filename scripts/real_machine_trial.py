#!/usr/bin/env python3
"""Automates one or more real-machine trial cycles:

  wait for target (Windows) reachable over SSH
  -> pick next unused DAG from an RD-Gen pool and stage it
  -> build awkernel with that DAG embedded (RD_GEN_DAGS_DIR)
  -> start minicom logging to log/trace_<n>.log
  -> bcdedit bootnext=PXE + shutdown /r /t 0 on the target
  -> wait for the trace to finish (marker in the log, or a timeout)
  -> stop minicom, record the trial in log/dag_selection.json

Run repeatedly with --trials N to fire off N trials unattended.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_AWKERNEL_DIR = Path("/home/nokosan/azumi-lab/awkernel")
DEFAULT_POOL_DIR = Path("/home/nokosan/azumi-lab/RD-Gen/test/awkernel_theory_pool_branching/DAGs")
DEFAULT_STAGING_DIR = Path("/home/nokosan/azumi-lab/RD-Gen/test/awkernel_staged")

# bcdedit's own field labels ("identifier"/"description") are localized to the
# target's Windows display language, so we don't match on them -- only on the
# GUID shape and on hint words that tend to survive localization (device/
# protocol names such as "Network"/"IPv4" are usually not translated).
BOOT_ENTRY_DESC_HINTS = ("efi network", "ipv4", "pxe", "network boot")
GUID_RE = re.compile(r"\{[0-9a-fA-F-]{8,}\}")


def decode_console_output(raw: bytes) -> str:
    """Windows console output over SSH is UTF-8 on English systems but
    CP932 (Shift-JIS) on Japanese ones; try both before giving up lossily."""
    for encoding in ("utf-8", "cp932"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path, default):
    if not path.exists():
        return default
    with path.open() as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


class SshResult:
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def ssh_run(host, user, remote_cmd, timeout=15, check=True):
    cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={timeout}",
        f"{user}@{host}",
        remote_cmd,
    ]
    raw = subprocess.run(cmd, capture_output=True, timeout=timeout + 5, check=False)
    result = SshResult(raw.returncode, decode_console_output(raw.stdout), decode_console_output(raw.stderr))
    if check and raw.returncode != 0:
        raise subprocess.CalledProcessError(raw.returncode, cmd, result.stdout, result.stderr)
    return result


def wait_for_ssh(host, user, timeout_secs, poll_interval=5):
    deadline = time.monotonic() + timeout_secs
    while time.monotonic() < deadline:
        try:
            ssh_run(host, user, "echo ok", timeout=5)
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            time.sleep(poll_interval)
    raise RuntimeError(f"{user}@{host} did not become SSH-reachable within {timeout_secs}s")


def discover_boot_guid(host, user):
    result = ssh_run(host, user, "bcdedit /enum firmware", timeout=15)
    # Entries are separated by a blank line; the GUID and the hint words we
    # look for can be anywhere within an entry's block regardless of which
    # (localized) field label they sit under.
    blocks = re.split(r"\r?\n\s*\r?\n", result.stdout.strip())
    for block in blocks:
        guid_match = GUID_RE.search(block)
        if not guid_match:
            continue
        if any(hint in block.lower() for hint in BOOT_ENTRY_DESC_HINTS):
            guid = guid_match.group(0).strip("{}")
            summary = " / ".join(line.strip() for line in block.splitlines() if line.strip())
            return guid, summary
    raise RuntimeError(
        "no network-boot firmware entry found in `bcdedit /enum firmware` output "
        "(looked for a block containing a GUID and one of: " + ", ".join(BOOT_ENTRY_DESC_HINTS) + ")\n"
        "raw output:\n" + result.stdout
    )


def get_boot_guid(host, user, cache_path, force_rediscover):
    cache = load_json(cache_path, {})
    if not force_rediscover and "guid" in cache:
        return cache["guid"]
    guid, description = discover_boot_guid(host, user)
    save_json(cache_path, {"guid": guid, "description": description, "discovered_at": now_iso()})
    print(f"[boot-entry] discovered {guid} ({description}), cached at {cache_path}")
    return guid


def trigger_reboot_to_pxe(host, user, guid):
    ssh_run(host, user, f"bcdedit /set {{fwbootmgr}} bootsequence {{{guid}}}", timeout=15)
    # The target reboots as soon as this runs, so the ssh session itself may
    # be torn down mid-flight -- that is expected, not a failure.
    try:
        ssh_run(host, user, "shutdown /r /t 0", timeout=10)
    except subprocess.SubprocessError:
        pass


def select_next_dag(pool_dir, used_files):
    candidates = sorted(
        (p for p in pool_dir.glob("dag_*.yaml") if p.name not in used_files),
        key=lambda p: int(p.stem.removeprefix("dag_")) if p.stem.removeprefix("dag_").isdigit() else 1 << 62,
    )
    if not candidates:
        raise RuntimeError(f"no unused dag_<N>.yaml left under {pool_dir}")
    return candidates[0]


def stage_dag(src, staging_dir):
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    dst = staging_dir / src.name
    shutil.copyfile(src, dst)
    return dst


def run_build(awkernel_dir, staging_dir, dry_run):
    print(f"[build] RD_GEN_DAGS_DIR={staging_dir} make x86_64 RELEASE=1")
    if dry_run:
        return
    subprocess.run(
        ["make", "x86_64", "RELEASE=1"],
        cwd=awkernel_dir,
        env={**__import__("os").environ, "RD_GEN_DAGS_DIR": str(staging_dir)},
        check=True,
    )


def start_minicom(serial_device, baud, log_path, use_sudo, dry_run):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()
    cmd = (["sudo"] if use_sudo else []) + [
        "minicom", "-D", serial_device, "-b", str(baud), "-C", str(log_path),
    ]
    print(f"[minicom] {' '.join(cmd)}")
    if dry_run:
        return None
    return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_minicom(proc):
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def wait_for_marker_or_timeout(log_path, marker, max_wait_secs, poll_interval=2):
    deadline = time.monotonic() + max_wait_secs
    while time.monotonic() < deadline:
        if log_path.exists() and marker in log_path.read_text(errors="ignore"):
            return True
        time.sleep(poll_interval)
    return False


def next_trial_id(state):
    return max((r["trial"] for r in state), default=0) + 1


def run_trial(args, state, boot_cache_path):
    trial_id = next_trial_id(state)
    print(f"=== trial {trial_id} ===")

    if not args.dry_run:
        wait_for_ssh(args.host, args.user, args.ssh_wait_secs)

    dag_src = select_next_dag(args.pool_dir, {r["dag_file"] for r in state})
    dag_staged = stage_dag(dag_src, args.staging_dir)
    print(f"[dag] selected {dag_src.name}")

    run_build(args.awkernel_dir, args.staging_dir, args.dry_run)

    log_path = args.log_dir / f"{args.log_prefix}{trial_id}.log"
    minicom_proc = start_minicom(args.serial_device, args.baud, log_path, args.sudo_minicom, args.dry_run)
    time.sleep(2)  # let minicom attach to the port before we trigger the reboot

    if args.dry_run:
        guid = "DRY-RUN"
        marker_found = False
    else:
        guid = get_boot_guid(args.host, args.user, boot_cache_path, args.rediscover_boot_entry)
        trigger_reboot_to_pxe(args.host, args.user, guid)
        marker_found = wait_for_marker_or_timeout(log_path, args.marker, args.max_wait_secs)
        stop_minicom(minicom_proc)

    record = {
        "trial": trial_id,
        "dag_file": dag_src.name,
        "dag_pool_path": str(dag_src),
        "log_file": str(log_path),
        "boot_entry_guid": guid,
        "marker_found": marker_found,
        "timestamp": now_iso(),
    }
    state.append(record)
    save_json(args.log_dir / "dag_selection.json", state)
    if args.dry_run:
        status = "dry-run, no reboot/wait performed"
    elif marker_found:
        status = "marker found"
    else:
        status = f"timed out after {args.max_wait_secs}s"
    print(f"[trial {trial_id}] done ({status}) -> {log_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", type=int, default=1, help="number of trial cycles to run (default: 1)")
    p.add_argument("--awkernel-dir", type=Path, default=DEFAULT_AWKERNEL_DIR)
    p.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR, help="RD-Gen DAG pool to draw from")
    p.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING_DIR,
                   help="scratch dir the selected DAG is copied into; pointed at via RD_GEN_DAGS_DIR")
    p.add_argument("--host", default="192.168.10.10")
    p.add_argument("--user", default="awkernel")
    p.add_argument("--serial-device", default="/dev/ttyUSB0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--sudo-minicom", action="store_true", help="run minicom under sudo (default: off; relies on dialout group membership)")
    p.add_argument("--log-dir", type=Path, default=None, help="default: <awkernel-dir>/log")
    p.add_argument("--log-prefix", default="trace_")
    p.add_argument("--marker", default="TRACE_END", help="string that marks trace completion in the growing log")
    p.add_argument("--max-wait-secs", type=int, default=90, help="hard cap per trial while waiting for the marker")
    p.add_argument("--ssh-wait-secs", type=int, default=180, help="how long to wait for the target to come back up before a trial")
    p.add_argument("--rediscover-boot-entry", action="store_true", help="force re-querying bcdedit instead of using the cached GUID")
    p.add_argument("--dry-run", action="store_true", help="print planned actions without touching the network/build/serial port")
    args = p.parse_args()
    if args.log_dir is None:
        args.log_dir = args.awkernel_dir / "log"
    return args


def main():
    args = parse_args()
    state_path = args.log_dir / "dag_selection.json"
    boot_cache_path = args.log_dir / "real_machine_boot_entry.json"
    state = load_json(state_path, [])

    for _ in range(args.trials):
        try:
            run_trial(args, state, boot_cache_path)
        except Exception as e:  # noqa: BLE001 -- surface any failure and stop the loop rather than burn through trials blind
            print(f"[abort] {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
