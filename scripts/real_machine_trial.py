#!/usr/bin/env python3
"""Automates one or more real-machine trial cycles:

  wait for target reachable over SSH
  -> pick DAG(s) to stage and stage them (see "Two selection modes" below)
  -> build awkernel with them embedded (RD_GEN_DAGS_DIR)
  -> start minicom logging to log/trace_<n>.log
  -> set the target's next boot to PXE (bcdedit or efibootmgr, see
     --target-os) and reboot it
  -> wait for the trace to finish (marker in the log, or a timeout)
  -> stop minicom, record the trial in log/dag_selection.json

Run repeatedly with --trials N to fire off N trials unattended.

Two selection modes (mutually exclusive):
  - Default (single-DAG): picks the next not-yet-used dag_<N>.yaml from
    --pool-dir (ascending N), one DAG per trial.
  - --trials-jsonl PATH (group-of-8): reads the JSON-Lines file
    rd_gen_to_dags's acceptance_ratio.rs (via --trials-jsonl on
    run_schedulability_evaluation.py) writes one record per resampled
    trial to, and consumes it top to bottom -- trial 1 uses the file's own
    line 1's 8 drawn dag_<N>.yaml names, trial 2 uses line 2, etc. -- so
    the exact task set an offline trial predicted as (in)admissible is the
    one actually staged and booted for real, not a fresh independent draw.
"""

import argparse
import json
import os
import pty
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_AWKERNEL_DIR = Path("/home/nokosan/ws/awkernel")
DEFAULT_POOL_DIR = Path("/home/nokosan/ws/RD-Gen/test/awkernel_theory_pool_branching/DAGs")
DEFAULT_STAGING_DIR = Path("/home/nokosan/ws/RD-Gen/test/awkernel_staged")

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


def wake_on_lan(mac, dry_run):
    print(f"[wol] wakeonlan {mac}")
    if dry_run:
        return
    # A missed/ignored magic packet isn't fatal here -- wait_for_ssh right
    # after this will just keep polling until ssh_wait_secs elapses, so a
    # transient failure to send surfaces as that timeout's own error rather
    # than needing its own handling.
    subprocess.run(["wakeonlan", mac], check=False, capture_output=True)


def wait_for_ssh(host, user, timeout_secs, poll_interval=5):
    deadline = time.monotonic() + timeout_secs
    warned_auth_failure = False
    while time.monotonic() < deadline:
        try:
            ssh_run(host, user, "echo ok", timeout=5)
            return
        except subprocess.CalledProcessError as e:
            # "Permission denied" (no key registered / password auth needed)
            # won't ever resolve itself by retrying, unlike "connection
            # refused"/timeout while the target is mid-reboot -- silently
            # retrying it for the full timeout looks indistinguishable from
            # a genuine hang. Warn once, loudly, but keep polling anyway (in
            # case someone fixes it -- e.g. `ssh-copy-id` -- in another
            # terminal while this waits).
            if not warned_auth_failure and "permission denied" in (e.stderr or "").lower():
                warned_auth_failure = True
                print(
                    f"[ssh] WARNING: {user}@{host} rejected authentication "
                    f"(not just 'still booting') -- {e.stderr.strip()!r}. "
                    "Will keep polling, but this needs fixing on its own "
                    "(e.g. `ssh-copy-id`), not more waiting.",
                    file=sys.stderr,
                )
            time.sleep(poll_interval)
        except subprocess.TimeoutExpired:
            time.sleep(poll_interval)
    raise RuntimeError(f"{user}@{host} did not become SSH-reachable within {timeout_secs}s")


def discover_boot_entry_windows(host, user):
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


def discover_boot_entry_linux(host, user, target_mac):
    # `efibootmgr -v` writes the same MAC in at least two different
    # delimiter styles on the same line -- "EFI PXE 0 for IPv4
    # (80-FA-5B-79-11-E1)" (hyphens) alongside "MAC(80fa5b7911e1,0)" (no
    # delimiter at all) -- so comparing after stripping all ':'/'-' from
    # both sides is what actually matches reliably, not a straight
    # colon-normalized substring check.
    result = ssh_run(host, user, "sudo efibootmgr -v", timeout=15)
    mac_needle = re.sub(r"[:-]", "", target_mac.lower())
    matches = []
    for line in result.stdout.splitlines():
        if mac_needle in re.sub(r"[:-]", "", line.lower()):
            m = re.match(r"Boot([0-9A-Fa-f]{4})", line.strip())
            if m:
                matches.append((m.group(1), line.strip()))
    if not matches:
        raise RuntimeError(
            f"no `efibootmgr -v` entry found whose MAC matches {target_mac}\n"
            "raw output:\n" + result.stdout
        )
    # A NIC with both IPv4 and IPv6 firmware boot entries (and sometimes a
    # stale duplicate of one) shares the same MAC across all of them --
    # prefer an explicit IPv4 entry so we don't end up on IPv6 or a
    # coincidental first match.
    ipv4_matches = [m for m in matches if "ipv4" in m[1].lower()]
    return (ipv4_matches or matches)[0]


def get_boot_entry(host, user, cache_path, force_rediscover, target_os, target_mac):
    cache = load_json(cache_path, {})
    if not force_rediscover and cache.get("target_os") == target_os and "value" in cache:
        return cache["value"]
    if target_os == "windows":
        value, description = discover_boot_entry_windows(host, user)
    else:
        value, description = discover_boot_entry_linux(host, user, target_mac)
    save_json(cache_path, {
        "target_os": target_os, "value": value, "description": description, "discovered_at": now_iso(),
    })
    print(f"[boot-entry] discovered {value} ({description}), cached at {cache_path}")
    return value


def trigger_reboot_to_pxe(host, user, target_os, value):
    if target_os == "windows":
        ssh_run(host, user, f"bcdedit /set {{fwbootmgr}} bootsequence {{{value}}}", timeout=15)
        reboot_cmd = "shutdown /r /t 0"
    else:
        ssh_run(host, user, f"sudo efibootmgr --bootnext {value}", timeout=15)
        reboot_cmd = "sudo reboot"
    # The target reboots as soon as this runs, so the ssh session itself may
    # be torn down mid-flight -- that is expected, not a failure.
    try:
        ssh_run(host, user, reboot_cmd, timeout=10)
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


def load_trials_jsonl(path):
    """Every record, in file order -- acceptance_ratio.rs appends one line
    per resampled trial (see its own doc comment for the schema), so line
    order is trial order across however many u_norm levels it swept."""
    trials = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                trials.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{path}:{line_no}: not valid JSON: {e}") from e
    if not trials:
        raise RuntimeError(f"{path} has no trial records")
    return trials


def select_next_group(trials, consumed_count):
    """The next not-yet-consumed record, top to bottom -- consumed_count is
    how many of `trials`' leading records earlier invocations already used
    (tracked via each state record's own "trials_jsonl_line", not just
    len(state), since state may also hold single-DAG-mode records)."""
    if consumed_count >= len(trials):
        raise RuntimeError(
            f"no more trial records left: {len(trials)} available, {consumed_count} already used"
        )
    return consumed_count, trials[consumed_count]


def stage_group(pool_dir, dag_names, staging_dir):
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    staged = []
    for name in dag_names:
        src = pool_dir / name
        if not src.exists():
            raise RuntimeError(
                f"'{name}' (from trials.jsonl) not found under {pool_dir} -- "
                "is --pool-dir the same RD-Gen pool the JSONL was generated from?"
            )
        dst = staging_dir / name
        shutil.copyfile(src, dst)
        staged.append(dst)
    return staged


def run_build(awkernel_dir, staging_dir, dry_run):
    print(f"[build] RD_GEN_DAGS_DIR={staging_dir} make x86_64 RELEASE=1")
    if dry_run:
        return
    subprocess.run(
        ["make", "x86_64", "RELEASE=1"],
        cwd=awkernel_dir,
        env={**os.environ, "RD_GEN_DAGS_DIR": str(staging_dir)},
        check=True,
    )


class MinicomHandle:
    def __init__(self, proc, master_fd, drain_thread):
        self.proc = proc
        self.master_fd = master_fd
        self.drain_thread = drain_thread


def _drain_pty(master_fd, sink_path):
    """Continuously reads the pty master side and discards it (to a file,
    for post-mortem debugging) for as long as minicom is alive. minicom is a
    full-screen terminal app: it redraws its own status line/screen
    periodically even while otherwise idle, and an unread pty has a small
    kernel buffer (same order as a pipe's) -- without something draining it
    for the whole trial, minicom would eventually block trying to write to
    it, same failure mode a pipe would have here."""
    with sink_path.open("wb") as f:
        while True:
            try:
                data = os.read(master_fd, 4096)
            except OSError:
                return
            if not data:
                return
            f.write(data)
            f.flush()


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
    # minicom is a full-screen terminal app (ncurses), not a plain filter:
    # given plain pipes/DEVNULL for its stdio it can't find a controlling
    # terminal / valid TERM, and exits almost immediately (successfully,
    # exit code 0, after printing its own startup banner) *without ever
    # reaching the loop that captures serial data to -C* -- this looks
    # nothing like the permission-error case (which fails loudly with a
    # nonzero exit and an "許可がありません" message) but is just as
    # silent-and-empty a failure for -C's log file. A real pty plus TERM
    # set is what makes it actually run as a background capture process.
    master_fd, slave_fd = pty.openpty()
    env = {**os.environ, "TERM": "xterm"}
    proc = subprocess.Popen(
        cmd, stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
        env=env, start_new_session=True,
    )
    os.close(slave_fd)  # only the child needs its own dup of this now

    stderr_path = log_path.with_suffix(log_path.suffix + ".minicom_stderr")
    drain_thread = threading.Thread(target=_drain_pty, args=(master_fd, stderr_path), daemon=True)
    drain_thread.start()

    # minicom exits near-instantly on a permission/device/tty error rather
    # than hanging -- catching that here means a whole trial doesn't
    # silently burn its build+reboot+max-wait-secs budget capturing
    # nothing. A real minicom session keeps running past this, so a short
    # grace period is enough to tell the two apart without slowing every
    # trial down noticeably.
    time.sleep(0.5)
    if proc.poll() is not None:
        drain_thread.join(timeout=2)
        output = stderr_path.read_text(errors="replace").strip() if stderr_path.exists() else ""
        os.close(master_fd)
        raise RuntimeError(
            f"minicom exited immediately (code {proc.returncode}): {output!r} -- "
            "if this is a permission error, either `sudo usermod -aG dialout $USER` "
            "and start a new login session, or pass --sudo-minicom; otherwise see "
            "start_minicom's own doc comment (pty/TERM requirement)"
        )
    return MinicomHandle(proc, master_fd, drain_thread)


def stop_minicom(handle):
    if handle is None:
        return
    proc = handle.proc
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    try:
        os.close(handle.master_fd)
    except OSError:
        pass
    handle.drain_thread.join(timeout=2)


def wait_for_marker_or_timeout(log_path, marker, max_wait_secs, poll_interval=2):
    deadline = time.monotonic() + max_wait_secs
    while time.monotonic() < deadline:
        if log_path.exists() and marker in log_path.read_text(errors="ignore"):
            return True
        time.sleep(poll_interval)
    return False


def next_trial_id(state):
    return max((r["trial"] for r in state), default=0) + 1


def run_trial(args, state, boot_cache_path, trials_jsonl):
    trial_id = next_trial_id(state)
    print(f"=== trial {trial_id} ===")

    if not args.dry_run:
        # The previous trial's own auto-reboot (see kernel/src/config.rs's
        # AUTO_REBOOT_*) already brings the target back up on its own (its
        # one-shot PXE bootsequence is consumed, so it falls through to the
        # normal boot order) -- this WoL is only a safety net for the case
        # where the target was left fully powered off some other way, and a
        # no-op (silently ignored) if it's already up, so keeping it here
        # costs nothing even though AUTO_REBOOT no longer requires it.
        wake_on_lan(args.target_mac, args.dry_run)
        wait_for_ssh(args.host, args.user, args.ssh_wait_secs)

    if trials_jsonl is not None:
        consumed = sum(1 for r in state if "trials_jsonl_line" in r)
        line_idx, trial_record = select_next_group(trials_jsonl, consumed)
        dag_names = trial_record["dags"]
        stage_group(args.pool_dir, dag_names, args.staging_dir)
        print(f"[dag] staged group from trials.jsonl line {line_idx + 1}: {', '.join(dag_names)}")
        selection_info = {
            "trials_jsonl_line": line_idx,
            "dag_files": dag_names,
            "u_norm": trial_record.get("u_norm"),
            "offline_trial": trial_record.get("trial"),
            "offline_federated_accepted": trial_record.get("federated_accepted"),
            "offline_vfed_accepted": trial_record.get("vfed_accepted"),
        }
    else:
        dag_src = select_next_dag(args.pool_dir, {r["dag_file"] for r in state if "dag_file" in r})
        stage_dag(dag_src, args.staging_dir)
        print(f"[dag] selected {dag_src.name}")
        selection_info = {"dag_file": dag_src.name, "dag_pool_path": str(dag_src)}

    run_build(args.awkernel_dir, args.staging_dir, args.dry_run)

    log_path = args.log_dir / f"{args.log_prefix}{trial_id}.log"
    minicom_handle = start_minicom(args.serial_device, args.baud, log_path, args.sudo_minicom, args.dry_run)
    time.sleep(2)  # let minicom attach to the port before we trigger the reboot

    if args.dry_run:
        boot_entry = "DRY-RUN"
        marker_found = False
    else:
        boot_entry = get_boot_entry(
            args.host, args.user, boot_cache_path, args.rediscover_boot_entry, args.target_os, args.target_mac,
        )
        trigger_reboot_to_pxe(args.host, args.user, args.target_os, boot_entry)
        marker_found = wait_for_marker_or_timeout(log_path, args.marker, args.max_wait_secs)
        stop_minicom(minicom_handle)

    record = {
        "trial": trial_id,
        **selection_info,
        "log_file": str(log_path),
        "boot_entry": boot_entry,
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
                   help="scratch dir the selected DAG(s) are copied into; pointed at via RD_GEN_DAGS_DIR")
    p.add_argument("--trials-jsonl", type=Path, default=None,
                   help="group-of-8 mode: consume acceptance_ratio.rs's per-trial JSONL top to "
                        "bottom instead of picking single DAGs from --pool-dir (see module docstring)")
    p.add_argument("--host", default="192.168.10.10")
    p.add_argument("--user", default="azumiken-admin")
    p.add_argument("--target-os", choices=["windows", "linux"], default="linux",
                    help="which one-shot-PXE-boot mechanism to drive over ssh: bcdedit+shutdown "
                         "(windows) or efibootmgr+reboot (linux)")
    p.add_argument("--target-mac", default="80:fa:5b:79:11:e1",
                    help="target's PXE NIC MAC: identifies the right `efibootmgr -v` entry on "
                         "--target-os linux, and is also a `wakeonlan` safety net sent before "
                         "each trial in case the target is ever left fully powered off (a "
                         "no-op when it's already up, the normal case with AUTO_REBOOT)")
    p.add_argument("--trial-interval-secs", type=int, default=30,
                    help="pause between trials, on top of --ssh-wait-secs, so the target's "
                         "own AUTO_REBOOT_SECS has time to actually fire before the next "
                         "bootsequence is set -- too short a gap can race this trial's "
                         "reboot against the previous one's")
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
    trials_jsonl = load_trials_jsonl(args.trials_jsonl) if args.trials_jsonl is not None else None

    for i in range(args.trials):
        try:
            run_trial(args, state, boot_cache_path, trials_jsonl)
        except Exception as e:  # noqa: BLE001 -- surface any failure and stop the loop rather than burn through trials blind
            # str(CalledProcessError) is just "Command '[...]' returned
            # non-zero exit status N" -- the actual reason (e.g. sudo's own
            # stderr) is a separate attribute and gets silently dropped
            # unless printed explicitly, which then requires reproducing
            # the failing command by hand just to see it.
            detail = getattr(e, "stderr", None)
            print(f"[abort] {e}" + (f"\n{detail.strip()}" if detail else ""), file=sys.stderr)
            sys.exit(1)
        if not args.dry_run and i < args.trials - 1 and args.trial_interval_secs > 0:
            print(f"[wait] pausing {args.trial_interval_secs}s before the next trial")
            time.sleep(args.trial_interval_secs)


if __name__ == "__main__":
    main()
