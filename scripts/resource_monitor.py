#!/usr/bin/env python3
"""
resource_monitor.py

Standalone process that samples CPU, memory, and NVIDIA GPU stats every
N seconds and appends them to a CSV file. Designed to run alongside any
job on an HPC node -- it does not need to know anything about the job
itself, it just samples node-wide and (optionally) per-PID stats.

Usage:
    python resource_monitor.py --interval 5 --out run1_resources.csv
    python resource_monitor.py --interval 5 --out run1_resources.csv --pid 12345
    python resource_monitor.py --interval 5 --out run1_resources.csv &   # background

Stop it with Ctrl+C (or `kill` if backgrounded) -- it flushes and exits
cleanly on SIGINT/SIGTERM.

Requires: psutil, nvidia-ml-py  (pip install psutil nvidia-ml-py)
"""

import argparse
import csv
import signal
import sys
import time
from datetime import datetime

import psutil

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


# ----------------------------------------------------------------------
# GPU sampling
# ----------------------------------------------------------------------

def init_gpus():
    """Initialize NVML and return a list of handles, one per GPU.

    Returns an empty list if NVML is unavailable or no GPUs are found --
    callers should handle that case by simply omitting GPU columns.
    """
    if not NVML_AVAILABLE:
        return []
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as e:
        print(f"Warning: NVML init failed ({e}); continuing without GPU stats", file=sys.stderr)
        return []
    count = pynvml.nvmlDeviceGetCount()
    return [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]


def sample_gpu(handle):
    """Return a dict of stats for one GPU handle.

    Wrapped defensively: some fields (power, clocks) aren't supported on
    every GPU/driver combination, so individual failures fall back to
    None rather than crashing the whole sampling loop.
    """
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

    def safe(fn):
        try:
            return fn()
        except pynvml.NVMLError:
            return None

    temp = safe(lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
    power_mw = safe(lambda: pynvml.nvmlDeviceGetPowerUsage(handle))
    power_w = round(power_mw / 1000, 1) if power_mw is not None else None
    fan = safe(lambda: pynvml.nvmlDeviceGetFanSpeed(handle))
    sm_clock = safe(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))

    return {
        "gpu_util_pct": util.gpu,
        "gpu_mem_util_pct": util.memory,
        "gpu_mem_used_mb": round(mem.used / (1024 ** 2), 1),
        "gpu_mem_total_mb": round(mem.total / (1024 ** 2), 1),
        "gpu_temp_c": temp,
        "gpu_power_w": power_w,
        "gpu_fan_pct": fan,
        "gpu_sm_clock_mhz": sm_clock,
    }


# ----------------------------------------------------------------------
# CPU / memory sampling
# ----------------------------------------------------------------------

def sample_cpu_mem(proc):
    """Node-wide CPU/RAM stats, plus per-process stats if proc is set.

    proc is a psutil.Process or None. Node-wide CPU percent uses a
    non-blocking call (interval=None) relying on the time elapsed since
    the previous call -- accurate as long as sample_cpu_mem is called
    once per loop iteration, which it is.
    """
    vmem = psutil.virtual_memory()
    stats = {
        "cpu_pct_total": psutil.cpu_percent(interval=None),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "load_avg_1m": psutil.getloadavg()[0] if hasattr(psutil, "getloadavg") else None,
        "ram_used_gb": round(vmem.used / (1024 ** 3), 2),
        "ram_total_gb": round(vmem.total / (1024 ** 3), 2),
        "ram_pct": vmem.percent,
    }

    if proc is not None:
        try:
            with proc.oneshot():
                stats["proc_cpu_pct"] = proc.cpu_percent(interval=None)
                stats["proc_num_threads"] = proc.num_threads()
                stats["proc_rss_gb"] = round(proc.memory_info().rss / (1024 ** 3), 2)
        except psutil.NoSuchProcess:
            stats["proc_cpu_pct"] = None
            stats["proc_num_threads"] = None
            stats["proc_rss_gb"] = None

    return stats


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------

def build_fieldnames(gpu_handles, track_pid):
    fields = ["timestamp", "cpu_pct_total", "cpu_count_logical", "load_avg_1m",
              "ram_used_gb", "ram_total_gb", "ram_pct"]
    if track_pid:
        fields += ["proc_cpu_pct", "proc_num_threads", "proc_rss_gb"]
    for i in range(len(gpu_handles)):
        fields += [f"gpu{i}_util_pct", f"gpu{i}_mem_util_pct", f"gpu{i}_mem_used_mb",
                   f"gpu{i}_mem_total_mb", f"gpu{i}_temp_c", f"gpu{i}_power_w",
                   f"gpu{i}_fan_pct", f"gpu{i}_sm_clock_mhz"]
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=5.0,
                        help="seconds between samples (default: 5)")
    parser.add_argument("--out", default="resources.csv",
                        help="output CSV path (default: resources.csv)")
    parser.add_argument("--pid", type=int, default=None,
                        help="also track this specific PID (CPU%%, threads, RSS)")
    args = parser.parse_args()

    proc = None
    if args.pid is not None:
        try:
            proc = psutil.Process(args.pid)
        except psutil.NoSuchProcess:
            print(f"Warning: PID {args.pid} not found at startup; will report None for proc_* fields", file=sys.stderr)

    gpu_handles = init_gpus()
    if NVML_AVAILABLE and gpu_handles:
        print(f"Monitoring {len(gpu_handles)} GPU(s)")
    elif NVML_AVAILABLE:
        print("NVML available but no GPUs detected -- logging CPU/RAM only")
    else:
        print("nvidia-ml-py not installed -- logging CPU/RAM only "
              "(pip install nvidia-ml-py for GPU stats)")

    fieldnames = build_fieldnames(gpu_handles, proc is not None)

    # Prime psutil's internal CPU-percent counter; the first real reading
    # needs a prior call to diff against, otherwise it returns 0.0 or 100.0.
    psutil.cpu_percent(interval=None)
    if proc is not None:
        try:
            proc.cpu_percent(interval=None)
        except psutil.NoSuchProcess:
            pass

    stop = {"flag": False}

    def handle_signal(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"Logging every {args.interval}s to {args.out} (Ctrl+C to stop)")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        while not stop["flag"]:
            loop_start = time.perf_counter()

            row = {"timestamp": datetime.now().isoformat(timespec="seconds")}
            row.update(sample_cpu_mem(proc))

            for i, handle in enumerate(gpu_handles):
                gpu_stats = sample_gpu(handle)
                row.update({f"gpu{i}_{k}": v for k, v in gpu_stats.items()})

            writer.writerow(row)
            f.flush()

            # Sleep for the remainder of the interval, accounting for
            # however long the sampling itself took.
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.0, args.interval - elapsed))

    if gpu_handles:
        pynvml.nvmlShutdown()
    print(f"\nStopped. Log written to {args.out}")


if __name__ == "__main__":
    main()
