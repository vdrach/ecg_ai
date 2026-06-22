#!/usr/bin/env python

"""
Convert LTAFDB records into fixed-length ECG windows -- parallel version.

Same input/output as preprocess.py, but processes records concurrently
using a ProcessPoolExecutor. Each record is independent (different input
files, different output files, no shared state), so this is the textbook
"embarrassingly parallel" case -- the only change vs the serial version
is wrapping the per-record work in a function and submitting it to a pool.

Usage:
    python preprocess_parallel.py                    # default: 1 worker (serial)
    python preprocess_parallel.py --workers 16        # 16 worker processes
    python preprocess_parallel.py --workers $(nproc)  # use all available cores

With ~75 records on a 16-core machine, expect roughly 10x speedup before
hitting disk I/O limits. Beyond that, more workers don't help and may hurt
due to disk contention -- worth profiling on your actual hardware.

Input:
    data/raw/ltafdb/

Output:
    data/processed/
        <record>.npz   (one file per input record)

Each .npz contains:
    X.shape = (n_windows, window_samples, n_channels)
    y.shape = (n_windows,)
"""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import argparse

import numpy as np
import wfdb
from tqdm import tqdm

FS = 128  # Hz


def build_label_array(n_samples, ann):
    labels = np.zeros(n_samples, dtype=np.uint8)
    rhythm_events = []

    for sample, note in zip(ann.sample, ann.aux_note):
        note = note.strip()
        if note:
            rhythm_events.append((sample, note))

    for idx in range(len(rhythm_events) - 1):
        start, rhythm = rhythm_events[idx]
        end, _ = rhythm_events[idx + 1]
        if "(AFIB" in rhythm:
            labels[start:end] = 1

    if rhythm_events:
        start, rhythm = rhythm_events[-1]
        if "(AFIB" in rhythm:
            labels[start:] = 1

    return labels


def extract_windows(signal, labels, window_seconds=10, stride_seconds=10):
    window_size = window_seconds * FS
    stride = stride_seconds * FS

    X = []
    y = []

    for start in range(0, len(signal) - window_size, stride):
        end = start + window_size
        window = signal[start:end]
        window_label = labels[start:end]

        # Majority vote
        label = int(window_label.mean() > 0.5)

        X.append(window.astype(np.float32))
        y.append(label)

    return X, y


def process_record(record_path, window_seconds, stride_seconds):
    record = wfdb.rdrecord(str(record_path))
    ann = wfdb.rdann(str(record_path), "atr")

    signal = record.p_signal
    labels = build_label_array(signal.shape[0], ann)

    return extract_windows(signal, labels, window_seconds, stride_seconds)


def process_one(header_file, output_dir, window_seconds, stride_seconds):
    """Process a single record end to end: read, window, save, return
    summary info. Designed to be called by ProcessPoolExecutor workers
    -- must be a top-level function (not a nested def or lambda) so it
    can be pickled and sent to child processes.

    Returns (record_name, n_windows) for the caller to aggregate.
    """
    record_base = header_file.with_suffix("")

    X, y = process_record(record_base, window_seconds, stride_seconds)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.uint8)

    output_file = output_dir / f"{record_base.name}.npz"
    np.savez_compressed(output_file, X=X, y=y)

    return record_base.name, len(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/ltafdb")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--window-seconds", type=int, default=10)
    parser.add_argument("--stride-seconds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1,
                        help="number of parallel worker processes (default: 1 = serial). "
                             "Try setting to the number of available CPU cores.")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = sorted(data_dir.glob("*.hea"))
    print(f"Found {len(records)} records, processing with {args.workers} worker(s)")

    # functools.partial lets us bake in the fixed arguments (output_dir,
    # window_seconds, stride_seconds) so pool.map only needs to vary the
    # one changing argument per task (the header_file). Equivalent to
    # writing a lambda, but pickle-safe (which lambdas are not, so they
    # don't work with ProcessPoolExecutor).
    worker = partial(
        process_one,
        output_dir=output_dir,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
    )

    total_windows = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        # pool.map returns results in the order tasks were submitted,
        # so the per-record print order matches the sorted record list.
        # Wrapping in tqdm gives a progress bar based on completed tasks.
        for name, n in tqdm(pool.map(worker, records), total=len(records)):
            total_windows += n

    print(f"\nSaved {len(records)} files")
    print(f"Total windows: {total_windows}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
