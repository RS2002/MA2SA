"""Prepare the frozen experiment protocol from user-downloaded public CSVs.

No dataset is distributed with this code. Preparation computes the common
historical resource snapshots once, then training restores them each episode.
Run ``python -m ma2sa_cloud.data --help`` for the required local input paths.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .environment import BatchEnvironment, HIDDEN_FIELDS, pricing, snapshot
from .matching import _history_assignment

JOBS_URL = "https://raw.githubusercontent.com/alibaba/clusterdata/3858992ca41bf92082e6d6522ddf0bebd14e838b/cluster-trace-gpu-v2020/simulator/traces/pai/pai_job_no_estimate_100K.csv"
MACHINES_URL = "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_machine_spec.tar.gz"
DATASET_URL = "https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020"
EXPECTED_JOBS_SHA256 = "dc4de3214e15b7c758b6485aaa1766c0c347a2482693e069ce7124009d1db56d"
EXPECTED_MACHINES_SHA256 = "251e9e33649155a6a935fdc1dc0e90cd4698206936b60bd8fce60ff2ccd51439"
TRAIN_ROWS, SCORE_SECONDS, INTERVAL, PATIENCE_SEED = 30000, 18000, 60, 20261027


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _exact(value):
    return dict(numerator=value.numerator, denominator=value.denominator, float=float(value))


def _ceil(value):
    return -(-value.numerator // value.denominator)


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _read(path):
    return json.loads(Path(path).read_text())


def _read_source(jobs_csv, machines_csv):
    machines = []
    with Path(machines_csv).open() as stream:
        for index, row in enumerate(csv.reader(stream)):
            machine_id, gpu_type, cpu, memory_gb, gpu = row
            machines.append(dict(index=index, machine_id=machine_id, gpu_type=gpu_type,
                cpu_units=int(Decimal(cpu) * 100), gpu_units=int(Decimal(gpu) * 100),
                memory_gb=float(memory_gb)))
    if len(machines) != 1897 or len({m["machine_id"] for m in machines}) != 1897:
        raise ValueError("the protocol requires the full 1897-machine inventory")
    cpus = np.array([machine["cpu_units"] for machine in machines], dtype=np.int64)
    gpus = np.array([machine["gpu_units"] for machine in machines], dtype=np.int64)
    if int(cpus.sum()) != 15657600 or int(gpus.sum()) != 674200:
        raise ValueError("machine resource totals differ from the frozen protocol")
    source = []
    with Path(jobs_csv).open() as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            instances = int(Decimal(row["num_inst"]))
            if instances <= 0 or Decimal(row["num_inst"]) != instances:
                raise ValueError("num_inst must be a positive integer")
            # Preserve the official loader's order: round each instance first.
            cpu = round(float(row["num_cpu"]) * 100) * instances
            gpu = round(float(row["num_gpu"]) * 100) * instances
            duration, submit = int(row["duration"]), int(row["submit_time"])
            if duration <= 0 or cpu < 0 or gpu < 0 or cpu + gpu <= 0:
                raise ValueError("invalid resource request / runtime")
            reason = None
            if cpu > int(cpus.sum()) or gpu > int(gpus.sum()):
                reason = "exceeds_cluster_total"
            elif not np.any((cpus >= cpu) & (gpus >= gpu)):
                reason = "no_single_machine_can_fit_whole_job"
            source.append(dict(job_id=int(row["job_id"]), source_row=index,
                source_csv_line=index + 2, source_submit=submit, duration=duration,
                num_inst=instances, cpu_units=cpu, gpu_units=gpu,
                recorded_gpu_type=row["gpu_type"], user=row["user"], excluded_reason=reason))
    if len(source) != 100000 or len({job["job_id"] for job in source}) != len(source):
        raise ValueError("expected the original 100000 unique jobs")
    if any(a["source_submit"] > b["source_submit"] for a, b in zip(source, source[1:])):
        raise ValueError("source rows must be in original submission order")
    return machines, source


def _calibration(machines, source):
    train = [job for job in source[:TRAIN_ROWS] if job["excluded_reason"] is None]
    span = source[TRAIN_ROWS - 1]["source_submit"] - source[0]["source_submit"]
    cpu_rho = Fraction(sum(j["cpu_units"] * j["duration"] for j in train),
                       span * sum(m["cpu_units"] for m in machines))
    gpu_rho = Fraction(sum(j["gpu_units"] * j["duration"] for j in train),
                       span * sum(m["gpu_units"] for m in machines))
    base = Fraction(85, 100) / max(cpu_rho, gpu_rho)
    peak = Fraction(300, 1) / Fraction(len(train) * 60, span)
    return base, peak


def _scenario(source, split, base, peak):
    training = split == "train"
    history = 4800 if training else 28800
    chosen_source = source[:TRAIN_ROWS] if training else source[TRAIN_ROWS:]
    origin = chosen_source[0]["source_submit"]
    seam = Fraction(origin) + history * base
    scale = Fraction(78, 100) if training else Fraction(1)
    high, low = peak * scale, base / 10 * scale

    def clock(t):
        if t <= 3600:
            return seam + high * t
        if t <= 7200:
            v = t - 3600
            return seam + high * 3600 + high * v + (low - high) * v * v / 7200
        return seam + 3600 * high + 1800 * (high + low) + low * (t - 7200)

    thresholds = [clock(t) for t in range(SCORE_SECONDS + 1)]
    if thresholds[-1] > chosen_source[-1]["source_submit"]:
        raise ValueError("insufficient unrepeated source coverage for frozen arrival clock")
    jobs = []
    for raw in chosen_source:
        if raw["excluded_reason"] is not None:
            continue
        job = {key: value for key, value in raw.items() if key != "excluded_reason"}
        source_time = Fraction(job["source_submit"])
        # DEV's original first-hour mapping was rational, retained by the taper.
        rational = source_time < seam or (not training and source_time <= thresholds[3600])
        if rational:
            real = ((source_time - origin) / base if source_time < seam
                    else history + (source_time - seam) / peak)
            submit = _ceil(real)
            if not 0 <= submit < history + SCORE_SECONDS:
                continue
            job.update(submit=submit, new_submit_real=float(real),
                       new_submit_numerator=real.numerator, new_submit_denominator=real.denominator)
        else:
            second = bisect_left(thresholds, source_time)
            if second >= SCORE_SECONDS:
                continue
            job.update(submit=history + second,
                       arrival_mapping="ceil_inverse_integrated_linear_taper_source_clock")
            if second:
                job.update(lower_source_threshold_numerator=thresholds[second - 1].numerator,
                    lower_source_threshold_denominator=thresholds[second - 1].denominator,
                    upper_source_threshold_numerator=thresholds[second].numerator,
                    upper_source_threshold_denominator=thresholds[second].denominator)
        job.update(pricing(job))
        jobs.append(job)
    if jobs != sorted(jobs, key=lambda job: (job["submit"], job["job_id"])):
        raise ValueError("mapped jobs are not sorted")
    return jobs, dict(split=split, history_seconds=history, score_seconds=SCORE_SECONDS,
        score_start=history, score_end=history + SCORE_SECONDS, interval=INTERVAL,
        train_score_clock_scale=_exact(scale), warmup_alpha_base_exact=_exact(base),
        reference_alpha_peak_exact=_exact(peak), scoring_alpha_peak_exact=_exact(high),
        scoring_alpha_low_exact=_exact(low), common_source_seam=_exact(seam),
        hourly_new_arrivals=[sum(history + h * 3600 <= j["submit"] < history + (h + 1) * 3600
                                for j in jobs) for h in range(5)])


def _history_snapshot(machines, jobs, history, split):
    env = BatchEnvironment(machines, jobs)
    for now in range(0, history, INTERVAL):
        env.advance(now)
        legal, free_cpu, free_gpu, req_cpu, req_gpu = env.observation()
        kept = np.flatnonzero(legal.any(axis=1))
        selected_jobs = [env.pending[int(j)] for j in kept]
        local, servers = _history_assignment(legal[kept], free_cpu, free_gpu,
                                             req_cpu[kept], req_gpu[kept], selected_jobs)
        env.apply(kept[local], servers, now, legal)
        env.check()
    env.advance(history)
    # An exact-boundary arrival belongs to scoring, not the carry-in queue.
    env.pending = [job for job in env.pending if job["submit"] < history]
    env.next_job = sum(job["submit"] < history for job in jobs)
    return snapshot(env, history, f"{split}_fixed_common_history")


def _patience(jobs, initial, end):
    start, initial_ids = initial["time"], set(initial["initial_queue_ids"])
    records = []
    for job in jobs:
        if job["job_id"] not in initial_ids and not start <= job["submit"] < end:
            continue
        key = f'ma2sa-cloud-patience-v1:{PATIENCE_SEED}:{job["job_id"]}'
        integer = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") >> 11
        uniform = (integer + 1) / (2**53 + 2)
        age = start - job["submit"] if job["job_id"] in initial_ids else 0
        draw = math.sqrt(age * age - 7200**2 * math.log(uniform) / math.log(2))
        seconds = max(age + 1, math.ceil(draw))
        records.append(dict(job_id=job["job_id"], submit=job["submit"], patience_seconds=seconds,
            deadline=job["submit"] + seconds,
            conditional_on_survived_to_score_start=job["job_id"] in initial_ids,
            conditional_age_seconds=age, uniform_open01=uniform))
    return dict(seed=PATIENCE_SEED, shape=2, median_seconds=7200,
        distribution="Weibull; initial queue conditional on survival to score_start",
        empirical_calibration=False, hidden_from_policies=True, records=records)


def prepare_data(jobs_csv, machines_csv, output):
    """Create new files, rejecting overwrite and nonmatching source CSVs."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refuse to overwrite a nonempty processed-data directory: {output}")
    hashes = dict(jobs_csv=digest(jobs_csv), machines_csv=digest(machines_csv))
    if hashes["jobs_csv"] != EXPECTED_JOBS_SHA256 or hashes["machines_csv"] != EXPECTED_MACHINES_SHA256:
        raise ValueError("input checksum differs; download the exact linked CSVs without modifying them")
    machines, source = _read_source(jobs_csv, machines_csv)
    base, peak = _calibration(machines, source)
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "MACHINES.json", dict(machines=machines))
    splits = {}
    for split in ("train", "dev"):
        jobs, manifest = _scenario(source, split, base, peak)
        initial = _history_snapshot(machines, jobs, manifest["history_seconds"], split)
        patience = _patience(jobs, initial, manifest["score_end"])
        destination = output / split
        destination.mkdir()
        _write(destination / "SCENARIO.json", dict(jobs=jobs, manifest=manifest))
        _write(destination / "INITIAL_SNAPSHOT.json", initial)
        _write(destination / "PATIENCE.json", patience)
        splits[split] = dict(jobs=len(jobs), new_arrivals=sum(manifest["score_start"] <= j["submit"]
             < manifest["score_end"] for j in jobs), initial_queue=len(initial["initial_queue_ids"]),
             initial_running=len(initial["background_assignments"]), **manifest)
        print(f'prepared {split}: {len(jobs)} jobs, {splits[split]["initial_running"]} running background jobs', flush=True)
    train_ids = {j["job_id"] for j in _read(output / "train/SCENARIO.json")["jobs"]}
    dev_ids = {j["job_id"] for j in _read(output / "dev/SCENARIO.json")["jobs"]}
    if train_ids & dev_ids:
        raise ValueError("TRAIN/DEV job identity overlap")
    protocol = dict(source_sha256=hashes, source_urls=dict(jobs=JOBS_URL, machines=MACHINES_URL),
        gamma=.99, decision_interval_seconds=60, score_decisions=300,
        reward_formula="1.5e-7+(0.072*CPU_cores+2.4192*GPU_cards)*duration_hours-0.01*wait_minutes",
        runtime="fixed oracle duration; no co-location slowdown; release on completion",
        patience="hidden Weibull(shape=2, median=120min), seed20261027; conditional carry-in queue",
        initial_history="common fixed FIFO resource placement; no historical cancellation rerun",
        whole_job_same_machine=True, max_new_jobs_per_server_per_decision=1, allow_voluntary_dummy=False,
        splits=splits, split_source_rows=dict(train="[0,30000)", dev="[30000,100000)"),
        limitations=["TRAIN scoring arrival clock is 0.78 of DEV and warmup is 80min versus 8h.",
            "Repeatedly observed DEV is not an independent final test.",
            "Reference contract value includes service beyond the episode; not realized window income.",
            "Memory, GPU model constraints, fragmentation and co-location slowdown are not modeled.",
            "The controlled peak/recovery arrival clock and patience parameters are research assumptions."])
    _write(output / "DATA_PROTOCOL.json", protocol)
    return protocol


def load_inputs(data_dir):
    """Return ``machines, {'greedy_train': train, 'dev': dev}`` for runners."""
    root = Path(data_dir)
    machines = _read(root / "MACHINES.json")["machines"]
    datasets = {}
    for split, name in (("train", "greedy_train"), ("dev", "dev")):
        scenario = _read(root / split / "SCENARIO.json")
        initial = _read(root / split / "INITIAL_SNAPSHOT.json")
        records = _read(root / split / "PATIENCE.json")["records"]
        jobs = scenario["jobs"]
        if any(not HIDDEN_FIELDS.isdisjoint(job) for job in jobs):
            raise ValueError("private cancellation information leaked into visible jobs")
        begin = scenario["manifest"]["score_start"]
        end = scenario["manifest"]["score_end"]
        datasets[name] = dict(machines=machines, jobs=jobs, initial=initial,
            patience={record["job_id"]: record for record in records}, begin=begin, end=end)
    return machines, datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-csv", type=Path, required=True, help="downloaded pai_job_no_estimate_100K.csv")
    parser.add_argument("--machines-csv", type=Path, required=True, help="extracted pai_machine_spec.csv (no header)")
    parser.add_argument("--output", type=Path, required=True, help="new/empty processed-data directory")
    args = parser.parse_args()
    prepare_data(args.jobs_csv, args.machines_csv, args.output)


if __name__ == "__main__":
    main()
