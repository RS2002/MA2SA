"""Shared learning-policy interaction and ledger evaluation."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import torch
from .environment import restore
from .model import encode, decode
from .metrics import metrics as ledger_metrics, resource_check
from .utils import save, write_csv


def step_env(env, now, cols, obs):
    real = cols < len(obs.task)
    offset = len(env.assignments)
    env.apply(cols[real], np.flatnonzero(real), now, obs.legal)
    env.check()
    assigned = env.assignments[offset:]
    reward = np.zeros(len(env.nodes), dtype=np.float32)
    for a in assigned:
        reward[a['server_index']] = a['reward']
    assert np.isclose(reward.sum(dtype=np.float64), sum(a['reward'] for a in assigned), rtol=1e-7, atol=1e-5)
    return reward, assigned


def summarize(env, data, offset, decision_times):
    jobs, initial, begin, end = (data[k] for k in ('jobs', 'initial', 'begin', 'end'))
    assignments = env.assignments[offset:]
    initial_ids = set(initial['initial_queue_ids'])
    population = [j for j in jobs if j['job_id'] in initial_ids or begin <= j['submit'] < end]
    by_id = {j['job_id']: j for j in jobs}
    totals = dict(cpu=float(env.cap_cpu.sum()) / 100, gpu=float(env.cap_gpu.sum()) / 100)
    args = (assignments, env.cancellations, initial['background_assignments'], by_id,
            initial_ids, begin, end, totals, begin, decision_times)
    values = ledger_metrics(population, *args)
    assert values['pending_jobs'] == len(env.pending)
    assert values['cancelled_jobs'] == len(env.cancellations)
    assert values['reward_units'] == sum(a['reward_units'] for a in assignments)
    values['subgroups'] = {}
    for name, cpu in [('cpu_only', True), ('gpu', False)]:
        background = [a for a in initial['background_assignments']
                      if (by_id[a['job_id']]['gpu_units'] == 0) == cpu]
        values['subgroups'][name] = ledger_metrics(
            [j for j in population if (j['gpu_units'] == 0) == cpu],
            assignments, env.cancellations, background, by_id, initial_ids,
            begin, end, totals, begin, decision_times)
    return values


def output_episode(output, env, data, offset, rows, values):
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'METRICS.json', values)
    write_csv(output / 'ASSIGNMENTS.csv', env.assignments[offset:])
    write_csv(output / 'CANCELLATIONS.csv', env.cancellations)
    write_csv(output / 'STEPS.csv', rows)
    save(output / 'FINAL_QUEUE.json', env.pending)
    capacity = resource_check(data['machines'], data['initial']['background_assignments'],
                             env.assignments[offset:], {j['job_id']: j for j in data['jobs']},
                             data['begin'], data['end'])
    save(output / 'CAPACITY_CHECK.json', capacity)


def evaluate(model, data, output=None, deadline=None):
    env = restore(data['machines'], data['jobs'], data['initial'], data['patience'])
    offset = len(env.assignments)
    history, rows, times = [], [], []
    model.eval()
    with torch.no_grad():
        for step, now in enumerate(range(data['begin'], data['end'], 60)):
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError('Frozen per-seed wall budget reached during evaluation')
            env.advance(now)
            before = time.perf_counter()
            obs, current = encode(env, now, data['begin'], history)
            cols, info = decode(obs, model(obs))
            _, assigned = step_env(env, now, cols, obs)
            history.append(current)
            times.append(time.perf_counter() - before)
            rows.append(dict(step=step, time=now, accepted=len(assigned), queued=len(env.pending),
                cancelled_total=len(env.cancellations), reward=sum(a['reward'] for a in assigned),
                max_cardinality=info['max_cardinality'], decision_seconds=times[-1]))
        env.advance(data['end'])
        env.check()
    values = summarize(env, data, offset, times)
    if output is not None:
        output_episode(output, env, data, offset, rows, values)
    return values
