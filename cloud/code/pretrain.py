"""Independent DDQN pretraining for the Cloud adaptation."""
from __future__ import annotations

import argparse
from collections import deque
import random
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch

from .data import load_inputs
from .environment import restore
from .model import DDQN, Transition, encode, decode
from .runner import evaluate, step_env, summarize, output_episode
from .utils import read, save, append, utc, manifest, make_deadline, remaining


def train(data_dir, output, seed, config):
    _, datasets = load_inputs(data_dir)
    data = datasets['greedy_train']
    out = Path(output) / f'seed{seed}'
    out.mkdir(parents=True, exist_ok=False)
    manifest(out, config, data_dir, seed=seed, stage='pretrain', torch_version=torch.__version__)
    state = dict(seed=seed, status='RUNNING', started_utc=utc(), episodes=0,
                 environment_steps=0, updates=0)
    save(out / 'STATE.json', state)
    started, deadline = time.perf_counter(), make_deadline(config)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng, generator = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
    agent = DDQN(config)
    replay = deque(maxlen=config['replay_capacity'])

    def monitor(episode):
        torch.save(agent.checkpoint(), out / f'checkpoint_ep{episode:03d}.pt')
        for split, split_data in datasets.items():
            values = evaluate(agent.online, split_data, out / f'eval_ep{episode:03d}' / split, deadline)
            append(out / 'MONITORING.jsonl', dict(seed=seed, episode=episode,
                   updates=agent.updates, split=split, **values))
            print(f'pretrain seed={seed} episode={episode} split={split} '
                  f'reward={values["reward"]:.6f}', flush=True)

    def stop_signal(signum, frame):
        raise KeyboardInterrupt(f'Stop signal {signum}')

    old_handler = signal.signal(signal.SIGTERM, stop_signal)
    try:
        monitor(0)
        for episode in range(1, config['episodes'] + 1):
            env = restore(data['machines'], data['jobs'], data['initial'], data['patience'])
            offset = len(env.assignments)
            history, rows, times = [], [], []
            env.advance(data['begin'])
            obs, current = encode(env, data['begin'], data['begin'], history)
            for step, now in enumerate(range(data['begin'], data['end'], 60)):
                remaining(deadline)
                epsilon = config['epsilon_end'] + (config['epsilon_start'] - config['epsilon_end']) * max(
                    0., 1. - state['environment_steps'] / config['epsilon_decay_steps'])
                before = time.perf_counter()
                with torch.no_grad():
                    cols, info = decode(obs, agent.online(obs), rng, epsilon, config['allow_idle'])
                reward, assigned = step_env(env, now, cols, obs)
                times.append(time.perf_counter() - before)
                history.append(current)
                env.advance(now + 60)
                next_obs, next_current = encode(env, now + 60, data['begin'], history)
                replay.append(Transition(obs, cols, reward, next_obs, now + 60 == data['end']))
                state['environment_steps'] += 1
                if len(replay) >= config['replay_warmup']:
                    indices = rng.choice(len(replay), size=config['batch_size'], replace=False)
                    log = agent.update([replay[int(i)] for i in indices], generator)
                    assert all(np.isfinite(v) for v in log.values())
                    append(out / 'UPDATES.jsonl', dict(episode=episode, step=step, epsilon=epsilon, **log))
                rows.append(dict(step=step, time=now, reward=sum(a['reward'] for a in assigned),
                    accepted=len(assigned), queued=len(env.pending), cancelled_total=len(env.cancellations),
                    epsilon=epsilon, max_cardinality=info['max_cardinality']))
                obs, current = next_obs, next_current
                state.update(updates=agent.updates, episode_in_progress=episode, step_in_progress=step)
                if step % 25 == 0:
                    save(out / 'STATE.json', state)
            values = summarize(env, data, offset, times)
            append(out / 'TRAIN_EPISODES.jsonl', dict(seed=seed, episode=episode, updates=agent.updates,
                   epsilon_end=epsilon, exploration=True, **values))
            state['episodes'] = episode
            torch.save(agent.checkpoint(), out / 'last.pt')
            if episode in config['monitor_episodes'] or episode == config['episodes']:
                monitor(episode)
            if episode == config['episodes']:
                output_episode(out / 'final_exploration_train', env, data, offset, rows, values)
            save(out / 'STATE.json', state)
        assert all(torch.isfinite(p).all() for p in agent.online.parameters())
        state.update(status='COMPLETE', parameters_finite=True)
    except BaseException as exc:
        state.update(status='FAILED_OR_STOPPED', error=repr(exc), traceback=traceback.format_exc())
        save(out / 'FAILURE.json', state)
        raise
    finally:
        torch.save(agent.checkpoint(), out / 'last.pt')
        state.update(updates=agent.updates, finished_utc=utc(), wall_seconds=time.perf_counter() - started)
        save(out / 'STATE.json', state)
        signal.signal(signal.SIGTERM, old_handler)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/pretrain.json')
    args = parser.parse_args()
    train(args.data_dir, args.output, args.seed, read(args.config))


if __name__ == '__main__':
    main()
