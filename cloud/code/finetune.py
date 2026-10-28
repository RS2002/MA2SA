"""Centralized fine-tuning from a pretrained Cloud policy checkpoint."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import gc
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch

from .central_model import CentralActor, TwinMatchingCritic, encode, decode
from .central_trainer import CentralTrainer, tensor_hash
from .data import load_inputs
from .environment import restore
from .runner import step_env, summarize, output_episode, evaluate
from .utils import read, save, append, utc, digest, manifest, make_deadline, remaining


@dataclass
class FineTransition:
    obs: object
    cols: np.ndarray
    reward: np.ndarray
    next_obs: object
    done: bool
    mc_return: float | None = None


def noisy(obs, raw, rng, sigma):
    scores = raw.detach().cpu().numpy().astype(np.float64)
    scores += rng.normal(0., sigma, scores.shape)
    return decode(obs, scores, allow_idle=False)


def episode(model, data, rng, deadline, sigma=0., on_transition=None):
    env = restore(data['machines'], data['jobs'], data['initial'], data['patience'])
    offset = len(env.assignments)
    history, rows, timings, transitions = [], [], [], []
    env.advance(data['begin'])
    obs, current = encode(env, data['begin'], data['begin'], history)
    for step, now in enumerate(range(data['begin'], data['end'], 60)):
        remaining(deadline)
        started = time.perf_counter()
        with torch.no_grad():
            raw = model(obs)
            cols, info = noisy(obs, raw, rng, sigma) if sigma else decode(obs, raw, allow_idle=False)
        reward, assigned = step_env(env, now, cols, obs)
        timings.append(time.perf_counter() - started)
        history.append(current)
        env.advance(now + 60)
        next_obs, next_current = encode(env, now + 60, data['begin'], history)
        tr = FineTransition(obs, cols, reward, next_obs, now + 60 == data['end'])
        if on_transition is None:
            transitions.append(tr)
        else:
            on_transition(tr, step, next_obs)
        rows.append(dict(step=step, time=now, reward=sum(a['reward'] for a in assigned),
            accepted=len(assigned), queued=len(env.pending), cancelled_total=len(env.cancellations),
            max_cardinality=info['max_cardinality'], matched_count=info['matched_count'],
            decision_seconds=timings[-1]))
        obs, current = next_obs, next_current
    return env, offset, rows, summarize(env, data, offset, timings), transitions


def counts(agent):
    return dict(critic_updates=agent.warmup_updates + agent.critic_updates,
                mc_critic_updates=agent.warmup_updates, td_critic_updates=agent.critic_updates,
                actor_updates=agent.actor_updates, target_updates=agent.target_updates)


def train(data_dir, output, seed, config, checkpoint):
    _, datasets = load_inputs(data_dir)
    data = datasets['greedy_train']
    parent = torch.load(checkpoint, map_location='cpu', weights_only=False)
    out = Path(output) / f'seed{seed}'
    out.mkdir(parents=True, exist_ok=False)
    manifest(out, config, data_dir, seed=seed, stage='central_finetune',
             parent_checkpoint=str(Path(checkpoint).resolve()), parent_sha256=digest(checkpoint),
             torch_version=torch.__version__)
    started, deadline = time.perf_counter(), make_deadline(config)
    state = dict(seed=seed, status='COLLECTING_FROZEN_PARENT', started_utc=utc(),
                 environment_steps=0, joint_episodes_completed=0)
    save(out / 'STATE.json', state)
    agent = None

    def stop_signal(signum, frame):
        raise KeyboardInterrupt(f'Stop signal {signum}')

    previous_handler = signal.signal(signal.SIGTERM, stop_signal)
    try:
        torch.manual_seed(22000 + seed)
        actor = CentralActor()
        actor.policy.load_state_dict(parent['online'])
        frozen_hash = tensor_hash(actor)
        collection_rng = np.random.default_rng(12000 + seed)
        warmup = []
        for ep in range(1, config['warmup_collection_episodes'] + 1):
            env, offset, rows, values, transitions = episode(actor, data, collection_rng, deadline,
                sigma=config['collection_noise'])
            cumulative = 0.
            for tr in reversed(transitions):
                cumulative = float(np.asarray(tr.reward, dtype=np.float64).sum()) + config['gamma'] * cumulative
                tr.mc_return = cumulative
            assert tensor_hash(actor) == frozen_hash
            warmup.extend(transitions)
            output_episode(out / 'collection' / f'ep{ep:03d}', env, data, offset, rows, values)
            append(out / 'COLLECTION.jsonl', dict(seed=seed, episode=ep, **values))
        with torch.no_grad():
            raw = actor.policy(warmup[0].obs)
            assert torch.equal(raw, actor(warmup[0].obs))
            cols, _ = decode(warmup[0].obs, raw)
            wrapped, _ = decode(warmup[0].obs, actor(warmup[0].obs))
            assert np.array_equal(cols, wrapped)
        save(out / 'PARENT_BRIDGE.json', dict(raw_exact=True, matching_exact=True, actor_unchanged=True,
             parent_sha256=digest(checkpoint), transitions=len(warmup)))
        del actor, transitions, env
        gc.collect()

        torch.manual_seed(32000 + seed)
        np_rng = np.random.default_rng(42000 + seed)
        sample_rng = np.random.default_rng(52000 + seed)
        actor = CentralActor()
        actor.policy.load_state_dict(parent['online'])
        agent = CentralTrainer(actor, TwinMatchingCritic(), config, seed)
        replay = deque(warmup, maxlen=config['replay_capacity'])

        def sample():
            indices = sample_rng.choice(len(replay), size=config['batch_size'], replace=False)
            return [replay[int(i)] for i in indices]

        def monitor(ep):
            torch.save(agent.state_dict(), out / f'checkpoint_ep{ep:03d}.pt')
            for split, split_data in datasets.items():
                values_eval = evaluate(agent.actor, split_data, out / f'eval_ep{ep:03d}' / split, deadline)
                append(out / 'MONITORING.jsonl', dict(seed=seed, episode=ep, split=split,
                       **counts(agent), **values_eval))
                print(f'finetune seed={seed} episode={ep} split={split} '
                      f'reward={values_eval["reward"]:.6f}', flush=True)

        # Deterministic monitoring changes neither parameters nor training RNGs.
        monitor(0)
        state.update(status='RUNNING', phase='warmup', **counts(agent))
        for update in range(1, config['warmup_updates'] + 1):
            remaining(deadline)
            log = agent.warmup_mc_update(sample())
            append(out / 'UPDATES.jsonl', dict(log, warmup_index=update, phase='warmup'))
        save(out / 'WARMUP_FREEZE_CHECK.json', agent.finish_warmup())
        torch.save(agent.state_dict(), out / 'checkpoint_after_warmup.pt')
        state['phase'] = 'joint'
        for ep in range(1, config['joint_episodes'] + 1):
            def update_step(tr, step, online_obs):
                replay.append(tr)
                state['environment_steps'] += 1
                if state['environment_steps'] % config['update_every_steps'] == 0:
                    log = agent.step(sample(), tr.obs)
                    assert all(np.isfinite(v) for v in log.values() if isinstance(v, (int, float, np.number)))
                    append(out / 'UPDATES.jsonl', dict(log, episode=ep, step=step, phase='joint'))
                state.update(episode_in_progress=ep, step_in_progress=step, **counts(agent))
                if step % 25 == 0:
                    save(out / 'STATE.json', state)
            env, offset, rows, values, _ = episode(agent.actor, data, np_rng, deadline,
                sigma=config['collection_noise'], on_transition=update_step)
            append(out / 'TRAIN_EPISODES.jsonl', dict(seed=seed, episode=ep, **counts(agent), **values))
            state['joint_episodes_completed'] = ep
            torch.save(agent.state_dict(), out / 'last.pt')
            if ep in config['monitor_episodes'] or ep == config['joint_episodes']:
                monitor(ep)
            if ep == config['joint_episodes']:
                output_episode(out / 'final_exploration_train', env, data, offset, rows, values)
            save(out / 'STATE.json', state)
        assert all(torch.isfinite(p).all() for module in [agent.actor, agent.critics] for p in module.parameters())
        state.update(status='COMPLETE', parameters_finite=True)
    except BaseException as exc:
        state.update(status='FAILED_OR_STOPPED', error=repr(exc), traceback=traceback.format_exc())
        save(out / 'FAILURE.json', state)
        raise
    finally:
        if agent is not None:
            torch.save(agent.state_dict(), out / 'last.pt')
            state.update(**counts(agent))
        state.update(finished_utc=utc(), wall_seconds=time.perf_counter() - started)
        save(out / 'STATE.json', state)
        signal.signal(signal.SIGTERM, previous_handler)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/finetune.json')
    args = parser.parse_args()
    train(args.data_dir, args.output, args.seed, read(args.config), args.checkpoint)


if __name__ == '__main__':
    main()
