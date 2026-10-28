"""Oracle-duration independent DDQN adaptation, with causal public context.

The legal graph never enters the feature encoders. Each server has a local
TD reward. A shared, exact maximum-cardinality decoder enforces joint actions.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from .matching import _fixed_cardinality_assignment
from .qk import qk_scores, normalized_square

torch.set_num_threads(1)
torch.backends.mha.set_fastpath_enabled(False)
MACHINE_DIM, TASK_DIM, RUNNING_DIM, GLOBAL_DIM = 12, 10, 7, 12


def logscale(value, scale):
    return np.log1p(np.maximum(value, 0.)) / np.log1p(scale)


@dataclass
class Observation:
    machine: np.ndarray
    task: np.ndarray
    running: np.ndarray
    owner: np.ndarray
    global_: np.ndarray
    legal: np.ndarray  # jobs x servers, used only after neural scoring
    job_ids: np.ndarray


def encode(env, now, score_start, history):
    legal, fc, fg, rc, rg = env.observation()
    machines, tasks = env.machines, env.pending
    capc, capg = env.cap_cpu, env.cap_gpu
    records = sorted(env.events, key=lambda event: (event[2], event[1]))
    # The episode job list is immutable. Cache only the index, never features
    # of future jobs: below, lookup remains restricted to currently running IDs.
    cached = getattr(env, '_pretrain_job_index', None)
    if cached is None or cached[0] is not env.jobs or cached[1] != len(env.jobs):
        cached = (env.jobs, len(env.jobs), {j['job_id']: j for j in env.jobs})
        env._pretrain_job_index = cached
    by_id = cached[2]
    owner = np.asarray([e[2] for e in records], dtype=np.int64)
    active = [by_id[event[1]] for event in records]
    duration_running = np.asarray([j['duration'] for j in active], dtype=np.float64)
    remaining = np.asarray([event[0] - now for event in records], dtype=np.float64)
    elapsed = duration_running - remaining
    assert np.all((elapsed >= 0) & (remaining > 0) & (remaining <= duration_running))
    cpu_running = np.asarray([j['cpu_units'] for j in active], dtype=np.float64)
    gpu_running = np.asarray([j['gpu_units'] for j in active], dtype=np.float64)
    inst_running = np.asarray([j['num_inst'] for j in active], dtype=np.float64)
    running = np.column_stack((cpu_running / 9600., gpu_running / 800.,
        logscale(duration_running, 86400.), logscale(elapsed, 86400.),
        logscale(remaining, 86400.), remaining / 3600., logscale(inst_running, 1024.)))
    counts = np.bincount(owner, minlength=len(machines)).astype(np.float32)
    soon = remaining <= 600
    remaining_c = np.bincount(owner[soon], weights=cpu_running[soon], minlength=len(machines)).astype(np.float32)
    remaining_g = np.bincount(owner[soon], weights=gpu_running[soon], minlength=len(machines)).astype(np.float32)
    step = (now - score_start) / 60.
    machine = np.column_stack((capc / 9600., capg / 800., fc / 9600., fg / 800.,
        fc / capc, fg / np.maximum(capg, 1), capg == 0, logscale(counts, 1024.),
        remaining_c / 9600., remaining_g / 800.,
        np.full(len(machines), step / 120.), np.full(len(machines), step / 120.)))
    # Last column explicitly distinguishes elapsed wall time in hours.
    machine[:, -1] = (now - score_start) / 3600.
    duration = np.asarray([j['duration'] for j in tasks], dtype=np.float64)
    wait = np.asarray([now - j['submit'] for j in tasks], dtype=np.float64)
    inst = np.asarray([j['num_inst'] for j in tasks], dtype=np.float64)
    task = np.column_stack((rc / 9600., rg / 800., rg == 0,
        logscale(inst, 1024.), logscale(duration, 86400.), duration / 3600.,
        logscale(wait, 7200.), wait / 600.,
        np.full(len(tasks), step / 120.), np.full(len(tasks), (now - score_start) / 3600.)))
    current = np.asarray([step / 120., len(tasks) / 1000., len(records) / 10000.,
        fc.sum() / capc.sum(), fg.sum() / max(1, capg.sum()),
        rc.sum() / capc.sum(), rg.sum() / max(1, capg.sum()),
        wait.mean() / 3600. if len(wait) else 0., wait.max() / 3600. if len(wait) else 0.,
        (rg > 0).sum() / 1000., (rg == 0).sum() / 1000.,
        (counts > 0).mean()], dtype=np.float32)
    # Explicit causal timestamps in each token; no arbitrary entity positions.
    globals_ = np.asarray([current] + [np.zeros(GLOBAL_DIM)] * (3 - len(history[-3:]))
                          + history[-3:], dtype=np.float32)
    obs = Observation(machine.astype(np.float32), task.astype(np.float32),
        np.asarray(running, dtype=np.float32).reshape(-1, RUNNING_DIM), owner,
        globals_, legal.copy(), np.asarray([j['job_id'] for j in tasks], dtype=np.int64))
    for name in ('machine', 'task', 'running', 'global_'):
        assert np.isfinite(getattr(obs, name)).all(), name
    return obs, current


class MLP(nn.Module):
    def __init__(self, width, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 2 * dim), nn.GELU(),
                                 nn.Linear(2 * dim, dim), nn.LayerNorm(dim))

    def forward(self, x):
        return self.net(x)


class RunningPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encode, self.attend = MLP(RUNNING_DIM, dim), nn.Linear(dim, 1)

    def forward(self, values, owner, count):
        out = values.new_zeros((count, self.attend.in_features))
        if len(values) == 0:
            return out
        encoded = self.encode(values)
        logits = self.attend(encoded).flatten()
        maximum = logits.new_full((count,), -torch.inf)
        maximum.scatter_reduce_(0, owner, logits, reduce='amax', include_self=True)
        weights = (logits - maximum[owner]).exp()
        total = logits.new_zeros(count).index_add(0, owner, weights)
        return out.index_add(0, owner, encoded * (weights / total[owner])[:, None])


class QNetwork(nn.Module):
    def __init__(self, dim=64, rank=32):
        super().__init__()
        self.machine = MLP(MACHINE_DIM, dim)
        self.task = MLP(TASK_DIM, dim)
        self.running = RunningPool(dim)
        self.global_projection = nn.Linear(GLOBAL_DIM, dim)
        layer = nn.TransformerEncoderLayer(dim, 4, 4 * dim, dropout=0.,
                                          activation='gelu', batch_first=True, norm_first=True)
        self.global_encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.global_pool = nn.Linear(dim, 1)
        self.machine_combine = nn.Sequential(nn.Linear(3 * dim, dim), nn.LayerNorm(dim))
        self.task_combine = nn.Sequential(nn.Linear(2 * dim, dim), nn.LayerNorm(dim))
        self.query, self.key = nn.Linear(dim, rank), nn.Linear(dim, rank)
        self.null = nn.Parameter(torch.randn(1, dim) * .02)
        self.reconstruction, self.prediction = nn.Linear(dim, GLOBAL_DIM), nn.Linear(dim, GLOBAL_DIM)

    def global_tokens(self, value):
        return self.global_encoder(self.global_projection(value)[None])[0]

    def entities(self, obs):
        mf, tf, running, gf = [torch.as_tensor(x, dtype=torch.float32) for x in
                              (obs.machine, obs.task, obs.running, obs.global_)]
        owner = torch.as_tensor(obs.owner, dtype=torch.long)
        global_tokens = self.global_tokens(gf)
        context = (global_tokens * self.global_pool(global_tokens).softmax(0)).sum(0, keepdim=True)
        machine = self.machine_combine(torch.cat((self.machine(mf), self.running(running, owner, len(mf)),
                                                   context.expand(len(mf), -1)), dim=1))
        tasks = torch.cat((self.task(tf), self.null), dim=0)
        tasks = self.task_combine(torch.cat((tasks, context.expand(len(tasks), -1)), dim=1))
        return self.query(machine), self.key(tasks)

    def forward(self, obs):
        q, k = self.entities(obs)
        return qk_scores(q, k, 'square_l2')

    def selected(self, obs, cols):
        q, k = self.entities(obs)
        keys = normalized_square(k)
        return (q * keys[torch.as_tensor(cols, dtype=torch.long)]).sum(-1)

    def auxiliary(self, current, next_, done, generator):
        target = torch.as_tensor(current, dtype=torch.float32)
        mask = torch.rand(target.shape, generator=generator) < .15
        noise = torch.randn(target.shape, generator=generator)
        tokens = self.global_tokens(torch.where(mask, noise, target))
        recon = F.mse_loss(self.reconstruction(tokens), target)
        pred = (F.mse_loss(self.prediction(tokens), torch.as_tensor(next_, dtype=torch.float32))
                if not done else tokens.sum() * 0.)
        return recon + pred, recon, pred


def decode(obs, scores, rng=None, epsilon=0., allow_idle=False):
    """All paths share the legal graph, uniqueness and exact maximum rank.

    Null Q only prices which servers are forced unmatched at fixed rank.
    It never permits reducing cardinality when allow_idle=False (the default).
    """
    values = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)
    jobs, servers = obs.legal.shape
    assert values.shape == (servers, jobs + 1) and np.isfinite(values).all()
    if rng is not None and rng.random() < epsilon:
        values = rng.uniform(-1, 1, values.shape)
    kept = np.flatnonzero(obs.legal.any(1))
    legal = obs.legal[kept]
    cols = np.full(servers, jobs, dtype=np.int64)
    if len(kept) == 0:
        return cols, {'max_cardinality': 0, 'matched_count': 0, 'legal_edges': 0}
    graph = csr_matrix(legal)
    rank = int((maximum_bipartite_matching(graph, perm_type='column') >= 0).sum())
    gains = values[:, kept].T.astype(np.float64) - values[:, -1].astype(np.float64)[None, :]
    if allow_idle:
        from scipy.optimize import linear_sum_assignment
        cost = np.concatenate((np.where(legal, -gains, np.inf), np.zeros((len(kept), len(kept)))), 1)
        ji, si = linear_sum_assignment(cost)
        keep = si < servers
        ji, si = ji[keep], si[keep]
    else:
        ji, si = _fixed_cardinality_assignment(-gains, legal, rank)
    cols[si] = kept[ji]
    real = cols < jobs
    assert len(set(cols[real].tolist())) == int(real.sum())
    assert obs.legal[cols[real], np.flatnonzero(real)].all()
    assert allow_idle or int(real.sum()) == rank
    return cols, dict(max_cardinality=rank, matched_count=int(real.sum()), legal_edges=int(legal.sum()))


@dataclass
class Transition:
    obs: Observation
    cols: np.ndarray
    reward: np.ndarray
    next_obs: Observation
    done: bool


class DDQN:
    def __init__(self, config):
        self.config = config
        self.online = QNetwork(config['width'], config['rank']).eval()
        self.target = copy.deepcopy(self.online).eval().requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config['learning_rate'])
        self.updates = 0

    def update(self, transitions, generator):
        self.optimizer.zero_grad(set_to_none=True)
        result = []
        for tr in transitions:
            chosen = self.online.selected(tr.obs, tr.cols)
            with torch.no_grad():
                label = torch.as_tensor(tr.reward / self.config['reward_scale'], dtype=torch.float32)
                if not tr.done:
                    cols, _ = decode(tr.next_obs, self.online(tr.next_obs), allow_idle=self.config['allow_idle'])
                    label = label + self.config['gamma'] * self.target.selected(tr.next_obs, cols)
            error = F.smooth_l1_loss(chosen, label, reduction='none')
            real = tr.cols < len(tr.obs.task)
            # Diagnostics only. Signed TD error means target minus prediction.
            delta = label - chosen.detach()
            diagnostics = dict(td_error_signed_mean=float(delta.mean()),
                               td_error_abs_mean=float(delta.abs().mean()))
            for group_name, group in (('real', real), ('idle', ~real)):
                group_mask = torch.as_tensor(group)
                group_count = int(group.sum())
                diagnostics[f'{group_name}_servers'] = group_count
                if group_count:
                    diagnostics.update({
                        f'{group_name}_td_error_signed_mean': float(delta[group_mask].mean()),
                        f'{group_name}_td_error_abs_mean': float(delta[group_mask].abs().mean()),
                        f'{group_name}_q_mean': float(chosen.detach()[group_mask].mean()),
                        f'{group_name}_target_mean': float(label[group_mask].mean()),
                        f'{group_name}_td_huber_mean': float(error.detach()[group_mask].mean()),
                    })
                else:
                    # A zero count distinguishes unavailable groups from true zero error.
                    diagnostics.update({f'{group_name}_{name}': 0. for name in (
                        'td_error_signed_mean', 'td_error_abs_mean', 'q_mean', 'target_mean', 'td_huber_mean')})
            # Empty/receiving-server counts vary; both sets receive equal total weight.
            groups = [error[torch.as_tensor(group)].mean() for group in (real, ~real) if group.any()]
            td = torch.stack(groups).mean()
            aux, recon, pred = self.online.auxiliary(tr.obs.global_, tr.next_obs.global_, tr.done, generator)
            loss = td + self.config['auxiliary_weight'] * aux
            (loss / len(transitions)).backward()
            result.append(dict(loss=float(loss.detach()), td=float(td.detach()), auxiliary=float(aux.detach()),
                reconstruction=float(recon.detach()), prediction=float(pred.detach()),
                q_mean=float(chosen.detach().mean()), q_abs_max=float(chosen.detach().abs().max()),
                target_mean=float(label.mean()), target_abs_max=float(label.abs().max()),
                local_reward_mean=float(tr.reward.mean()), local_reward_max=float(tr.reward.max()),
                reward_sum=float(tr.reward.sum()), receiving_servers=int(real.sum()), **diagnostics))
        norm = nn.utils.clip_grad_norm_(self.online.parameters(), self.config['gradient_clip'], error_if_nonfinite=True)
        self.optimizer.step()
        with torch.no_grad():
            for target, source in zip(self.target.parameters(), self.online.parameters()):
                target.lerp_(source, self.config['target_tau'])
        self.updates += 1
        return dict({k: float(np.mean([r[k] for r in result])) for k in result[0]},
                    gradient_norm=float(norm), gradient_preclip_norm=float(norm),
                    gradient_was_clipped=bool(norm > self.config['gradient_clip']), updates=self.updates)

    def checkpoint(self):
        return dict(online=self.online.state_dict(), target=self.target.state_dict(),
                    optimizer=self.optimizer.state_dict(), updates=self.updates, config=self.config)
