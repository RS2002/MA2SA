"""Exact pretrained actor bridge and independent complete-matching critics.

No extra all-entity Actor-BERT is inserted into the actor. Its previously
trained causal global Transformer and square-L2 QK arithmetic are retained.
The critics additionally see a pooled summary of every public pending task,
including those absent from the chosen matching. No patience sidecar is read.

API:
    actor = CentralActor()
    actor.policy.load_state_dict(parent_checkpoint['online'], strict=True)
    raw = actor(obs)                 # Tensor, compatible with existing evaluate
    output = actor.details(obs)      # CentralOutput, same raw scores
    q1, q2 = critics(output.detach(), cols, obs.legal)

``legal`` is jobs x servers and used only to validate the proposed action.
All maximum-cardinality behavior remains enforced by the common decoder.
The trainer, rather than this module, controls actor-feature detachment,
reward scaling, optimizers, target updates and all scientific execution.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .model import QNetwork, DDQN, Observation, Transition, encode, decode
from .qk import qk_scores


@dataclass
class CentralOutput:
    raw: torch.Tensor       # [servers, pending_tasks + null]
    machine: torch.Tensor   # [servers, 64], before the pretrained query head
    task: torch.Tensor      # [pending_tasks + null, 64], before key head
    global_: torch.Tensor   # [4, 64], pretrained causal global tokens

    def detach(self):
        return CentralOutput(*(getattr(self, name).detach()
                               for name in ('raw', 'machine', 'task', 'global_')))


class CentralActor(nn.Module):
    def __init__(self, policy=None):
        super().__init__()
        self.policy = QNetwork(dim=64, rank=32) if policy is None else policy
        if self.policy.query.in_features != 64:
            raise ValueError('This frozen adaptation requires 64-dimensional pretrained entities')

    def details(self, obs):
        # Keep the exact operation order in the frozen QNetwork.entities.
        mf, tf, running, gf = [torch.as_tensor(x, dtype=torch.float32) for x in
                              (obs.machine, obs.task, obs.running, obs.global_)]
        owner = torch.as_tensor(obs.owner, dtype=torch.long)
        global_tokens = self.policy.global_tokens(gf)
        context = (global_tokens * self.policy.global_pool(global_tokens).softmax(0)).sum(0, keepdim=True)
        machine = self.policy.machine_combine(torch.cat((self.policy.machine(mf),
            self.policy.running(running, owner, len(mf)), context.expand(len(mf), -1)), dim=1))
        tasks = torch.cat((self.policy.task(tf), self.policy.null), dim=0)
        tasks = self.policy.task_combine(torch.cat((tasks, context.expand(len(tasks), -1)), dim=1))
        raw = qk_scores(self.policy.query(machine), self.policy.key(tasks), 'square_l2')
        return CentralOutput(raw, machine, tasks, global_tokens)

    def forward(self, obs):
        return self.details(obs).raw


def _columns(output, cols, legal):
    n, columns = output.raw.shape
    m = columns - 1
    if output.machine.shape != (n, 64) or output.task.shape != (columns, 64) or output.global_.shape != (4, 64):
        raise ValueError('Invalid public actor feature layout')
    if any(not torch.isfinite(value).all() for value in
           (output.raw, output.machine, output.task, output.global_)):
        raise FloatingPointError('Nonfinite public actor features')
    indices = torch.as_tensor(cols, device=output.machine.device)
    if indices.ndim != 1 or indices.shape != (n,) or indices.dtype == torch.bool or indices.is_floating_point() or indices.is_complex():
        raise ValueError('Matching columns must be one integer per server')
    indices = indices.to(torch.long)
    if torch.any(indices < 0) or torch.any(indices >= columns):
        raise ValueError('Matching column out of range')
    real = indices < m
    if indices[real].numel() != torch.unique(indices[real]).numel():
        raise ValueError('One pending job was matched more than once')
    mask = torch.as_tensor(legal, device=indices.device)
    if mask.dtype != torch.bool or mask.shape != (m, n):
        raise ValueError('Capacity mask must be boolean [pending_jobs, servers]')
    rows = torch.arange(n, device=indices.device)
    if real.any() and not mask[indices[real], rows[real]].all():
        raise ValueError('Illegal capacity edge supplied to matching critic')
    return indices


class MatchingCritic(nn.Module):
    """Position-free scalar Q for an entire server-to-task matching."""
    def __init__(self):
        super().__init__()
        self.pair_projection = nn.Linear(128, 64)
        self.pending_attention = nn.Linear(64, 4)
        self.cls = nn.Parameter(torch.randn(1, 64) * .02)
        layer = nn.TransformerEncoderLayer(64, 4, 256, dropout=0., activation='gelu',
                                          batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.attention = nn.Linear(64, 4)
        self.head = nn.Linear(256, 1)

    def forward(self, output, cols, legal):
        indices = _columns(output, cols, legal)
        pair = self.pair_projection(torch.cat((output.machine, output.task[indices]), dim=-1))
        pending = output.task[:-1]
        if len(pending):
            weights = self.pending_attention(pending).softmax(dim=0)
            pending_summary = weights.transpose(0, 1) @ pending
        else:
            pending_summary = pair.new_zeros((4, 64))
        tokens = torch.cat((self.cls, pair, pending_summary, output.global_), dim=0)
        encoded = self.encoder(tokens[None])[0]
        weights = self.attention(encoded).softmax(dim=0)
        pooled = (weights.transpose(0, 1) @ encoded).reshape(-1)
        return self.head(pooled).squeeze(-1)


class TwinMatchingCritic(nn.Module):
    """Two critics with no shared parameters, including their encoders."""
    def __init__(self):
        super().__init__()
        self.q1, self.q2 = MatchingCritic(), MatchingCritic()

    def forward(self, output, cols, legal):
        return self.q1(output, cols, legal), self.q2(output, cols, legal)
