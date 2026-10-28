"""Conservative centralized fine-tuning with detached critic features.

No runner, environment interaction, automatic evaluation selection or rollback
is implemented here. The actor starts from its supplied pretrained weights.
K8 centering is a score surrogate, not an exact Gaussian policy gradient.
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np
import torch
from torch.nn import functional as F
from .central_model import decode


DEFAULTS = dict(actor_lr=2e-6, critic_lr=1e-4, gamma=.99, tau=.005,
    gradient_clip=1., team_reward_scale=1000., policy_delay=2,
    target_noise=.05, noise_clip=.1, baseline_samples=8,
    collection_noise=.05, anchor_lambda=1., regularizer_temperature=.1)


def tensor_hash(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def decode_scores(obs, scores):
    """Same exact-cardinality raw-minus-null objective as pretraining."""
    columns, _ = decode(obs, scores, allow_idle=False)
    return columns


def matching_score(raw, columns):
    columns = np.asarray(columns, dtype=np.int64)
    real = columns < raw.shape[1] - 1
    rows = torch.as_tensor(np.flatnonzero(real), dtype=torch.long, device=raw.device)
    selected = torch.as_tensor(columns[real], dtype=torch.long, device=raw.device)
    if not len(rows):
        return raw.sum() * 0.
    return (raw[rows, selected] - raw[rows, -1]).sum()


def scalar_q(value):
    assert value.numel() == 1, f"expected scalar team Q, got {tuple(value.shape)}"
    result = value.reshape(())
    assert torch.isfinite(result)
    return result


class CentralTrainer:
    """Actor, twin matching critic and frozen pretraining teacher.

    API: warmup_mc_update(batch), finish_warmup(), then step(batch, online_obs).
    Transitions are duck typed: obs/cols/reward/next_obs/done/mc_return.
    `mc_return` is the unscaled discounted *global* behavior return in USD.
    """

    def __init__(self, actor, critics, config=None, seed=0):
        self.config = {**DEFAULTS, **(config or {})}
        assert self.config["baseline_samples"] == 8 and self.config["policy_delay"] == 2
        assert self.config["team_reward_scale"] > 0
        assert self.config["regularizer_temperature"] > 0
        self.actor = actor.eval()
        self.critics = critics.eval()
        self.target_actor = copy.deepcopy(actor).eval().requires_grad_(False)
        self.target_critics = copy.deepcopy(critics).eval().requires_grad_(False)
        self.teacher = copy.deepcopy(actor).eval().requires_grad_(False)
        actor_parameters = list(actor.parameters())
        critic_parameters = list(critics.parameters())
        assert actor_parameters and critic_parameters
        assert not {id(p) for p in actor_parameters} & {id(p) for p in critic_parameters}
        assert all(p.requires_grad for p in actor_parameters + critic_parameters)
        self.actor_optimizer = torch.optim.Adam(actor_parameters, lr=self.config["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(critic_parameters, lr=self.config["critic_lr"])
        self.actor_parameters, self.critic_parameters = actor_parameters, critic_parameters
        self.teacher_hash = tensor_hash(self.teacher)
        self.initial_actor_hash = tensor_hash(actor)
        assert self.teacher_hash == self.initial_actor_hash == tensor_hash(self.target_actor)
        self.initial_target_critic_hash = tensor_hash(self.target_critics)
        self.main_rng = np.random.default_rng(61000 + seed)
        self.baseline_rng = np.random.default_rng(71000 + seed)
        self.target_rng = np.random.default_rng(81000 + seed)
        self.seed = int(seed)
        self.phase = "warmup"
        self.warmup_updates = self.critic_updates = self.actor_updates = self.target_updates = 0
        self.baseline_matching_calls = self.baseline_critic_calls = 0

    def noisy_columns(self, obs, raw, rng, standard_deviation, clip=None):
        values = raw.detach().cpu().numpy().astype(np.float64, copy=True)
        if standard_deviation:
            noise = rng.normal(0., standard_deviation, values.shape)
            if clip is not None:
                noise = np.clip(noise, -clip, clip)
            values += noise
        return decode_scores(obs, values)

    def teacher_row_kl(self, obs, output):
        with torch.no_grad():
            teacher = self.teacher.details(obs).raw
        current = output.raw
        assert current.shape == teacher.shape
        legal = torch.as_tensor(obs.legal.T, dtype=torch.bool, device=current.device)
        legal = torch.cat((legal, torch.ones((len(legal), 1), dtype=torch.bool, device=current.device)), dim=1)
        temperature = self.config["regularizer_temperature"]
        assert torch.isfinite(current[legal]).all() and torch.isfinite(teacher[legal]).all()
        current_log = F.log_softmax((current / temperature).masked_fill(~legal, -torch.inf), dim=1)
        teacher_log = F.log_softmax((teacher / temperature).masked_fill(~legal, -torch.inf), dim=1)
        safe_current = torch.where(legal, current_log, torch.zeros_like(current_log))
        safe_teacher = torch.where(legal, teacher_log, torch.zeros_like(teacher_log))
        probabilities = torch.where(legal, safe_teacher.exp(), torch.zeros_like(safe_teacher))
        result = (probabilities * (safe_teacher - safe_current)).sum(dim=1).mean()
        assert torch.isfinite(result)
        return result

    def _frozen_output(self, obs):
        with torch.no_grad():
            return self.actor.details(obs).detach()

    def _critic_fit(self, batch, mc):
        assert batch
        self.actor.eval()
        self.critics.eval()
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        records = []
        for tr in batch:
            output = self._frozen_output(tr.obs)
            q1, q2 = self.critics(output, tr.cols, tr.obs.legal)
            q1, q2 = scalar_q(q1), scalar_q(q2)
            with torch.no_grad():
                if mc:
                    assert tr.mc_return is not None and np.isfinite(tr.mc_return)
                    target = q1.new_tensor(float(tr.mc_return) / self.config["team_reward_scale"])
                else:
                    reward = float(np.asarray(tr.reward, dtype=np.float64).sum())
                    target = q1.new_tensor(reward / self.config["team_reward_scale"])
                    if not tr.done:
                        next_output = self.target_actor.details(tr.next_obs)
                        cols = self.noisy_columns(tr.next_obs, next_output.raw, self.target_rng,
                            self.config["target_noise"], self.config["noise_clip"])
                        tq1, tq2 = self.target_critics(next_output.detach(), cols, tr.next_obs.legal)
                        target = target + self.config["gamma"] * torch.minimum(scalar_q(tq1), scalar_q(tq2))
            loss = F.smooth_l1_loss(q1, target) + F.smooth_l1_loss(q2, target)
            assert torch.isfinite(loss) and torch.isfinite(target)
            (loss / len(batch)).backward()
            records.append(dict(critic_loss=float(loss.detach()), q1=float(q1.detach()), q2=float(q2.detach()),
                target=float(target), q1_td_error=float(target - q1.detach()), q2_td_error=float(target - q2.detach()),
                terminal=float(bool(tr.done))))
        assert all(p.grad is None for p in self.actor_parameters), "critic update leaked to actor features"
        norm = torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.config["gradient_clip"], error_if_nonfinite=True)
        self.critic_optimizer.step()
        if mc:
            self.warmup_updates += 1
        else:
            self.critic_updates += 1
        result = {key: float(np.mean([r[key] for r in records])) for key in records[0]}
        result.update(critic_gradient_preclip_norm=float(norm), warmup_updates=self.warmup_updates,
            critic_updates=self.critic_updates, actor_updates=self.actor_updates, target_updates=self.target_updates,
            phase=self.phase, critic_features_detached=True)
        return result

    def warmup_mc_update(self, batch):
        assert self.phase == "warmup"
        result = self._critic_fit(batch, mc=True)
        # These hashes are checked at the freeze boundary, not every batch.
        return result

    def finish_warmup(self):
        assert self.phase == "warmup" and self.warmup_updates > 0
        assert tensor_hash(self.actor) == self.initial_actor_hash
        assert tensor_hash(self.target_actor) == self.initial_actor_hash
        assert tensor_hash(self.teacher) == self.teacher_hash
        assert tensor_hash(self.target_critics) == self.initial_target_critic_hash
        self.target_critics.load_state_dict(self.critics.state_dict())
        self.phase = "joint"
        return dict(phase=self.phase, warmup_updates=self.warmup_updates,
            actor_target_teacher_exactly_frozen=True, target_critic_hardcopy_at_warmup_boundary=True)

    def critic_update(self, batch):
        assert self.phase == "joint"
        return self._critic_fit(batch, mc=False)

    def actor_update(self, obs):
        assert self.phase == "joint"
        self.actor.eval()
        self.critics.eval()
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        output = self.actor.details(obs)
        with torch.no_grad():
            main = self.noisy_columns(obs, output.raw, self.main_rng, self.config["collection_noise"])
            main_q, _ = self.critics(output.detach(), main, obs.legal)
            main_value = float(scalar_q(main_q).double())
            baseline_values, cache, signatures = [], {}, []
            main_signature = tuple(map(int, main))
            cache[main_signature] = main_value
            critic_calls = 0
            for _ in range(self.config["baseline_samples"]):
                columns = self.noisy_columns(obs, output.raw, self.baseline_rng, self.config["collection_noise"])
                signature = tuple(map(int, columns))
                signatures.append(signature)
                if signature not in cache:
                    value, _ = self.critics(output.detach(), columns, obs.legal)
                    cache[signature] = float(scalar_q(value).double())
                    critic_calls += 1
                baseline_values.append(cache[signature])
            # Center in Python double; actor and baseline Q share no gradient.
            baseline = float(np.mean(np.asarray(baseline_values, dtype=np.float64)))
            advantage = main_value - baseline
        score = matching_score(output.raw, main)
        surrogate = -advantage * score
        row_kl = self.teacher_row_kl(obs, output)
        loss = surrogate + self.config["anchor_lambda"] * row_kl
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None for p in self.critic_parameters), "actor gradient leaked into critic"
        norm = torch.nn.utils.clip_grad_norm_(self.actor_parameters, self.config["gradient_clip"], error_if_nonfinite=True)
        self.actor_optimizer.step()
        self.actor_updates += 1
        with torch.no_grad():
            for target_module, online_module in ((self.target_actor, self.actor), (self.target_critics, self.critics)):
                for target, online in zip(target_module.parameters(), online_module.parameters()):
                    target.lerp_(online, self.config["tau"])
        self.target_updates += 1
        self.baseline_matching_calls += self.config["baseline_samples"]
        self.baseline_critic_calls += critic_calls
        return dict(actor_loss=float(loss.detach()), actor_surrogate_loss=float(surrogate.detach()),
            actor_teacher_row_kl=float(row_kl.detach()), actor_advantage=advantage, actor_main_q=main_value,
            actor_state_baseline=baseline, actor_matching_score=float(score.detach()),
            actor_gradient_preclip_norm=float(norm), actor_updates=self.actor_updates, target_updates=self.target_updates,
            unique_sampled_matchings=len(set(signatures + [main_signature])),
            baseline_unique_sampled_matchings=len(set(signatures)), baseline_extra_critic_calls=critic_calls,
            baseline_matching_calls=self.baseline_matching_calls, baseline_critic_calls=self.baseline_critic_calls,
            anchor_lambda=self.config["anchor_lambda"])

    def step(self, batch, online_obs):
        result = self.critic_update(batch)
        result["actor_updated"] = self.critic_updates % self.config["policy_delay"] == 0
        if result["actor_updated"]:
            result.update(self.actor_update(online_obs))
        return result

    def state_dict(self):
        assert tensor_hash(self.teacher) == self.teacher_hash
        return dict(schema="central_patience_detached_critic_k8_v1", config=copy.deepcopy(self.config), seed=self.seed,
            actor=self.actor.state_dict(), critics=self.critics.state_dict(), target_actor=self.target_actor.state_dict(),
            target_critics=self.target_critics.state_dict(), teacher=self.teacher.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(), critic_optimizer=self.critic_optimizer.state_dict(),
            initial_actor_hash=self.initial_actor_hash, teacher_hash=self.teacher_hash,
            initial_target_critic_hash=self.initial_target_critic_hash, phase=self.phase,
            warmup_updates=self.warmup_updates, critic_updates=self.critic_updates, actor_updates=self.actor_updates,
            target_updates=self.target_updates, baseline_matching_calls=self.baseline_matching_calls,
            baseline_critic_calls=self.baseline_critic_calls, main_rng=copy.deepcopy(self.main_rng.bit_generator.state),
            baseline_rng=copy.deepcopy(self.baseline_rng.bit_generator.state), target_rng=copy.deepcopy(self.target_rng.bit_generator.state))

    def load_state_dict(self, state):
        assert state["schema"] == "central_patience_detached_critic_k8_v1" and state["config"] == self.config
        for name in ("actor", "critics", "target_actor", "target_critics", "teacher"):
            getattr(self, name).load_state_dict(state[name])
        for name in ("actor_optimizer", "critic_optimizer"):
            getattr(self, name).load_state_dict(state[name])
        for name in ("seed", "initial_actor_hash", "teacher_hash", "initial_target_critic_hash", "phase", "warmup_updates",
                     "critic_updates", "actor_updates", "target_updates", "baseline_matching_calls", "baseline_critic_calls"):
            setattr(self, name, state[name])
        for name in ("main_rng", "baseline_rng", "target_rng"):
            getattr(self, name).bit_generator.state = copy.deepcopy(state[name])
        assert tensor_hash(self.teacher) == self.teacher_hash

    def engineering_actor_invariance(self, obs):
        """No optimizer step: frozen action/score checks Q-zero-point cancellation.

        Artificial scalar credits isolate score/centering arithmetic; this does
        not validate learned critic ranking or actual long-term policy gain.
        """
        parent_hash, state = tensor_hash(self.actor), copy.deepcopy(self.main_rng.bit_generator.state)
        output = self.actor.details(obs)
        columns = decode_scores(obs, output.raw)
        score = matching_score(output.raw, columns)
        # Binary fractions keep the prescribed +10 shift exact in double.
        main = np.float64(1.25)
        values = np.asarray([.5, .75, 1., 1.25, 1.5, 1.75, 2., 2.25], dtype=np.float64)
        advantage = float(main - values.mean())
        shifted = float(main + 10. - (values + 10.).mean())
        constant = float(np.float64(3.) - np.full(8, 3., dtype=np.float64).mean())
        gradients = []
        for coefficient in (advantage, shifted, constant):
            pieces = torch.autograd.grad(-coefficient * score, self.actor_parameters, retain_graph=True, allow_unused=True)
            gradients.append(torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                                       for p, g in zip(self.actor_parameters, pieces)]).detach())
        difference = float((gradients[0] - gradients[1]).norm())
        zero = float(gradients[2].norm())
        assert difference == 0. and zero == 0.
        assert tensor_hash(self.actor) == parent_hash and self.main_rng.bit_generator.state == state
        return dict(status="PASS_ENGINEERING_ONLY", q_plus10_gradient_difference_norm=difference,
            constant_q_gradient_norm=zero, actor_parameters_unchanged=True, optimizer_updates=0,
            limits="artificial fixed credits, no learned-ranking or policy-return guarantee")


# Compatible descriptive alias for callers.
CentralFineTuner = CentralTrainer
