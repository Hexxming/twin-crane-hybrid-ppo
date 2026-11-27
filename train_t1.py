"""Hybrid PPO (T1), manuscript configuration. Requires torch, numpy and gymnasium.
Train: python train_t1.py --train-instances TRAIN_DIR --test-instances TEST_DIR
Evaluate: python train_t1.py --evaluate runs/seed42/final.pt --test-instances TEST_DIR
Each directory contains one <seed>.json per instance. For zero-based episode i the
training file ID is 1000000000 + 20000 * training_seed + i. Test IDs start at 180000.
See YardSimulator._load_instance for the top-first stack format.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from yard_simulator import YardSimulator


@dataclass(frozen=True)
class PPOConfig:
    episodes: int = 10_000
    lanes: int = 4
    episodes_per_update: int = 28
    batch_size: int = 250
    epochs: int = 5
    lr: float = 1e-4
    gamma: float = 1.
    gae_lambda: float = 0.98
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.
    max_grad_norm: float = 0.5


def row_encoder(width):
    return nn.Sequential(nn.Linear(width, 128), nn.LayerNorm(128), nn.GELU(),
                         nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU())


class T1(nn.Module):
    def __init__(self):
        super().__init__()
        self.source = row_encoder(16)
        self.slot = row_encoder(8)
        self.quality = row_encoder(7)
        self.types = nn.Embedding(2, 128)
        self.slot_ids = nn.Embedding(6, 128)
        layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=0., activation="gelu",
                                           norm_first=True, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.actor = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 32),
                                   nn.GELU(), nn.Linear(32, 1))
        self.query = nn.Parameter(torch.zeros(1, 1, 128))
        self.pool = nn.MultiheadAttention(128, 4, batch_first=True)
        self.critic = nn.Sequential(nn.Linear(384, 256), nn.GELU(), nn.Linear(256, 128),
                                    nn.GELU(), nn.Linear(128, 1))
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.orthogonal_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.orthogonal_(m.in_proj_weight)
                nn.init.zeros_(m.in_proj_bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=.01)

    def forward(self, obs):
        sources = self.source(obs["source"]) + self.types.weight[0]
        slots = self.slot(obs["handover"]) + self.types.weight[1] + self.slot_ids.weight
        tokens = torch.cat((sources, slots), dim=1) + self.quality(obs["quality"])
        valid = torch.cat((obs["source_mask"].bool(),
                           torch.ones_like(slots[:, :, 0], dtype=torch.bool)), dim=1)
        tokens = self.transformer(tokens, src_key_padding_mask=~valid)
        mean = (tokens * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
        peak = tokens.masked_fill(~valid.unsqueeze(-1), -torch.inf).amax(1)
        pooled, _ = self.pool(self.query.expand(len(tokens), -1, -1), tokens, tokens,
                              key_padding_mask=~valid, need_weights=False)
        value = self.critic(torch.cat((mean, peak, pooled[:, 0]), dim=-1)).squeeze(-1)
        logits = self.actor(tokens).squeeze(-1).masked_fill(~obs["action_mask"].bool(), -torch.inf)
        return logits, value


def tensors(observations, device):
    return {k: torch.as_tensor(np.stack([o[k] for o in observations]), device=device)
            for k in observations[0]}


@torch.no_grad()
def collect(policy, envs, start, stop, seed, config, record, device, train_instances):
    """Collect exactly stop-start complete trajectories, one independent instance per episode."""
    live = {}
    next_episode = start
    samples = []

    def refill(lane):
        nonlocal next_episode
        if next_episode < stop:
            index = next_episode
            next_episode += 1
            instance_id = 1_000_000_000 + seed*20_000 + index
            obs, _ = envs[lane].reset(options={"instance": train_instances / f"{instance_id}.json"})
            live[lane] = (index, obs, [])

    for lane in range(len(envs)):
        refill(lane)
    while live:
        lanes = list(live)
        logits, values = policy(tensors([live[l][1] for l in lanes], device))
        dist = Categorical(logits=logits)
        actions = dist.sample()
        logp = dist.log_prob(actions)
        for j, lane in enumerate(lanes):
            index, obs, path = live[lane]
            a = int(actions[j])
            nxt, reward, done, truncated, info = envs[lane].step(a)
            if truncated:
                raise RuntimeError("Unexpected truncated trajectory")
            path.append((obs, a, float(logp[j]), float(values[j]), reward))
            if done:
                ret = sum(t[-1] for t in path)
                if not np.isclose(ret, -info['makespan']):
                    raise RuntimeError("Return does not equal negative makespan")
                record.writerow((index+1, info['makespan'], ret))
                advantage = next_value = 0.
                for o, act, lp, v, r in reversed(path):
                    delta = r + config.gamma*next_value - v
                    advantage = delta + config.gamma*config.gae_lambda*advantage
                    samples.append((o, act, lp, advantage, v+advantage))
                    next_value = v
                del live[lane]
                refill(lane)
            else:
                live[lane] = (index, nxt, path)
    return samples


def update(policy, opt, samples, config, rng, device):
    obs, actions, old_logp, advantages, returns = zip(*samples)
    obs = tensors(obs, device)
    actions = torch.tensor(actions, device=device)
    old_logp, advantages, returns = [torch.tensor(x, device=device, dtype=torch.float32)
                                     for x in (old_logp, advantages, returns)]
    for _ in range(config.epochs):
        indices = rng.permutation(len(samples))
        for start in range(0, len(indices), config.batch_size):
            idx = torch.as_tensor(indices[start:start+config.batch_size], device=device)
            logits, prediction = policy({k: v[idx] for k,v in obs.items()})
            dist = Categorical(logits=logits)
            ratio = (dist.log_prob(actions[idx]) - old_logp[idx]).exp()
            adv = advantages[idx]
            if len(idx) > 1:
                adv = (adv-adv.mean()) / (adv.std()+1e-8)
            clipped = ratio.clamp(1-config.clip, 1+config.clip)
            objective = torch.minimum(ratio*adv, clipped*adv).mean()
            value_loss = (prediction-returns[idx]).square().mean()
            loss = -objective + config.value_coef*value_loss - config.entropy_coef*dist.entropy().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm, error_if_nonfinite=True)
            opt.step()


def train(seed, config, output, device, train_instances):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    policy = T1().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr, eps=1e-5)
    envs = [YardSimulator() for _ in range(config.lanes)]
    run = output / f"seed{seed}"
    run.mkdir(parents=True, exist_ok=False)
    with (run/"training.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("episode", "makespan", "return"))
        for start in range(0, config.episodes, config.episodes_per_update):
            stop = min(start+config.episodes_per_update, config.episodes)
            data = collect(policy, envs, start, stop, seed, config, writer, device, train_instances)
            update(policy, optimizer, data, config, rng, device)
            f.flush()
    checkpoint = run/"final.pt"
    torch.save(dict(model=policy.state_dict(), config=asdict(config), seed=seed,
                    completed_episodes=config.episodes), checkpoint)
    return checkpoint


@torch.inference_mode()
def evaluate(checkpoint, count, K, H, rho, capacity, test_instances):
    policy = T1()
    policy.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
    policy.eval()
    env = YardSimulator(K, H, rho, capacity)
    makespans = []
    for seed in range(180000, 180000+count):
        obs, _ = env.reset(options={"instance": test_instances / f"{seed}.json"})
        while True:
            logits, _ = policy(tensors([obs], "cpu"))
            obs, _, done, _, info = env.step(int(logits.argmax(1).item()))
            if done:
                makespans.append(info["makespan"])
                break
    print(f"{checkpoint} | {count} instances | mean makespan {np.mean(makespans):.2f}", flush=True)
    return makespans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[24, 42, 43])
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--evaluate", type=Path, help="Evaluate a final checkpoint on CPU")
    parser.add_argument("--test-count", type=int, default=200)
    parser.add_argument("--K", type=int, default=16, help="Test active source count")
    parser.add_argument("--H", type=int, default=5, help="Test initial stack height")
    parser.add_argument("--rho", type=float, default=.5, help="Test cross-region ratio")
    parser.add_argument("--capacity", type=int, default=6, help="Test handover capacity")
    parser.add_argument("--train-instances", type=Path, help="Directory of training instance JSON files")
    parser.add_argument("--test-instances", type=Path, required=True, help="Directory of test instance JSON files")
    args = parser.parse_args()
    if not args.evaluate and args.train_instances is None:
        parser.error("--train-instances is required for training")
    if args.episodes < 1 or args.test_count < 1:
        parser.error("Episode and test counts must be positive")
    torch.set_num_threads(1)
    if args.evaluate:
        evaluate(args.evaluate, args.test_count, args.K, args.H, args.rho, args.capacity, args.test_instances)
    else:
        for seed in args.seeds:
            final = train(seed, PPOConfig(episodes=args.episodes), args.output, args.device, args.train_instances)
            evaluate(final, args.test_count, args.K, args.H, args.rho, args.capacity, args.test_instances)


if __name__ == "__main__":
    main()

