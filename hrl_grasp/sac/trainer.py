import os, time, math, csv, random, json
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from hrl_grasp.runtime_sampler import RuntimeSampler, fixed_eval_seeds
from .replay_buffer import ReplayBuffer
from .networks import Actor, Critic

@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2  # initial entropy coef (auto-tune optional)
    lr: float = 3e-4
    batch_size: int = 256
    replay_size: int = 100_000
    start_steps: int = 2_000
    update_after: int = 1_000
    updates_per_step: float = 1.0
    max_env_steps: int = 200_000
    eval_every: int = 10_000
    log_dir: str = "/home/naren/HRL_part/rlbench_data/sac_logs"
    # Nice-to-haves
    seed: int = 12345
    checkpoint_every: int = 100  # episodes
    checkpoint_dir_name: str = "checkpoints"
    best_dir_name: str = "best"

class SACAgent:
    def __init__(self, obs_dim: int, action_dim: int, cfg: SACConfig):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim, action_dim).to(self.device)
        self.critic_target = Critic(obs_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.lr)
        # entropy coef (fixed here per request)
        self.log_alpha = torch.tensor(math.log(cfg.alpha), device=self.device)
        self.alpha = cfg.alpha
        self.target_entropy = -action_dim

    def act(self, obs: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            if eval_mode:
                mu, _std = self.actor(o)
                a = torch.tanh(mu)
            else:
                a, _logp, _mu_a = self.actor.sample(o)
        a = a.squeeze(0).cpu().numpy()
        # Map [-1,1] action to env’s action bounds: Δpos and grip in [-1,1] scale.
        # Our env expects [dx,dy,dz,grip] with internal clamp; scale to 0.03m.
        scale = np.array([0.03, 0.03, 0.03, 1.0], dtype=np.float32)
        return a * scale

    def update(self, rb: ReplayBuffer, writer: Optional[SummaryWriter], step: int):
        if len(rb) < self.cfg.batch_size:
            return None, None
        obs, act, rew, next_obs, done = rb.sample(self.cfg.batch_size)
        device = self.device
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        act_t = torch.as_tensor(act, dtype=torch.float32, device=device)
        rew_t = torch.as_tensor(rew, dtype=torch.float32, device=device)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
        done_t = torch.as_tensor(done, dtype=torch.float32, device=device)

        with torch.no_grad():
            next_a, next_logp, _ = self.actor.sample(next_obs_t)
            q1_t, q2_t = self.critic_target(next_obs_t, next_a)
            q_t_min = torch.min(q1_t, q2_t)
            target = rew_t + self.cfg.gamma * (1 - done_t) * (q_t_min - self.alpha * next_logp)

        # Critic update
        q1, q2 = self.critic(obs_t, act_t)
        critic_loss = ((q1 - target).pow(2) + (q2 - target).pow(2)).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update
        a, logp, _ = self.actor.sample(obs_t)
        q1_pi, q2_pi = self.critic(obs_t, a)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * logp - q_pi).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # Polyak update
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_targ.data.mul_(1 - self.cfg.tau)
                p_targ.data.add_(self.cfg.tau * p.data)

        if writer is not None:
            writer.add_scalar('loss/critic', critic_loss.item(), step)
            writer.add_scalar('loss/actor', actor_loss.item(), step)
        return float(critic_loss.item()), float(actor_loss.item())


def train_sac():
    cfg = SACConfig()
    os.makedirs(cfg.log_dir, exist_ok=True)
    writer = SummaryWriter(cfg.log_dir)

    # Seeding (global)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    try:
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
    except Exception:
        pass

    env = RuntimeSampler(headless=True, debug_steps=False, log_csv=os.path.join(cfg.log_dir, 'episodes.csv'),
                          save_scene_images=True, save_eval_images=True)
    # Deterministic eval seed list
    try:
        env.eval_seeds = fixed_eval_seeds(200, base_seed=cfg.seed)
    except Exception:
        pass
    # Optional: force curriculum phase as needed with env.set_phase(None)

    # Warm-up to get obs_dim
    env.eval_mode = False
    obs, info = env.reset()
    obs_dim = int(obs.shape[0])
    action_dim = 4

    rb = ReplayBuffer(cfg.replay_size, obs_dim, action_dim)
    agent = SACAgent(obs_dim, action_dim, cfg)

    total_steps = 0
    last_eval = 0
    ep_idx = 0
    best_eval_success = -1.0
    best_eval_return = -1e9

    # Persist seed info
    try:
        with open(os.path.join(cfg.log_dir, 'seeds.json'), 'w') as f:
            json.dump({"seed": cfg.seed}, f)
    except Exception:
        pass
    writer.add_text('meta/seed', str(cfg.seed))

    # Checkpoint helpers
    ckpt_dir = os.path.join(cfg.log_dir, cfg.checkpoint_dir_name)
    best_dir = os.path.join(cfg.log_dir, cfg.best_dir_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    def save_checkpoint(tag: str, step: int, ep: int):
        path = os.path.join(ckpt_dir, f"{tag}_step{step}_ep{ep}.pt")
        payload = {
            'actor': agent.actor.state_dict(),
            'critic': agent.critic.state_dict(),
            'critic_target': agent.critic_target.state_dict(),
            'actor_opt': agent.actor_opt.state_dict(),
            'critic_opt': agent.critic_opt.state_dict(),
            'step': step, 'episode': ep, 'cfg': vars(cfg), 'obs_dim': obs_dim, 'action_dim': action_dim,
        }
        try:
            torch.save(payload, path)
            print(f"CHECKPOINT saved -> {path}")
        except Exception as e:
            print(f"[WARN] checkpoint save failed: {e}")

    def save_best(step: int, ep: int):
        path = os.path.join(best_dir, f"best_step{step}_ep{ep}.pt")
        payload = {
            'actor': agent.actor.state_dict(),
            'critic': agent.critic.state_dict(),
            'critic_target': agent.critic_target.state_dict(),
            'actor_opt': agent.actor_opt.state_dict(),
            'critic_opt': agent.critic_opt.state_dict(),
            'step': step, 'episode': ep, 'cfg': vars(cfg), 'obs_dim': obs_dim, 'action_dim': action_dim,
        }
        try:
            torch.save(payload, path)
            print(f"BEST saved -> {path}")
        except Exception as e:
            print(f"[WARN] best save failed: {e}")

    try:
        while total_steps < cfg.max_env_steps:
            # Start a new sub-episode rollout
            pid = getattr(env, 'curr_target_pid', None)
            cat = None
            try:
                if pid is not None:
                    cat = env.project_map[pid]["category"]
            except Exception:
                cat = None
            print(f"EP START ep={ep_idx} t_total={total_steps} arr={env.arrangement_id} sub={env.subep_idx}/{getattr(env,'_B_current',0)} pid={pid} cat={cat}")

            ep_steps = 0
            ep_return = 0.0
            done = False
            # Rollout up to horizon
            for t in range(env.horizon):
                # Action selection: random until start_steps, then policy
                if total_steps < cfg.start_steps:
                    a = np.random.uniform(low=[-0.03,-0.03,-0.03,-1.0], high=[0.03,0.03,0.03,1.0]).astype(np.float32)
                else:
                    a = agent.act(obs, eval_mode=False)

                next_obs, r, done, info = env.step(a)
                rb.push(obs, a, r, next_obs, done)
                ep_steps += 1
                ep_return += float(r)
                total_steps += 1

                # Learn
                if total_steps > cfg.update_after:
                    for _ in range(int(cfg.updates_per_step)):
                        ql, pl = agent.update(rb, writer, total_steps)
                    if total_steps % 1000 == 0:
                        print(f"TRAIN t_total={total_steps} replay={len(rb)} q_loss={ql} pi_loss={pl}")

                # TB scalars
                writer.add_scalar('env/reward', float(r), total_steps)
                writer.add_scalar('env/success', float(info.get('success', False)), total_steps)

                obs = next_obs
                if done:
                    break

            # End of episode
            if ep_steps == 0:
                print("ZERO_STEP_EP: episode terminated without steps; raising")
                raise RuntimeError("Zero-step episode detected")
            env.finish_episode(bool(info.get('success', False)))
            print(f"EP DONE ep={ep_idx} steps={ep_steps} ret={ep_return:.3f} succ={bool(info.get('success', False))}")
            # Periodic checkpoint
            if (ep_idx % cfg.checkpoint_every) == 0:
                save_checkpoint('periodic', total_steps, ep_idx)
            ep_idx += 1

            # Prepare next episode
            obs, info = env.reset()

            # Periodic eval (one arrangement)
            if total_steps - last_eval >= cfg.eval_every:
                env.eval_mode = True
                obs_eval, _ = env.reset()
                eval_ret_sum = 0.0
                eval_succ_sum = 0.0
                eval_eps = 0
                for _sub in range(getattr(env, '_B_current', 1)):
                    done_e = False
                    ep_ret_e = 0.0
                    while not done_e:
                        a_eval = agent.act(obs_eval, eval_mode=True)
                        obs_eval, r_e, done_e, info_e = env.step(a_eval)
                        ep_ret_e += float(r_e)
                    env.finish_episode(bool(info_e.get('success', False)))
                    eval_eps += 1
                    eval_ret_sum += ep_ret_e
                    eval_succ_sum += float(bool(info_e.get('success', False)))
                    if _sub+1 < getattr(env, '_B_current', 1):
                        obs_eval, _ = env.reset()
                env.eval_mode = False
                last_eval = total_steps
                # Metrics and best save
                avg_ret = (eval_ret_sum / max(1, eval_eps))
                succ_rate = (eval_succ_sum / max(1, eval_eps))
                writer.add_scalar('eval/avg_return', avg_ret, total_steps)
                writer.add_scalar('eval/success_rate', succ_rate, total_steps)
                print(f"EVAL step={total_steps} eps={eval_eps} avg_ret={avg_ret:.3f} succ_rate={succ_rate:.3f}")
                is_best = (succ_rate > best_eval_success) or (succ_rate == best_eval_success and avg_ret > best_eval_return)
                if is_best:
                    best_eval_success = succ_rate
                    best_eval_return = avg_ret
                    save_best(total_steps, ep_idx)
        
    finally:
        env.shutdown()
        writer.flush(); writer.close()

if __name__ == '__main__':
    train_sac()
