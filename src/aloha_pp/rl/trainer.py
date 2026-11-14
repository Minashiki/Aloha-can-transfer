"""Collect chunk transitions and run SAC. Validation uses the val seeds only."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from aloha_pp.planning.metrics import wilson_interval
from aloha_pp.policy.rollout import write_video
from aloha_pp.rl.checkpoint import load_training, save_training, sha256_file
from aloha_pp.rl.noise import clip_standard_normal
from aloha_pp.rl.replay import Replay
from aloha_pp.rl.spec import (
    assert_seed_ranges,
    eval_seed_set,
    privileged_dim,
    val_seed_set,
    SeedStream,
)
from aloha_pp.rl.vec import SerialChunkEnv, SubprocChunkEnv


def assert_m7_output(path: Path, root: Path) -> None:
    resolved = Path(path).resolve()
    outputs = (root / "outputs").resolve()
    try:
        relative = resolved.relative_to(outputs)
    except ValueError:
        return
    if relative.parts and relative.parts[0] in {"m3", "m4", "m5", "m6"}:
        raise SystemExit(f"refusing to write under outputs/{relative.parts[0]}: {resolved}")


def env_setup(cfg: dict) -> dict:
    return {
        "timeout_s": float(cfg["timeout_s"]),
        "staged": cfg["staged_reward"],
        "n_obs": int(cfg["n_obs_steps"]),
        "pred_horizon": int(cfg["pred_horizon"]),
        "exec_horizon": int(cfg["exec_horizon"]),
        "action_dim": int(cfg["action_dim"]),
        "video_fps": int(cfg["video_fps"]),
        "trace": False,
    }


def make_vec(n: int, setup: dict, *, live_viewer: bool):
    if live_viewer and int(n) != 1:
        raise SystemExit("--live-viewer runs a single in-process environment")
    if live_viewer:
        return SerialChunkEnv(int(n), setup, live_viewer=True)
    return SubprocChunkEnv(int(n), setup)


def estimate_seconds(
    *,
    total_env_steps: int,
    n_envs: int,
    vec_step_s: float,
    update_s: float,
    utd: int,
    init_rollout_steps: int,
    val_every: int,
    n_val: int,
    val_rl_steps: int,
) -> dict:
    """Wall clock from a measured vec step and a measured SAC update.

    A validation wave reuses the training env count. ``val_rl_steps`` is the
    assumed length of one validation episode in RL steps.
    """
    vec_steps = float(total_env_steps) / float(n_envs)
    collect_s = vec_steps * float(vec_step_s)
    learning_vec = max(0.0, vec_steps - float(init_rollout_steps))
    update_total = learning_vec * float(utd) * float(update_s)
    n_vals = 0 if val_every <= 0 else int(math.ceil(float(total_env_steps) / float(val_every)))
    waves = int(math.ceil(float(n_val) / float(n_envs)))
    val_s = float(n_vals) * float(waves) * float(val_rl_steps) * float(vec_step_s)
    total = collect_s + update_total + val_s
    return {
        "collect_s": collect_s,
        "update_s": update_total,
        "val_s": val_s,
        "n_vals": n_vals,
        "total_s": total,
        "total_h": total / 3600.0,
    }


def _log(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _sync_device(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _health(stats: dict | None, key: str):
    if stats is None:
        return None
    return stats[key]


def _empty_timing() -> dict:
    return {
        "encode_s": 0.0,
        "collect_s": 0.0,
        "update_s": 0.0,
        "val_s": 0.0,
        "replay_s": 0.0,
        "checkpoint_s": 0.0,
        "n_updates": 0,
        "n_vals": 0,
        "n_checkpoints": 0,
        "warmup_vec_steps": 0,
        "learn_vec_steps": 0,
        "fill_vec_steps": 0,
        "sync": "cuda.synchronize at the boundary of each timed section",
    }


def _validate(
    policy, actor, vec, seeds: list[int], *, device, video_path: Path, video_fps: int, video_max_mb: float, record: bool,
) -> dict:
    shape = tuple(policy.noise_shape)
    n = len(seeds)
    state = vec.reset(seeds, record_index=0 if record else None)
    active = np.ones(n, dtype=bool)
    success = np.zeros(n, dtype=bool)
    video = None
    while bool(active.any()):
        index = np.flatnonzero(active)
        packed = {key: value[index] for key, value in state["obs"].items()}
        with torch.no_grad():
            feat = policy.encode_actor_obs(packed)
            noise = actor.deterministic(feat).view(len(index), *shape)
            chunks = policy.sample(packed, noise=noise).detach().cpu().numpy()
        full = np.zeros((n, *shape), dtype=np.float32)
        full[index] = chunks
        out = vec.step(full, active=active)
        packed_i = 0
        for local in index:
            info = out["info"][packed_i]
            ended = bool(out["terminated"][packed_i] or out["truncated"][packed_i])
            if ended:
                success[local] = bool(out["terminated"][packed_i])
                active[local] = False
                frames = info.get("frames")
                if record and frames is not None and video is None:
                    write_video([frame for frame in frames], video_path, int(video_fps), float(video_max_mb))
                    video = str(video_path)
            else:
                for key in state["obs"]:
                    state["obs"][key][local] = out["obs"][key][packed_i]
            packed_i += 1
    n_success = int(success.sum())
    low, high = wilson_interval(n_success, n)
    return {
        "n": n,
        "n_success": n_success,
        "success_rate": float(n_success) / float(n),
        "wilson_95": [low, high],
        "seeds": [int(s) for s in seeds],
        "video": video,
    }


def benchmark_collection(policy, cfg: dict, counts: list[int], n_steps: int, device) -> list[dict]:
    """RL transitions per second at each env count. Noise is zero; SAC does not update."""
    setup = env_setup(cfg)
    shape = tuple(policy.noise_shape)
    banned = eval_seed_set(int(cfg["eval_seed_start"]), int(cfg["eval_seed_count"]))
    banned |= val_seed_set(int(cfg["val_seed_start"]), int(cfg["n_val_episodes"]))
    rows = []
    for n in counts:
        seeds = [30000 + i for i in range(int(n))]
        if set(seeds) & banned:
            raise RuntimeError("benchmark seeds collide with val or eval")
        vec = make_vec(int(n), setup, live_viewer=False)
        try:
            state = vec.reset(seeds)
            started = time.perf_counter()
            for _ in range(int(n_steps)):
                with torch.no_grad():
                    noise = torch.zeros((int(n), *shape), device=device)
                    chunks = policy.sample(state["obs"], noise=noise).detach().cpu().numpy()
                out = vec.step(chunks)
                done = out["terminated"] | out["truncated"]
                obs = out["obs"]
                priv = out["priv"]
                if bool(done.any()):
                    done_index = [int(i) for i in np.flatnonzero(done)]
                    fresh_seeds = [30000 + 1000 + i for i in done_index]
                    reset = vec.reset(fresh_seeds, indices=done_index)
                    for column, index in enumerate(done_index):
                        for key in obs:
                            obs[key][index] = reset["obs"][key][column]
                        priv[index] = reset["priv"][column]
                state = {"obs": obs, "priv": priv}
            elapsed = time.perf_counter() - started
        finally:
            vec.close()
        transitions = int(n) * int(n_steps)
        rows.append({
            "n_envs": int(n),
            "vec_steps": int(n_steps),
            "transitions": transitions,
            "seconds": elapsed,
            "transitions_per_s": transitions / elapsed,
            "vec_step_s": elapsed / float(n_steps),
        })
        print(
            f"benchmark n={n} transitions/s {rows[-1]['transitions_per_s']:.2f} "
            f"vec_step {rows[-1]['vec_step_s']:.3f}s",
            flush=True,
        )
    return rows


def benchmark_updates(agent, replay: Replay, batch: int, n_updates: int, device) -> dict:
    rng = np.random.default_rng(1)
    n = max(int(batch), 64)
    replay.add(
        rng.standard_normal((n, replay.feat_dim)).astype(np.float32),
        rng.standard_normal((n, replay.priv_dim)).astype(np.float32),
        rng.standard_normal((n, replay.action_dim)).astype(np.float32),
        rng.standard_normal(n).astype(np.float32),
        rng.standard_normal((n, replay.feat_dim)).astype(np.float32),
        rng.standard_normal((n, replay.priv_dim)).astype(np.float32),
        np.zeros(n, np.float32),
        np.zeros(n, np.float32),
    )
    # One untimed update compiles kernels.
    agent.update(replay.sample(int(batch), device))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(int(n_updates)):
        agent.update(replay.sample(int(batch), device))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {"n_updates": int(n_updates), "seconds": elapsed, "update_s": elapsed / float(n_updates), "batch": int(batch)}


def _save(path: Path, policy_actor, agent, cfg, extra, replay: Replay | None) -> None:
    save_training(path, actor=policy_actor, agent=agent, cfg=cfg, extra=extra)
    if replay is not None:
        replay.save(path.with_name("replay.pt"))


def train(cfg: dict, *, policy, actor, agent, out_dir: Path, device, live_viewer: bool, resumed: dict | None) -> dict:
    policy.eval()
    policy.requires_grad_(False)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    n_envs = int(cfg["n_envs"])
    total = int(cfg["total_env_steps"])
    init_steps = int(cfg["init_rollout_steps"])
    batch = int(cfg["sac"]["batch_size"])
    utd = int(cfg["sac"]["utd"])
    val_every = int(cfg["val_every"])
    n_val = int(cfg["n_val_episodes"])
    evaluation = eval_seed_set(int(cfg["eval_seed_start"]), int(cfg["eval_seed_count"]))
    validation = val_seed_set(int(cfg["val_seed_start"]), n_val)
    assert_seed_ranges(
        int(cfg["train_seed_start"]),
        int(cfg["val_seed_start"]),
        n_val,
        int(cfg["eval_seed_start"]),
        int(cfg["eval_seed_count"]),
    )
    if evaluation & validation:
        raise RuntimeError("eval and val seeds overlap")
    stream = SeedStream(int(cfg["train_seed_start"]), evaluation | validation)
    warmup_rng = np.random.default_rng(int(cfg["seed"]))
    env_steps = 0
    vec_steps = 0
    episodes_done = 0
    episode_successes = 0
    best_val = -1.0
    best_val_step = -1
    if resumed is not None:
        env_steps = int(resumed["env_steps"])
        vec_steps = int(resumed["vec_steps"])
        episodes_done = int(resumed.get("episodes", 0))
        episode_successes = int(resumed.get("episode_successes", 0))
        best_val = float(resumed.get("best_val_success", -1.0))
        best_val_step = int(resumed.get("best_val_step", -1))
        stream.next = int(resumed["seed_stream_next"])
        warmup_rng.bit_generator.state = resumed["warmup_rng"]
        print(
            f"resumed env_steps {env_steps} vec_steps {vec_steps} replay_will_load {cfg.get('save_replay')}",
            flush=True,
        )
    feat_dim = int(actor.feat_dim)
    action_dim = int(actor.action_dim)
    replay = Replay(int(cfg["replay_capacity"]), feat_dim, privileged_dim(), action_dim, rng=np.random.default_rng(int(cfg["seed"]) + 1))
    replay_path = out_dir / "replay.pt"
    if resumed is not None and bool(cfg.get("save_replay")) and replay_path.is_file():
        replay = Replay.load(replay_path)
        print(f"loaded replay size {replay.size}", flush=True)
    elif resumed is not None:
        print("resume kept the weights and started an empty replay (save_replay is off or the file is missing)", flush=True)
    setup = env_setup(cfg)
    vec = make_vec(n_envs, setup, live_viewer=live_viewer)
    val_vec = None
    video_vec = None
    last_val_at = -1
    timing = _empty_timing()
    shape = tuple(int(v) for v in policy.noise_shape)
    scale = float(cfg["noise_scale"])
    rate_base = int(env_steps)
    started = time.perf_counter()
    try:
        seeds = stream.take(n_envs)
        state = vec.reset(seeds)
        ep_return = np.zeros(n_envs, dtype=np.float64)
        ep_rl = np.zeros(n_envs, dtype=np.int64)
        ep_ctrl = np.zeros(n_envs, dtype=np.int64)
        ep_seed = np.asarray(seeds, dtype=np.int64)
        while env_steps < total:
            warming = vec_steps < init_steps
            obs = state["obs"]
            priv = np.asarray(state["priv"], dtype=np.float32)
            _sync_device(device)
            t_mark = time.perf_counter()
            with torch.no_grad():
                feat = policy.encode_actor_obs(obs)
            _sync_device(device)
            timing["encode_s"] += time.perf_counter() - t_mark
            _sync_device(device)
            t_mark = time.perf_counter()
            with torch.no_grad():
                if warming:
                    w_np = clip_standard_normal(warmup_rng, (n_envs, action_dim), scale)
                    w = torch.as_tensor(w_np, device=device)
                else:
                    w, _logp = actor.sample(feat)
                    w_np = w.detach().cpu().numpy()
                chunks = policy.sample(obs, noise=w.view(n_envs, *shape)).detach().cpu().numpy()
            out = vec.step(chunks)
            _sync_device(device)
            timing["collect_s"] += time.perf_counter() - t_mark
            _sync_device(device)
            t_mark = time.perf_counter()
            with torch.no_grad():
                next_feat = policy.encode_actor_obs(out["obs"]).detach().cpu().numpy()
            _sync_device(device)
            timing["encode_s"] += time.perf_counter() - t_mark
            t_mark = time.perf_counter()
            replay.add(
                feat.detach().cpu().numpy(),
                priv,
                w_np,
                out["reward"],
                next_feat,
                np.asarray(out["priv"], dtype=np.float32),
                out["terminated"].astype(np.float32),
                out["truncated"].astype(np.float32),
            )
            timing["replay_s"] += time.perf_counter() - t_mark
            env_steps += n_envs
            vec_steps += 1
            done = out["terminated"] | out["truncated"]
            for index in range(n_envs):
                ep_return[index] += float(out["reward"][index])
                ep_rl[index] += 1
                ep_ctrl[index] += int(out["info"][index]["n_control"])
                if not bool(done[index]):
                    continue
                episodes_done += 1
                succeeded = bool(out["terminated"][index])
                episode_successes += int(succeeded)
                _log(log_path, {
                    "event": "episode",
                    "seed": int(ep_seed[index]),
                    "success": succeeded,
                    "truncated": bool(out["truncated"][index]),
                    "terminated": succeeded,
                    "return": float(ep_return[index]),
                    "rl_steps": int(ep_rl[index]),
                    "control_steps": int(ep_ctrl[index]),
                    "env_steps": int(env_steps),
                })
                print(
                    f"episode seed {int(ep_seed[index])} success {succeeded} "
                    f"return {ep_return[index]:.3f} rl_steps {int(ep_rl[index])} "
                    f"control {int(ep_ctrl[index])}",
                    flush=True,
                )
                ep_return[index] = 0.0
                ep_rl[index] = 0
                ep_ctrl[index] = 0
            if bool(done.any()):
                indices = [int(i) for i in np.flatnonzero(done)]
                fresh = stream.take(len(indices))
                reset = vec.reset(fresh, indices=indices)
                for column, index in enumerate(indices):
                    for key in out["obs"]:
                        out["obs"][key][index] = reset["obs"][key][column]
                    out["priv"][index] = reset["priv"][column]
                    ep_seed[index] = int(fresh[column])
            state = {"obs": out["obs"], "priv": out["priv"]}
            stats = None
            if warming:
                timing["warmup_vec_steps"] += 1
            elif replay.size >= batch:
                timing["learn_vec_steps"] += 1
                _sync_device(device)
                t_mark = time.perf_counter()
                for _ in range(utd):
                    stats = agent.update(replay.sample(batch, device))
                    timing["n_updates"] += 1
                _sync_device(device)
                timing["update_s"] += time.perf_counter() - t_mark
                if stats is not None and not math.isfinite(stats["critic_loss"]):
                    raise RuntimeError(f"critic loss is not finite: {stats}")
            else:
                timing["fill_vec_steps"] += 1
            elapsed = time.perf_counter() - started
            if env_steps % int(cfg["log_every"]) < n_envs:
                record = {
                    "event": "train",
                    "env_steps": int(env_steps),
                    "vec_steps": int(vec_steps),
                    "replay": int(replay.size),
                    "episodes": int(episodes_done),
                    "episode_successes": int(episode_successes),
                    "steps_per_s": float(env_steps - rate_base) / elapsed,
                    "warming": bool(warming),
                    "critic_loss": _health(stats, "critic_loss"),
                    "actor_loss": _health(stats, "actor_loss"),
                    "alpha": _health(stats, "alpha"),
                    "logp_mean": _health(stats, "logp_mean"),
                    "q_mean": _health(stats, "q_mean"),
                    "q_max": _health(stats, "q_max"),
                    "target_q_mean": _health(stats, "target_q_mean"),
                    "target_q_max": _health(stats, "target_q_max"),
                    "actor_abs_mean": _health(stats, "actor_abs_mean"),
                }
                _log(log_path, record)
                print(
                    f"steps {env_steps} replay {replay.size} episodes {episodes_done} "
                    f"successes {episode_successes} steps/s {record['steps_per_s']:.2f} "
                    f"critic {record['critic_loss']} actor {record['actor_loss']} alpha {record['alpha']} "
                    f"logp_mean {record['logp_mean']} q_mean {record['q_mean']} q_max {record['q_max']} "
                    f"target_q_mean {record['target_q_mean']} target_q_max {record['target_q_max']} "
                    f"actor_abs_mean {record['actor_abs_mean']}",
                    flush=True,
                )
            end_val = bool(cfg.get("val_at_end", True)) and env_steps >= total
            # Fire once when this env-step jump crosses a multiple. Exact equality
            # would skip the validation whenever n_envs does not divide val_every.
            prev_steps = env_steps - n_envs
            crossed_val = val_every > 0 and (env_steps // val_every) > (prev_steps // val_every)
            at_val = crossed_val or end_val
            if at_val and env_steps != last_val_at:
                if val_vec is None:
                    val_vec = make_vec(min(n_envs, n_val), setup, live_viewer=False)
                val_seeds = list(range(int(cfg["val_seed_start"]), int(cfg["val_seed_start"]) + n_val))
                if set(val_seeds) & evaluation:
                    raise RuntimeError("validation tried to use an eval seed")
                video_path = out_dir / f"val_step{env_steps:08d}" / "videos" / f"seed{int(cfg['val_seed_start']):05d}.mp4"
                actor.eval()
                # The counted episodes never record. One extra episode, on its
                # own environment and the same val_seed_start, writes the video
                # and is left out of the rate and the checkpoint choice.
                _sync_device(device)
                t_mark = time.perf_counter()
                pool = min(n_envs, n_val)
                merged_success = 0
                for offset in range(0, n_val, pool):
                    wave = val_seeds[offset : offset + pool]
                    part = _validate(
                        policy, actor, val_vec, wave,
                        device=device,
                        video_path=video_path,
                        video_fps=int(cfg["video_fps"]),
                        video_max_mb=float(cfg["video_max_mb"]),
                        record=False,
                    )
                    if part["video"] is not None:
                        raise RuntimeError("validation statistics recorded a video")
                    merged_success += int(part["n_success"])
                if video_vec is None:
                    video_vec = make_vec(1, setup, live_viewer=False)
                video_part = _validate(
                    policy, actor, video_vec, [int(cfg["val_seed_start"])],
                    device=device,
                    video_path=video_path,
                    video_fps=int(cfg["video_fps"]),
                    video_max_mb=float(cfg["video_max_mb"]),
                    record=True,
                )
                video = video_part["video"]
                _sync_device(device)
                timing["val_s"] += time.perf_counter() - t_mark
                timing["n_vals"] += 1
                actor.train()
                low, high = wilson_interval(merged_success, n_val)
                val_record = {
                    "event": "val",
                    "env_steps": int(env_steps),
                    "n": n_val,
                    "n_success": int(merged_success),
                    "success_rate": float(merged_success) / float(n_val),
                    "wilson_95": [low, high],
                    "seeds": val_seeds,
                    "video": video,
                    "video_seed": int(cfg["val_seed_start"]),
                    "video_success": bool(video_part["n_success"]),
                    "video_in_success": False,
                    "checkpoint_rule": cfg["checkpoint_select"],
                }
                _log(log_path, val_record)
                print(
                    f"val steps {env_steps} success {merged_success}/{n_val} "
                    f"wilson {val_record['wilson_95']} video {video} "
                    f"(video seed {int(cfg['val_seed_start'])} is not in the success rate)",
                    flush=True,
                )
                rate = float(val_record["success_rate"])
                if rate > best_val:
                    best_val = rate
                    best_val_step = int(env_steps)
                    t_mark = time.perf_counter()
                    _save(out_dir / "ckpt_best.pt", actor, agent, cfg, _extra(
                        cfg, policy, actor, env_steps, vec_steps, episodes_done, episode_successes,
                        best_val, best_val_step, stream, warmup_rng,
                    ), replay if cfg.get("save_replay") else None)
                    timing["checkpoint_s"] += time.perf_counter() - t_mark
                    timing["n_checkpoints"] += 1
                last_val_at = int(env_steps)
            if int(cfg["save_every"]) > 0 and env_steps % int(cfg["save_every"]) < n_envs:
                t_mark = time.perf_counter()
                _save(out_dir / "ckpt_last.pt", actor, agent, cfg, _extra(
                    cfg, policy, actor, env_steps, vec_steps, episodes_done, episode_successes,
                    best_val, best_val_step, stream, warmup_rng,
                ), replay if cfg.get("save_replay") else None)
                timing["checkpoint_s"] += time.perf_counter() - t_mark
                timing["n_checkpoints"] += 1
        extra = _extra(
            cfg, policy, actor, env_steps, vec_steps, episodes_done, episode_successes,
            best_val, best_val_step, stream, warmup_rng,
        )
        t_mark = time.perf_counter()
        _save(out_dir / "ckpt_last.pt", actor, agent, cfg, extra, replay if cfg.get("save_replay") else None)
        timing["checkpoint_s"] += time.perf_counter() - t_mark
        timing["n_checkpoints"] += 1
        chosen = "ckpt_last.pt"
        reason = "checkpoint_select is last"
        if cfg["checkpoint_select"] == "best_val" and (out_dir / "ckpt_best.pt").is_file():
            chosen = "ckpt_best.pt"
            reason = f"highest validation success {best_val:.4f} at env step {best_val_step}"
        elif cfg["checkpoint_select"] == "best_val":
            reason = "best_val was requested but no validation completed; using ckpt_last"
        selection = {
            "checkpoint_select": cfg["checkpoint_select"],
            "selected": chosen,
            "reason": reason,
            "best_val_success": None if best_val < 0 else best_val,
            "best_val_step": None if best_val_step < 0 else best_val_step,
            "eval_seeds_used_for_selection": False,
            "val_seeds": sorted(validation),
        }
        (out_dir / "selected.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
        print(f"selected {chosen}: {reason}", flush=True)
        timing["io_s"] = timing["replay_s"] + timing["checkpoint_s"]
        timing["n_envs"] = n_envs
        timing["env_steps"] = int(env_steps)
        timing["vec_steps"] = int(vec_steps)
        (out_dir / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
        return {
            "env_steps": env_steps,
            "episodes": episodes_done,
            "episode_successes": episode_successes,
            "selected": selection,
            "seconds": time.perf_counter() - started,
            "timing": timing,
        }
    finally:
        vec.close()
        if val_vec is not None:
            val_vec.close()
        if video_vec is not None:
            video_vec.close()


def _extra(cfg, policy, actor, env_steps, vec_steps, episodes, successes, best_val, best_val_step, stream, warmup_rng) -> dict:
    return {
        "feat_dim": int(actor.feat_dim),
        "priv_dim": int(privileged_dim()),
        "action_dim": int(actor.action_dim),
        "noise_shape": [int(v) for v in policy.noise_shape],
        "env_steps": int(env_steps),
        "vec_steps": int(vec_steps),
        "episodes": int(episodes),
        "episode_successes": int(successes),
        "best_val_success": float(best_val),
        "best_val_step": int(best_val_step),
        "bc_checkpoint": str(cfg["bc_checkpoint"]),
        "bc_sha256": sha256_file(Path(cfg["bc_checkpoint_resolved"])),
        "seed_stream_next": int(stream.next),
        "warmup_rng": warmup_rng.bit_generator.state,
    }


def load_resumed(path: Path, device, bc_hash: str):
    actor, agent, blob = load_training(path, device)
    if blob.get("bc_sha256") != bc_hash:
        raise SystemExit(
            f"resume checkpoint BC hash {blob.get('bc_sha256')} does not match {bc_hash}"
        )
    return actor, agent, blob
