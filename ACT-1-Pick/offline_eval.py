#!/usr/bin/env python3
"""离线评估一个训练好的 ACT 检查点：不接机械臂，只读数据集。

它回答三个问题，全部只用「训练时没见过的回合」：

  1. 开环：模型看一眼就交出未来 100 步，这 100 步和示教里真实发生的差多少？
     并按「第几步」分别统计——分块越往后越难，这条曲线比一个平均数有用得多。
  2. 半闭环：把模型自己的输出当关节目标反馈回去（真机上就是这样），
     误差会不会越滚越大。画面仍然来自数据集，⛔ 所以这不是真机成功率。
  3. 越界：预测出来的动作有没有跑到关节行程之外，以及比示教见过的范围外推了多远。

⛔ 它证明不了任务会成功。没有 SO-101 的仿真环境，成功率只有真机能测。
   它挡的是另一类事故：训练跑偏了、检查点选错了、动作发散或越界——
   这些在花一小时摆 20 次真机测试之前就该发现。

⚠️ 单位不猜，必须用 --action-units 说明（见 --help）。LeRobot 0.6.1 的
   so101_follower 默认 use_degrees=True ⇒ 五个本体关节是「度」，
   而夹爪永远是 RANGE_0_100 的「百分比」（so_follower.py 里写死的，不随 use_degrees 变）。
   早期版本默认 RANGE_M100_100 ⇒ 六个都是百分比。官方的
   lerobot/svla_so101_pickplace 就是百分比那一代，⛔ 别把它的数字当度数报。

用法示例（从仓根跑）：

    # 我们自己的数据集（0.6.1 录的 ⇒ 度）
    .venv/bin/python ACT-1-Pick/offline_eval.py \
        --policy-path outputs/train/act_snack_in_box/checkpoints/last/pretrained_model \
        --dataset-repo-id "$HF_USER/so101_snack_in_box_v1" \
        --dataset-root datasets/so101_snack_in_box_v1 \
        --episodes 45 46 47 48 49 \
        --action-units degrees --model-dir models/so101

    # 同时比几个检查点，挑最好的那个带去真机
    .venv/bin/python ACT-1-Pick/offline_eval.py \
        --policy-path outputs/train/act_snack_in_box/checkpoints/{005000,035000,050000,100000}/pretrained_model \
        ... （其余同上）
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# 夹爪在 LeRobot 里永远是 0–100 的百分比，不随 use_degrees 变
# （site-packages/lerobot/robots/so_follower/so_follower.py 里写死 MotorNormMode.RANGE_0_100）。
GRIPPER_JOINT = "gripper"

# 半闭环反馈的一阶低通系数：目标与当前状态之间每步走多少。
# 取值同 lerobot-doctor 的 DEMO_FOLLOW_ALPHA，好让两边的数字可比。
# ⚠️ 这是个模拟参数，不是真机的跟随特性；换了它闭环误差就会变，所以它出现在报告里。
FOLLOW_ALPHA = 0.35

# 百分比模式下的硬边界（MotorNormMode.RANGE_M100_100 / RANGE_0_100）。
PERCENT_BOUNDS = (-100.0, 100.0)
GRIPPER_PERCENT_BOUNDS = (0.0, 100.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--policy-path", type=Path, nargs="+", required=True,
        help="一个或多个 pretrained_model 目录。给多个就逐个评、最后排个序。",
    )
    p.add_argument("--dataset-repo-id", required=True, help="数据集的 repo id")
    p.add_argument(
        "--dataset-root", type=Path, required=True,
        help="数据集在磁盘上的根目录（含 meta/、data/、videos/）",
    )
    p.add_argument(
        "--episodes", type=int, nargs="+", required=True,
        help="要评的回合号。⛔ 必须是训练时留出、没喂给模型的那些，否则这个数只是背书成绩。",
    )
    p.add_argument(
        "--action-units", choices=("degrees", "percent"), required=True,
        help="数据集 action 的单位。degrees=五个本体关节是度（LeRobot 0.6.1 默认）；"
             "percent=六个都是 −100..100 百分比（早期默认，官方 svla_so101_pickplace 属此类）。"
             "⛔ 不猜：报错的单位会让后面所有数字失去意义。",
    )
    p.add_argument(
        "--model-dir", type=Path, default=None,
        help="含 so101_new_calib.urdf 的目录。给了才能按真实关节行程查越界；"
             "不给就只按归一化硬边界查，并在报告里标明。",
    )
    p.add_argument("--device", default="cuda", help="cuda / cpu / mps")
    p.add_argument(
        "--max-frames-per-episode", type=int, default=0,
        help="每个回合最多看多少帧，0 = 全看。调试时用。",
    )
    p.add_argument("--json-out", type=Path, default=None, help="把完整结果写成 JSON")
    return p.parse_args(argv)


def urdf_limits_deg(model_dir: Path) -> dict[str, tuple[float, float]]:
    """从 URDF 读每个转动关节的行程，转成度。⛔ 不在代码里写死任何角度。"""
    urdf = model_dir / "so101_new_calib.urdf"
    if not urdf.is_file():
        raise FileNotFoundError(f"找不到 {urdf}；--model-dir 指对了吗？")
    out: dict[str, tuple[float, float]] = {}
    for joint in ET.parse(urdf).getroot().iter("joint"):
        name = joint.get("name")
        limit = joint.find("limit")
        if not name or joint.get("type") != "revolute" or limit is None:
            continue
        lower, upper = limit.get("lower"), limit.get("upper")
        if lower is None or upper is None:
            continue
        out[name] = (math.degrees(float(lower)), math.degrees(float(upper)))
    if not out:
        raise ValueError(f"{urdf} 里没读到任何 revolute 关节的 limit")
    return out


def bounds_for(joint_names: list[str], units: str,
               model_dir: Path | None) -> tuple[dict[str, tuple[float, float]], str]:
    """每个关节的合法区间 + 这些区间是怎么来的（要写进报告）。"""
    if units == "percent":
        b = {n: (GRIPPER_PERCENT_BOUNDS if n == GRIPPER_JOINT else PERCENT_BOUNDS)
             for n in joint_names}
        return b, "归一化硬边界（本体 ±100，夹爪 0–100）"
    # degrees：本体查 URDF，夹爪仍是百分比
    if model_dir is None:
        b = {n: (GRIPPER_PERCENT_BOUNDS if n == GRIPPER_JOINT else (-math.inf, math.inf))
             for n in joint_names}
        return b, "⚠️ 未给 --model-dir，本体关节未查越界；夹爪按 0–100"
    lim = urdf_limits_deg(model_dir)
    b, missing = {}, []
    for n in joint_names:
        if n == GRIPPER_JOINT:
            b[n] = GRIPPER_PERCENT_BOUNDS
        elif n in lim:
            b[n] = lim[n]
        else:
            b[n] = (-math.inf, math.inf)
            missing.append(n)
    src = f"URDF 行程（{model_dir / 'so101_new_calib.urdf'}）；夹爪按 0–100"
    if missing:
        src += f"；⚠️ URDF 里没找到这些关节，未查：{', '.join(missing)}"
    return b, src


def joint_names_from_meta(meta) -> list[str]:
    """action 的六个分量叫什么。数据集里是 'shoulder_pan.pos' 这种，去掉 .pos 后缀。

    ⚠️ LeRobot 0.6.1 起 info 是对象不是字典，dict 取法会报 DeprecationWarning；
    两种都兜住，好让这个脚本不跟着上游小版本一起碎。"""
    features = getattr(meta.info, "features", None)
    if features is None:
        features = meta.info["features"]
    action = features["action"]
    names = getattr(action, "names", None)
    if names is None and isinstance(action, dict):
        names = action.get("names")
    return [str(n).removesuffix(".pos") for n in (names or [])]


def _as_float_list(value) -> list[float] | None:
    """stats 里的 min/max 可能是 numpy 数组——⛔ 别对它做真值判断。"""
    if value is None:
        return None
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        return None


def evaluate_one(policy_path: Path, args, device, ds_cache: dict) -> dict:
    """评一个检查点。返回一份可直接进 JSON 的结果。"""
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    cfg.pretrained_path = str(policy_path)
    cfg.device = str(device)

    # 数据集按 episodes 缓存：同一批回合评多个检查点时不重复解码
    key = tuple(args.episodes)
    if key not in ds_cache:
        ds_cache[key] = LeRobotDataset(
            args.dataset_repo_id, root=str(args.dataset_root), episodes=list(args.episodes)
        )
    ds = ds_cache[key]
    meta = ds.meta

    policy = make_policy(cfg, ds_meta=meta)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    cams = list(meta.camera_keys)
    names = joint_names_from_meta(meta)
    n_joints = len(names)
    chunk = int(getattr(policy.config, "chunk_size", 100))
    n_action_steps = int(getattr(policy.config, "n_action_steps", chunk))
    try:
        task = str(meta.tasks.index[0])
    except Exception:
        task = str(ds[0].get("task", ""))
    robot_type = meta.robot_type or "so101_follower"

    bounds, bounds_src = bounds_for(names, args.action_units, args.model_dir)

    # ---- 逐回合把帧取出来（动作与画面），按回合切开，⛔ 不跨回合混 ----
    ep_index_col = ds.hf_dataset["episode_index"]
    by_ep: dict[int, list[int]] = {}
    for i, e in enumerate(ep_index_col):
        by_ep.setdefault(int(e), []).append(i)

    # ---- 开环：每 n_action_steps 查一次，和真实未来逐步比 ----
    horizon_abs: list[list[float]] = [[] for _ in range(chunk)]   # 每个 horizon 一桶，跨关节平均
    per_joint_abs: list[list[float]] = [[] for _ in range(n_joints)]
    oob = {n: 0 for n in names}
    oob_total = 0
    extrapolation = {n: 0.0 for n in names}
    stats = meta.stats or {}
    seen_lo = _as_float_list((stats.get("action", {}) or {}).get("min"))
    seen_hi = _as_float_list((stats.get("action", {}) or {}).get("max"))

    # ---- 半闭环 ----
    closed_abs: list[float] = []
    diverged_episodes: list[int] = []
    # ⚠️ 队列一次装 n_action_steps 个动作，装满了才会再查一次模型。
    # 所以回合比一个动作块还短时，整段只有一次前向、状态反馈根本没机会起作用，
    # 半闭环会和开环逐位相同——那个数没有意义，必须说出来而不是让人误读。
    short_episodes: list[int] = []

    with torch.inference_mode():
        for ep, rows in sorted(by_ep.items()):
            if args.max_frames_per_episode:
                rows = rows[: args.max_frames_per_episode]
            actions = [ds[i]["action"].numpy().tolist() for i in rows]
            frames = []
            for i in rows:
                item = ds[i]
                f = {}
                for c in cams:
                    img = item[c]
                    if img.dtype != torch.uint8:
                        img = (img * 255).round().clamp(0, 255).to(torch.uint8)
                    f[c] = img
                frames.append(f)
            n = len(rows)

            # 开环
            for t in range(0, n, n_action_steps):
                policy.reset()
                obs = _obs(frames[t], actions[t], device, task, robot_type, cams)
                pred = post(policy.predict_action_chunk(pre(obs)))[0]
                pred = pred.detach().to("cpu").float().numpy()
                for h in range(min(chunk, n - t)):
                    p, truth = pred[h], actions[t + h]
                    diffs = [abs(float(p[j]) - float(truth[j])) for j in range(n_joints)]
                    horizon_abs[h].append(sum(diffs) / n_joints)
                    for j in range(n_joints):
                        per_joint_abs[j].append(diffs[j])
                        lo, hi = bounds[names[j]]
                        v = float(p[j])
                        if v < lo or v > hi:
                            oob[names[j]] += 1
                            oob_total += 1
                        if seen_lo is not None and seen_hi is not None:
                            over = max(seen_lo[j] - v, v - seen_hi[j], 0.0)
                            if over > extrapolation[names[j]]:
                                extrapolation[names[j]] = over

            # 半闭环：状态用模型自己的输出反馈
            if n <= n_action_steps:
                short_episodes.append(ep)
            policy.reset()
            sim = list(actions[0])
            errs = []
            for t in range(n):
                obs = _obs(frames[t], sim, device, task, robot_type, cams)
                act = post(policy.select_action(pre(obs)))[0]
                target = act.detach().to("cpu").float().numpy().tolist()
                sim = [s + FOLLOW_ALPHA * (tg - s) for s, tg in zip(sim, target)]
                errs.append(sum(abs(float(a) - float(b))
                                for a, b in zip(target, actions[t])) / n_joints)
            closed_abs.extend(errs)
            # 发散判据：后四分之一的平均误差比前四分之一大三倍以上
            q = max(1, len(errs) // 4)
            if statistics.mean(errs[-q:]) > 3.0 * max(statistics.mean(errs[:q]), 1e-6):
                diverged_episodes.append(ep)

    def m(xs):
        return round(statistics.mean(xs), 3) if xs else None

    horizon_curve = [m(b) for b in horizon_abs if b]
    return {
        "policy_path": str(policy_path),
        "chunk_size": chunk,
        "n_action_steps": n_action_steps,
        "episodes": list(args.episodes),
        "frames_scored": len(closed_abs),
        "action_units": args.action_units,
        "gripper_units": "percent (RANGE_0_100, 与 use_degrees 无关)",
        "open_loop": {
            "mean_abs_all_joints": m([x for b in horizon_abs for x in b]),
            "per_joint_mean_abs": {names[j]: m(per_joint_abs[j]) for j in range(n_joints)},
            "by_horizon_first": horizon_curve[0] if horizon_curve else None,
            "by_horizon_last": horizon_curve[-1] if horizon_curve else None,
            "by_horizon_curve": horizon_curve,
        },
        "closed_loop": {
            "mean_abs_all_joints": m(closed_abs),
            "follow_alpha": FOLLOW_ALPHA,
            "diverged_episodes": diverged_episodes,
            "degenerate_episodes_shorter_than_one_chunk": short_episodes,
            "meaningful": not short_episodes,
        },
        "limits": {
            "source": bounds_src,
            "out_of_bounds_predictions": oob_total,
            "out_of_bounds_by_joint": {k: v for k, v in oob.items() if v},
            "max_extrapolation_beyond_demo_range": {
                k: round(v, 3) for k, v in extrapolation.items() if v > 0
            },
        },
    }


def _obs(frame: dict, state, device, task: str, robot_type: str, cams: list[str]):
    import torch
    obs = {c: (frame[c].to(torch.float32) / 255.0).unsqueeze(0).to(device) for c in cams}
    obs["observation.state"] = torch.as_tensor(
        [float(x) for x in state], dtype=torch.float32
    ).unsqueeze(0).to(device)
    obs["task"] = task
    obs["robot_type"] = robot_type
    return obs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        print("⛔ 要 cuda 但 torch.cuda.is_available() 是 False；用 --device cpu 或修环境", file=sys.stderr)
        return 2
    device = torch.device(args.device)

    unit_word = "度（夹爪为百分比）" if args.action_units == "degrees" else "百分比"
    results, cache = [], {}
    for path in args.policy_path:
        if not (path / "config.json").is_file():
            print(f"⛔ {path} 里没有 config.json，不像是 pretrained_model 目录", file=sys.stderr)
            return 2
        print(f"—— 评 {path}", flush=True)
        results.append(evaluate_one(path, args, device, cache))

    print()
    print(f"数据集单位：{unit_word}　留出回合：{args.episodes}")
    print(f"越界判据来源：{results[0]['limits']['source']}")
    print()
    head = f"{'检查点':<30} {'开环':>9} {'第1步':>8} {'第100步':>8} {'半闭环':>9} {'越界':>7} {'发散回合':>9}"
    print(head)
    print("-" * len(head))
    for r in sorted(results, key=lambda x: (x["open_loop"]["mean_abs_all_joints"] is None,
                                            x["open_loop"]["mean_abs_all_joints"] or 0)):
        pp = Path(r["policy_path"])
        name = pp.parent.name if pp.name == "pretrained_model" else pp.name
        o = r["open_loop"]
        print(f"{name:<30} {str(o['mean_abs_all_joints']):>9} {str(o['by_horizon_first']):>8} "
              f"{str(o['by_horizon_last']):>8} {str(r['closed_loop']['mean_abs_all_joints']):>9} "
              f"{r['limits']['out_of_bounds_predictions']:>7} "
              f"{len(r['closed_loop']['diverged_episodes']):>9}")
    print()
    degenerate = sorted({e for r in results
                         for e in r["closed_loop"]["degenerate_episodes_shorter_than_one_chunk"]})
    if degenerate:
        print(f"⚠️ 回合 {degenerate} 比一个动作块（{results[0]['n_action_steps']} 步）还短：")
        print("   整段只有一次前向，状态反馈没起作用 ⇒ 这些回合的半闭环数等于开环数，⛔ 别当成两个独立结论。")
        print()
    print("⛔ 这几个数都不是成功率。它们只说明策略学到了示教、输出合法且不发散；")
    print("   任务做不做得成，只有真机能测。")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n完整结果：{args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
