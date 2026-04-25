"""Full difficulty sweep: baseline vs experience-augmented agent across difficulty.

Runs Phase A (baseline, no experience) then Phase B (with the analyzer-built
experience library injected) for each difficulty preset, and produces a 2-line
comparison plot showing reward vs difficulty.

Usage:
    python scripts/difficulty_sweep.py --seeds 3 --max-turns 80 \
        --levels very_easy easy medium
"""

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from test_claude_agent import run_catan_episode  # noqa: E402
from analyze_trajectory import analyze_trajectory, DEFAULT_EXPERIENCE_PATH  # noqa: E402

DEFAULT_SEEDS = [101, 102, 103]
DEFAULT_LEVELS = ["very_easy", "easy", "medium"]


async def run_one(
    *,
    phase: str,
    difficulty: str,
    seed: int,
    max_turns: int,
    extra_prompt: str,
    model: str,
) -> dict:
    label = f"{phase}_{difficulty}_s{seed}"
    print(f"\n=== [{phase} {difficulty}] seed={seed} ===", flush=True)
    r = await run_catan_episode(
        task_spec={
            "seed": seed,
            "agent_color": "RED",
            "max_ticks": 4000,
            "difficulty": difficulty,
        },
        max_turns=max_turns,
        model=model,
        extra_system_prompt=extra_prompt,
        label=label,
    )
    record = {
        "phase": phase,
        "difficulty": difficulty,
        "seed": seed,
        "ticks": r["ticks"],
        "winner": r["winner"],
        "agent_vp_final": r["agent_vp_final"],
        "cum_reward": r["cum_reward"],
        "finished": r["finished"],
        "trajectory_path": r["trajectory_path"],
        "last_result_subtype": r["last_result_subtype"],
        "cli_error": r["cli_error"],
    }
    print(f"  -> {record}", flush=True)
    return record


def _summarize(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0}
    cum = [r["cum_reward"] for r in records]
    vp = [r["agent_vp_final"] for r in records]
    wins = sum(1 for r in records if r["winner"] == "RED")
    return {
        "n": n,
        "mean_cum_reward": sum(cum) / n,
        "min_cum_reward": min(cum),
        "max_cum_reward": max(cum),
        "mean_final_vp": sum(vp) / n,
        "win_rate": wins / n,
    }


def make_plot(
    levels: list[str],
    baseline: list[dict],
    with_exp: list[dict],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def stats_by_level(records, key):
        means, mins, maxs = [], [], []
        for lvl in levels:
            vals = [r[key] for r in records if r["difficulty"] == lvl]
            if vals:
                means.append(float(np.mean(vals)))
                mins.append(float(np.min(vals)))
                maxs.append(float(np.max(vals)))
            else:
                means.append(float("nan"))
                mins.append(float("nan"))
                maxs.append(float("nan"))
        return np.array(means), np.array(mins), np.array(maxs)

    base_r, base_r_lo, base_r_hi = stats_by_level(baseline, "cum_reward")
    exp_r, exp_r_lo, exp_r_hi = stats_by_level(with_exp, "cum_reward")
    base_v, base_v_lo, base_v_hi = stats_by_level(baseline, "agent_vp_final")
    exp_v, exp_v_lo, exp_v_hi = stats_by_level(with_exp, "agent_vp_final")

    x = np.arange(len(levels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.fill_between(x, base_r_lo, base_r_hi, alpha=0.18, color="#888888")
    ax.plot(x, base_r, "o-", color="#666666", linewidth=2, label="baseline", markersize=7)
    ax.fill_between(x, exp_r_lo, exp_r_hi, alpha=0.18, color="#3b82f6")
    ax.plot(x, exp_r, "o-", color="#1d4ed8", linewidth=2, label="+ experience library", markersize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_xlabel("difficulty")
    ax.set_ylabel("cumulative step reward (mean across seeds)")
    ax.set_title("Reward vs difficulty")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.fill_between(x, base_v_lo, base_v_hi, alpha=0.18, color="#888888")
    ax.plot(x, base_v, "o-", color="#666666", linewidth=2, label="baseline", markersize=7)
    ax.fill_between(x, exp_v_lo, exp_v_hi, alpha=0.18, color="#3b82f6")
    ax.plot(x, exp_v, "o-", color="#1d4ed8", linewidth=2, label="+ experience library", markersize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_xlabel("difficulty")
    ax.set_ylabel("final agent VP (mean across seeds)")
    ax.set_title("Final VP vs difficulty")
    # mark each level's VP target
    from catan_env.env import DIFFICULTY  # noqa: WPS433
    for i, lvl in enumerate(levels):
        vps = DIFFICULTY[lvl][2]
        ax.hlines(vps, i - 0.3, i + 0.3, color="green", linestyles="--", alpha=0.7)
    ax.legend()
    ax.grid(alpha=0.25)

    fig.suptitle("Catan: reward & VP vs difficulty (Claude Sonnet 4.5)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved plot -> {out_path}", flush=True)


async def main_async(args):
    ts = int(time.time())
    out_dir = REPO_ROOT / "trajectories" / f"sweep_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}", flush=True)
    print(f"levels: {args.levels}", flush=True)
    print(f"seeds:  {args.seeds_list}", flush=True)
    print(f"model:  {args.model}", flush=True)
    print(f"max_turns: {args.max_turns}", flush=True)

    # Phase A: baseline runs at every level (no experience), analyzer updates after each.
    print("\n############ PHASE A: BASELINE ############", flush=True)
    baseline: list[dict] = []
    for lvl in args.levels:
        for seed in args.seeds_list:
            r = await run_one(
                phase="baseline",
                difficulty=lvl,
                seed=seed,
                max_turns=args.max_turns,
                extra_prompt="",
                model=args.model,
            )
            baseline.append(r)
            (out_dir / "results_baseline.json").write_text(json.dumps(baseline, indent=2))
            try:
                await analyze_trajectory(Path(r["trajectory_path"]))
            except Exception as e:
                print(f"  [analyze] FAILED: {type(e).__name__}: {e}", file=sys.stderr, flush=True)

    # Snapshot the experience library before phase B.
    exp_text = (
        DEFAULT_EXPERIENCE_PATH.read_text()
        if DEFAULT_EXPERIENCE_PATH.exists()
        else "# Catan agent experience library\n\n(empty)"
    )
    (out_dir / "experience_after_phaseA.md").write_text(exp_text)
    print(f"\nexperience library after phase A ({len(exp_text)} chars):\n{exp_text}", flush=True)

    # Phase B: same (level, seed) pairs with the frozen experience library.
    print("\n############ PHASE B: WITH EXPERIENCE ############", flush=True)
    with_exp: list[dict] = []
    for lvl in args.levels:
        for seed in args.seeds_list:
            r = await run_one(
                phase="with_exp",
                difficulty=lvl,
                seed=seed,
                max_turns=args.max_turns,
                extra_prompt=exp_text,
                model=args.model,
            )
            with_exp.append(r)
            (out_dir / "results_with_exp.json").write_text(json.dumps(with_exp, indent=2))

    all_results = baseline + with_exp
    (out_dir / "results.json").write_text(json.dumps(all_results, indent=2))

    print("\n=== Per-(phase, level) summaries ===", flush=True)
    for lvl in args.levels:
        b = _summarize([r for r in baseline if r["difficulty"] == lvl])
        e = _summarize([r for r in with_exp if r["difficulty"] == lvl])
        lift = e.get("mean_cum_reward", 0.0) - b.get("mean_cum_reward", 0.0)
        vp_lift = e.get("mean_final_vp", 0.0) - b.get("mean_final_vp", 0.0)
        print(
            f"  {lvl:<10}  baseline={b}  with_exp={e}  reward_lift={lift:+.3f}  vp_lift={vp_lift:+.2f}",
            flush=True,
        )

    plot_path = out_dir / "difficulty_curve.png"
    make_plot(args.levels, baseline, with_exp, plot_path)
    shutil.copy(DEFAULT_EXPERIENCE_PATH, out_dir / "experience_final.md")
    print(f"\nartifacts in: {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3,
                    help=f"How many seeds from {DEFAULT_SEEDS} to use (default 3).")
    ap.add_argument("--seeds-list", type=int, nargs="*", default=None,
                    help="Explicit list of seeds (overrides --seeds).")
    ap.add_argument("--levels", nargs="*", default=DEFAULT_LEVELS,
                    help=f"Difficulty levels to sweep (default {DEFAULT_LEVELS}).")
    ap.add_argument("--max-turns", type=int, default=80)
    ap.add_argument("--model", default="claude-sonnet-4-5")
    args = ap.parse_args()

    if args.seeds_list is None:
        args.seeds_list = DEFAULT_SEEDS[: args.seeds]

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
