"""Smoke test: does a trajectory-driven experience library help our Catan agent?

Pipeline:
  Phase A (baseline): for each seed, run Claude vs 3 RandomPlayer opponents with
    the original short prompt, then run scripts/analyze_trajectory.py to extract
    lessons into experiences/catan_experience.md.
  Phase B (with experience): for the same seeds, run again with the (now
    populated) experience library injected into the system prompt.

Saves all per-run records, a snapshot of the final experience file, and a
side-by-side comparison plot under trajectories/smoketest_<ts>/.

Usage:
    python scripts/experience_smoketest.py --seeds 3 --max-turns 80
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


async def run_phase(
    phase: str,
    seeds: list[int],
    max_turns: int,
    extra_prompt_fn,
    out_dir: Path,
    update_experience: bool,
) -> list[dict]:
    """extra_prompt_fn(seed) -> str. update_experience: run analyzer after each run."""
    results = []
    for seed in seeds:
        label = f"{phase}_s{seed}"
        print(f"\n=== [{phase}] seed={seed} ===")
        extra = extra_prompt_fn(seed)
        r = await run_catan_episode(
            task_spec={"seed": seed, "agent_color": "RED", "max_ticks": 4000},
            max_turns=max_turns,
            extra_system_prompt=extra,
            label=label,
        )
        record = {
            "phase": phase,
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
        results.append(record)
        print(f"  -> {record}")

        if update_experience:
            print(f"  [analyze] updating experience library from {r['trajectory_path']}")
            try:
                await analyze_trajectory(Path(r["trajectory_path"]))
            except Exception as e:
                print(f"  [analyze] FAILED: {type(e).__name__}: {e}", file=sys.stderr)

    # incremental save after each phase so a crash doesn't lose data
    (out_dir / f"results_{phase}.json").write_text(json.dumps(results, indent=2))
    return results


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


def make_plot(baseline: list[dict], with_exp: list[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    seeds = sorted({r["seed"] for r in baseline} | {r["seed"] for r in with_exp})
    base_by_seed = {r["seed"]: r for r in baseline}
    exp_by_seed = {r["seed"]: r for r in with_exp}

    base_cum = [base_by_seed.get(s, {}).get("cum_reward", float("nan")) for s in seeds]
    exp_cum = [exp_by_seed.get(s, {}).get("cum_reward", float("nan")) for s in seeds]
    base_vp = [base_by_seed.get(s, {}).get("agent_vp_final", float("nan")) for s in seeds]
    exp_vp = [exp_by_seed.get(s, {}).get("agent_vp_final", float("nan")) for s in seeds]

    x = np.arange(len(seeds))
    w = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.bar(x - w / 2, base_cum, w, label="baseline", color="#bbbbbb")
    ax.bar(x + w / 2, exp_cum, w, label="+ experience library", color="#3b82f6")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed={s}" for s in seeds])
    ax.set_ylabel("cumulative step reward")
    ax.set_title("Cumulative reward per seed")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend()

    base_mean = float(np.nanmean(base_cum)) if base_cum else 0.0
    exp_mean = float(np.nanmean(exp_cum)) if exp_cum else 0.0
    ax.text(
        0.02,
        0.98,
        f"mean baseline = {base_mean:+.3f}\nmean +exp     = {exp_mean:+.3f}\nlift          = {exp_mean - base_mean:+.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
    )

    ax = axes[1]
    ax.bar(x - w / 2, base_vp, w, label="baseline", color="#bbbbbb")
    ax.bar(x + w / 2, exp_vp, w, label="+ experience library", color="#3b82f6")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed={s}" for s in seeds])
    ax.set_ylabel("final agent VP")
    ax.set_title("Final victory points per seed")
    ax.set_ylim(0, 10)
    ax.axhline(10, color="green", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.legend()

    fig.suptitle("Catan: experience-library smoke test (vs RandomPlayer)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved plot -> {out_path}")


async def main_async(args):
    ts = int(time.time())
    out_dir = REPO_ROOT / "trajectories" / f"smoketest_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}")

    seeds = args.seeds_list or DEFAULT_SEEDS[: args.seeds]

    # Phase A: baseline runs, analyzer updates experience library after each.
    print("\n############ PHASE A: BASELINE ############")
    baseline = await run_phase(
        phase="baseline",
        seeds=seeds,
        max_turns=args.max_turns,
        extra_prompt_fn=lambda s: "",
        out_dir=out_dir,
        update_experience=True,
    )

    # Snapshot the experience library after phase A.
    exp_text = (
        DEFAULT_EXPERIENCE_PATH.read_text()
        if DEFAULT_EXPERIENCE_PATH.exists()
        else "# Catan agent experience library\n\n(empty)"
    )
    (out_dir / "experience_after_phaseA.md").write_text(exp_text)
    print(f"\nexperience library after phase A ({len(exp_text)} chars):")
    print(exp_text)

    # Phase B: same seeds, library frozen and injected into system prompt.
    print("\n############ PHASE B: WITH EXPERIENCE ############")
    with_exp = await run_phase(
        phase="with_exp",
        seeds=seeds,
        max_turns=args.max_turns,
        extra_prompt_fn=lambda s: exp_text,
        out_dir=out_dir,
        update_experience=False,
    )

    all_results = baseline + with_exp
    (out_dir / "results.json").write_text(json.dumps(all_results, indent=2))

    base_summary = _summarize(baseline)
    exp_summary = _summarize(with_exp)
    print("\n=== Phase summaries ===")
    print("baseline:", base_summary)
    print("with_exp:", exp_summary)
    lift = exp_summary.get("mean_cum_reward", 0.0) - base_summary.get("mean_cum_reward", 0.0)
    vp_lift = exp_summary.get("mean_final_vp", 0.0) - base_summary.get("mean_final_vp", 0.0)
    print(f"\n>>> reward lift from experience library: {lift:+.3f}")
    print(f">>> final-VP lift from experience library: {vp_lift:+.2f}")

    plot_path = out_dir / "comparison.png"
    make_plot(baseline, with_exp, plot_path)

    # Also keep a copy of the experience file the with-experience phase actually saw.
    shutil.copy(DEFAULT_EXPERIENCE_PATH, out_dir / "experience_final.md")
    print(f"\nartifacts in: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seeds",
        type=int,
        default=3,
        help=f"How many seeds from {DEFAULT_SEEDS} to use (default 3).",
    )
    ap.add_argument(
        "--seeds-list",
        type=int,
        nargs="*",
        default=None,
        help="Explicit list of seeds (overrides --seeds).",
    )
    ap.add_argument("--max-turns", type=int, default=80)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
