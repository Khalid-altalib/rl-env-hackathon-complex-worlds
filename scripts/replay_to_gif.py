"""Replay an overcooked trajectory JSON and render it to a GIF.

Usage:
    python scripts/replay_to_gif.py trajectories/claude_overcooked_1777120061.json
    python scripts/replay_to_gif.py trajectories/claude_overcooked_1777120061.json --out replay.gif --fps 4
"""

import argparse
import json
import os
import random
import tempfile
from pathlib import Path

# Headless pygame — must be set before any pygame/overcooked import
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from PIL import Image


def extract_actions(traj: list[dict]) -> tuple[dict, list[int]]:
    """Return (task_spec, list_of_action_indices) from a trajectory JSON."""
    task_spec = {}
    for entry in traj:
        if entry.get("kind") == "config":
            task_spec = entry.get("task_spec", {})
            break

    indices = []
    for entry in traj:
        if entry.get("kind") == "message" and entry.get("role") == "assistant":
            for block in entry.get("content", []):
                if block.get("type") == "tool_use" and "play_action" in block.get("name", ""):
                    indices.append(int(block["input"]["index"]))
    return task_spec, indices


def replay(task_spec: dict, action_indices: list[int]) -> tuple[list, list]:
    """Re-run the env with the recorded actions; return (states, grids)."""
    layout = task_spec.get("layout", "cramped_room")
    horizon = int(task_spec.get("horizon", 400))

    mdp = OvercookedGridworld.from_layout_name(layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
    env.reset()

    states = [env.state]
    grid = mdp.terrain_mtx

    for idx in action_indices:
        agent_action = Action.INDEX_TO_ACTION[idx]
        partner_action = random.choice(Action.ALL_ACTIONS)
        env.step((agent_action, partner_action))
        states.append(env.state)

    return states, grid


def _draw_me_label(surface: "pygame.Surface", player_pos: tuple, tile_size: int) -> None:
    """Draw a 'Me' label above player 0's tile on the given surface."""
    pygame.font.init()
    font = pygame.font.SysFont("sans", int(tile_size * 0.38), bold=True)
    label = font.render("Me", True, (255, 255, 255))
    # centre the label horizontally over the tile, sit it near the top of the tile
    px = player_pos[0] * tile_size + (tile_size - label.get_width()) // 2
    py = player_pos[1] * tile_size + int(tile_size * 0.04)
    # dark drop-shadow for readability
    shadow = font.render("Me", True, (0, 0, 0))
    surface.blit(shadow, (px + 1, py + 1))
    surface.blit(label, (px, py))


def render_gif(states: list, grid: list, out_path: str, fps: int = 4) -> None:
    tile_size = 60
    visualizer = StateVisualizer(tile_size=tile_size, is_rendering_hud=False)

    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, state in enumerate(states):
            surface = visualizer.render_state(state, grid)
            _draw_me_label(surface, state.players[0].position, tile_size)
            png = os.path.join(tmp, f"frame_{i:04d}.png")
            pygame.image.save(surface, png)
            frames.append(Image.open(png).copy())

    duration_ms = int(1000 / fps)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration_ms,
    )
    print(f"Saved {len(frames)}-frame GIF → {out_path}  ({fps} fps)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", help="Path to trajectory JSON file")
    parser.add_argument("--out", default=None, help="Output GIF path (default: alongside JSON)")
    parser.add_argument("--fps", type=int, default=4, help="Frames per second (default: 4)")
    args = parser.parse_args()

    traj_path = Path(args.trajectory)
    out_path = args.out or str(traj_path.with_suffix(".gif"))

    print(f"Loading trajectory: {traj_path}")
    traj = json.loads(traj_path.read_text())
    task_spec, action_indices = extract_actions(traj)
    print(f"Layout: {task_spec.get('layout', 'cramped_room')}  Actions: {len(action_indices)}")

    print("Replaying actions...")
    states, grid = replay(task_spec, action_indices)
    print(f"Collected {len(states)} states")

    print("Rendering frames...")
    render_gif(states, grid, out_path, fps=args.fps)


if __name__ == "__main__":
    main()
