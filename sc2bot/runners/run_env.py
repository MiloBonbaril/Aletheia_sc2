from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

# add 'sc2bot' package to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from sc2bot.env_sc2 import StarCraftIIGymEnv


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a short StarCraft II episode using the Gymnasium environment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to a run configuration YAML file (defaults to configs/run.yaml).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Maximum number of environment steps to execute.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed passed to env.reset.",
    )
    parser.add_argument(
        "--render-mode",
        choices=["human"],
        default=None,
        help="Requested render mode for the environment.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level to use (e.g. DEBUG, INFO).",
    )
    return parser.parse_args(argv)


def run_episode(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    env = StarCraftIIGymEnv(
        config_path=args.config,
        render_mode=args.render_mode,
    )

    try:
        observation, info = env.reset(seed=args.seed)
        logging.info(
            "Environment reset: seed=%s episode=%s config=%s",
            info.get("seed"),
            info.get("episode"),
            info.get("config_path"),
        )

        for step_idx in range(args.steps):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, step_info = env.step(action)
            logging.info(
                "step=%d action=%s reward=%.2f terminated=%s truncated=%s info=%s",
                step_idx,
                action,
                reward,
                terminated,
                truncated,
                step_info,
            )
            if terminated or truncated:
                logging.info("Episode ended early at step %d", step_idx)
                break
    finally:
        env.close()


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    run_episode(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
