from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Mapping, Tuple

from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race
from sc2.ids.unit_typeid import UnitTypeId
from sc2.main import run_game
from sc2.player import Bot, Computer

from sc2bot.macros import MacroResult, drone_5, expand, ling_rush, overlord, queen_inject

MacroCallable = Callable[[BotAI], Awaitable[MacroResult]]

GAME_LOOPS_PER_SECOND = 22.4
SECONDS_PER_MINUTE = 60.0
GAME_LOOPS_PER_MINUTE = GAME_LOOPS_PER_SECOND * SECONDS_PER_MINUTE

# The SC2MacroEnv action space uses this ordering, keep in sync with env_sc2.MACROS.
MACRO_NAMES_IN_ACTION_ORDER: Tuple[str, ...] = ("drone_5", "overlord", "queen_inject", "expand", "ling_rush")
MACRO_INDEX_BY_NAME: Dict[str, int] = {name: idx for idx, name in enumerate(MACRO_NAMES_IN_ACTION_ORDER)}

EARLY_STAGE_WEIGHTS: Dict[str, float] = {
    "overlord": 4.0,
    "drone_5": 5.0,
    "queen_inject": 3.0,
    "expand": 1.0,
    "ling_rush": 2.0,
}
MID_STAGE_WEIGHTS: Dict[str, float] = {
    "overlord": 2.5,
    "drone_5": 3.0,
    "queen_inject": 3.5,
    "expand": 3.0,
    "ling_rush": 3.5,
}
LATE_STAGE_WEIGHTS: Dict[str, float] = {
    "overlord": 1.5,
    "drone_5": 1.5,
    "queen_inject": 2.5,
    "expand": 4.0,
    "ling_rush": 4.0,
}


@dataclass(frozen=True)
class MacroEntry:
    name: str
    func: MacroCallable


class ScriptedZergAgent:
    """Macro-level policy that mirrors ScriptedZergBot heuristics for SC2MacroEnv."""

    MACRO_NAMES: Tuple[str, ...] = MACRO_NAMES_IN_ACTION_ORDER

    def __init__(
        self,
        *,
        decision_interval_loops: int = 8,
        seed: int | None = None,
    ) -> None:
        self._decision_interval = max(1, int(decision_interval_loops))
        self._rng = random.Random(seed)
        self._next_decision_loop: int = 0
        self._last_observed_loop: int = 0

    def reset(self, observation: Mapping[str, Any], info: Mapping[str, Any]) -> None:
        loop = self._extract_game_loop(observation, info)
        self._next_decision_loop = loop
        self._last_observed_loop = loop

    @property
    def macro_names(self) -> Tuple[str, ...]:
        return self.MACRO_NAMES

    def cooldown_seconds(self, observation: Mapping[str, Any], info: Mapping[str, Any]) -> float:
        current_loop = max(self._extract_game_loop(observation, info), self._last_observed_loop)
        deficit = self._next_decision_loop - current_loop
        if deficit <= 0:
            return 0.0
        return deficit / GAME_LOOPS_PER_SECOND

    def select_action(self, observation: Mapping[str, Any], info: Mapping[str, Any]) -> tuple[int, Dict[str, Any]]:
        current_loop = max(self._extract_game_loop(observation, info), self._last_observed_loop)
        self._last_observed_loop = current_loop

        stage, weights = self._weights_for_loop(current_loop)
        forced_macro, reason = self._supply_mask_choice(observation)

        if forced_macro is not None:
            macro_name = forced_macro
            reason = reason or "supply_mask"
            log_weights = {"overlord": 1.0}
        else:
            macro_name = self._weighted_choice(weights)
            reason = f"weighted(stage={stage})"
            log_weights = {name: round(weights.get(name, 0.0), 2) for name in self.MACRO_NAMES}

        self._next_decision_loop = current_loop + self._decision_interval
        decision = {
            "macro": macro_name,
            "stage": stage,
            "reason": reason,
            "next_loop": self._next_decision_loop,
            "weights": log_weights,
        }
        return MACRO_INDEX_BY_NAME[macro_name], decision

    def _weighted_choice(self, weights: Dict[str, float]) -> str:
        distribution = [max(0.0, weights.get(name, 0.0)) for name in self.MACRO_NAMES]
        if sum(distribution) <= 0.0:
            distribution = [1.0] * len(self.MACRO_NAMES)
        index = self._rng.choices(range(len(self.MACRO_NAMES)), weights=distribution, k=1)[0]
        return self.MACRO_NAMES[index]

    def _weights_for_loop(self, game_loop: int) -> tuple[str, Dict[str, float]]:
        minutes = game_loop / GAME_LOOPS_PER_MINUTE if GAME_LOOPS_PER_MINUTE > 0 else 0.0
        if minutes < 3.0:
            return "early", EARLY_STAGE_WEIGHTS
        if minutes < 6.0:
            return "mid", MID_STAGE_WEIGHTS
        return "late", LATE_STAGE_WEIGHTS

    def _supply_mask_choice(self, observation: Mapping[str, Any]) -> tuple[str | None, str | None]:
        supply_cap = self._vec_value(observation, 3)
        supply_left = self._vec_value(observation, 5)
        if supply_cap >= 200:
            return None, None
        if supply_left > 1:
            return None, None
        return "overlord", f"supply_left={int(supply_left)}"

    def _extract_game_loop(self, observation: Mapping[str, Any], info: Mapping[str, Any]) -> int:
        loop = info.get("game_loop")
        if loop is not None:
            try:
                return int(loop)
            except (TypeError, ValueError):
                pass
        vec = observation.get("vec") if observation is not None else None
        if vec is None:
            return 0
        try:
            return int(float(vec[0]))
        except (TypeError, ValueError, IndexError):
            return 0

    def _vec_value(self, observation: Mapping[str, Any], index: int, default: float = 0.0) -> float:
        vec = observation.get("vec") if observation is not None else None
        if vec is None:
            return default
        try:
            return float(vec[index])
        except (TypeError, ValueError, IndexError):
            return default


class ScriptedZergBot(BotAI):
    """Hand-authored Zerg agent that stitches together macro helpers."""

    _MACROS: Tuple[MacroEntry, ...] = (
        MacroEntry("overlord", overlord),
        MacroEntry("drone_5", drone_5),
        MacroEntry("queen_inject", queen_inject),
        MacroEntry("expand", expand),
        MacroEntry("ling_rush", ling_rush),
    )

    _EARLY_WEIGHTS: Dict[str, float] = EARLY_STAGE_WEIGHTS
    _MID_WEIGHTS: Dict[str, float] = MID_STAGE_WEIGHTS
    _LATE_WEIGHTS: Dict[str, float] = LATE_STAGE_WEIGHTS

    def __init__(
        self,
        *,
        seed: int | None = None,
        decision_interval_loops: int = 8,
    ) -> None:
        super().__init__()
        self._rng = random.Random(seed)
        self._decision_interval = max(1, int(decision_interval_loops))
        self._next_decision_loop: int = 0
        self._logger = logging.getLogger(self.__class__.__name__)

    async def on_start(self) -> None:
        await super().on_start()
        self._logger.info("Scripted Zerg agent ready. Decision interval: %d loops.", self._decision_interval)

    async def on_step(self, iteration: int) -> None:
        await super().on_step(iteration)
        game_loop = int(self.state.game_loop)
        if game_loop < self._next_decision_loop:
            return
        self._next_decision_loop = game_loop + self._decision_interval

        stage, weights = self._weights_for_current_phase()
        forced_macro, mask_reason = self._supply_mask_choice()

        if forced_macro is not None:
            macro_entry = forced_macro
            selection_reason = f"supply_mask({mask_reason})"
        else:
            macro_entry = self._weighted_choice(weights)
            selection_reason = f"weighted(stage={stage})"

        try:
            result = await macro_entry.func(self)
        except Exception:  # pragma: no cover - defensive guard rail
            self._logger.exception("Macro %s raised an exception", macro_entry.name)
            return
        if forced_macro is None:
            weights_for_log = {name: round(weight, 2) for name, weight in weights.items()}
        else:
            weights_for_log = {"overlord": 1.0}

        self._logger.info(
            "loop=%d iter=%d stage=%s macro=%s choice=%s success=%s latency=%.3f details=%s weights=%s",
            game_loop,
            iteration,
            stage,
            macro_entry.name,
            selection_reason,
            result.success,
            result.latency,
            result.details,
            weights_for_log,
        )

    def _supply_mask_choice(self) -> Tuple[MacroEntry | None, str | None]:
        if self.supply_cap >= 200:
            return None, None
        if self.supply_left > 1:
            return None, None
        if self.already_pending(UnitTypeId.OVERLORD) > 0:
            return None, None
        macro = next(entry for entry in self._MACROS if entry.name == "overlord")
        reason = f"supply_left={self.supply_left}"
        return macro, reason

    def _weighted_choice(self, weights: Dict[str, float]) -> MacroEntry:
        choices: list[MacroEntry] = []
        distribution: list[float] = []
        for entry in self._MACROS:
            choices.append(entry)
            distribution.append(max(0.0, weights.get(entry.name, 0.0)))

        # Fallback to uniform distribution if all weights collapse to zero.
        if sum(distribution) <= 0.0:
            distribution = [1.0] * len(choices)

        index = self._rng.choices(range(len(choices)), weights=distribution, k=1)[0]
        return choices[index]

    def _weights_for_current_phase(self) -> Tuple[str, Dict[str, float]]:
        minutes = self.state.game_loop / GAME_LOOPS_PER_MINUTE
        if minutes < 3.0:
            return "early", self._EARLY_WEIGHTS
        if minutes < 6.0:
            return "mid", self._MID_WEIGHTS
        return "late", self._LATE_WEIGHTS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the scripted Zerg proof-of-concept bot.")
    parser.add_argument("--map", default="2000AtmospheresAIE", help="Ladder map to load (default: %(default)s)")
    parser.add_argument(
        "--difficulty",
        default="Easy",
        choices=[name for name in Difficulty.__members__.keys()],
        help="Computer opponent difficulty (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for macro decisions")
    parser.add_argument(
        "--decision-interval",
        type=int,
        default=8,
        help="Macro decision cadence in game loops (default: %(default)s)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Run the match in realtime mode (default: off)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    difficulty = Difficulty[args.difficulty]
    try:
        selected_map = maps.get(args.map)
    except Exception as exc:  # pragma: no cover - map availability can vary
        logging.getLogger("ScriptedZergBot").warning(
            "Falling back to AbyssalReefLE after failing to load map '%s': %s",
            args.map,
            exc,
        )
        selected_map = maps.get("AbyssalReefLE")

    bot = ScriptedZergBot(seed=args.seed, decision_interval_loops=args.decision_interval)
    logging.getLogger("ScriptedZergBot").info(
        "Launching match on %s vs Computer(%s).", selected_map, difficulty.name
    )
    run_game(
        selected_map,
        [
            Bot(Race.Zerg, bot),
            Computer(Race.Terran, difficulty),
        ],
        realtime=args.realtime,
    )


if __name__ == "__main__":
    main()
