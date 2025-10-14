from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Protocol

from sc2.bot_ai import BotAI
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units


class SupportsPosition(Protocol):
    position: Point2


Predicate = Callable[[], bool]
AsyncPredicate = Callable[[], Awaitable[bool]]


@dataclass(slots=True)
class MacroResult:
    """Container describing the outcome of a macro execution."""

    success: bool
    latency: float
    details: Optional[str] = None


def select_units(bot: BotAI, unit_type: UnitTypeId, ready_only: bool = False) -> Units:
    """
    Return the bot units of a given type, optionally restricting to ready ones.

    For Zerg larvae, the `ready` filter is a no-op but included for symmetry
    with structures.
    """
    units = bot.units(unit_type)
    return units.ready if ready_only else units


def select_structures(bot: BotAI, unit_type: UnitTypeId, ready_only: bool = True) -> Units:
    """
    Return structures of a given type, defaulting to ready structures only.
    """
    structures = bot.structures(unit_type)
    return structures.ready if ready_only else structures


def first_idle(units: Units) -> Optional[Unit]:
    """Return the first idle unit, if any."""
    if not units:
        return None
    idle = units.idle
    return idle.random if idle else None


def available_larva(bot: BotAI) -> Units:
    """Small helper returning all current larvae."""
    return bot.larva


def can_train(bot: BotAI, unit_type: UnitTypeId) -> bool:
    """Check both resource and supply requirements for morphing from a larva."""
    return bot.can_afford(unit_type)


def issue_train_from_larva(bot: BotAI, unit_type: UnitTypeId) -> bool:
    """
    Issue a morph command to the first available larva.

    Returns True if the order could be issued.
    """
    larvae = available_larva(bot)
    if not larvae:
        return False
    if not can_train(bot, unit_type):
        return False
    larva = larvae.random
    larva.train(unit_type)
    return True


def already_pending_count(bot: BotAI, unit_type: UnitTypeId) -> int:
    """Wrapper to expose bot's pending count for readability."""
    return bot.already_pending(unit_type)


def find_closest(unit: SupportsPosition, candidates: Units) -> Optional[Unit]:
    """Return the closest candidate to a reference unit-like object."""
    if not candidates:
        return None
    return candidates.closest_to(unit)


async def wait_for_condition(
    predicate: Predicate | AsyncPredicate,
    timeout: float,
    interval: float = 0.1,
) -> bool:
    """
    Wait for a condition to become True, returning False on timeout.
    """
    if timeout <= 0:
        return False
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        result = predicate()
        if asyncio.iscoroutine(result):
            result = await result  # type: ignore[assignment]
        if result:
            return True
        await asyncio.sleep(interval)
    return False


async def find_placement(
    bot: BotAI,
    structure: UnitTypeId,
    near: Point2,
    max_distance: float = 10.0,
    step: float = 1.0,
) -> Optional[Point2]:
    """
    Wrapper around BotAI.find_placement with sane defaults.
    """
    return await bot.find_placement(
        structure,
        near,
        max_distance=max_distance,
        random_alternative=False,
        placement_step=step,
    )


def issue_build_structure(
    bot: BotAI,
    worker: Unit,
    structure: UnitTypeId,
    position: Point2,
) -> bool:
    """
    Issue a build command with the provided worker.
    """
    if worker is None or not worker.is_alive:
        return False
    if not bot.can_afford(structure):
        return False
    ability = _BUILD_ABILITIES.get(structure)
    if ability is None:
        ability = bot._game_data.units[structure.value].creation_ability
    if ability is None:
        return False
    worker(ability, position)
    return True


def operation_latency(start_time: float) -> float:
    """Compute latency given a perf_counter start."""
    return max(0.0, time.perf_counter() - start_time)


_BUILD_ABILITIES: dict[UnitTypeId, AbilityId] = {
    UnitTypeId.HATCHERY: AbilityId.BUILD_HATCHERY,
    UnitTypeId.SPAWNINGPOOL: AbilityId.BUILD_SPAWNINGPOOL,
    UnitTypeId.EXTRACTOR: AbilityId.BUILD_EXTRACTOR,
    UnitTypeId.EVOLUTIONCHAMBER: AbilityId.BUILD_EVOLUTIONCHAMBER,
    UnitTypeId.ROACHWARREN: AbilityId.BUILD_ROACHWARREN,
    UnitTypeId.BANELINGNEST: AbilityId.BUILD_BANELINGNEST,
    UnitTypeId.SPIRE: AbilityId.BUILD_SPIRE,
}
