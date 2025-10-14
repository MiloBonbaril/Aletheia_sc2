from __future__ import annotations

import time
from typing import Optional

from sc2.bot_ai import BotAI
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.unit import Unit
from sc2.units import Units

from .exec_utils import (
    MacroResult,
    already_pending_count,
    available_larva,
    find_placement,
    issue_train_from_larva,
    issue_build_structure,
    operation_latency,
    select_structures,
    select_units,
)

_TOWNHALL_IDS: tuple[UnitTypeId, ...] = (UnitTypeId.HATCHERY, UnitTypeId.LAIR, UnitTypeId.HIVE)
async def drone_5(bot: BotAI, target: int = 5) -> MacroResult:
    """
    Ensure we have at least `target` drones by morphing larvae as needed.
    """
    start = time.perf_counter()
    try:
        current = bot.workers.amount
        pending = already_pending_count(bot, UnitTypeId.DRONE)
        missing = max(0, target - (current + pending))
        if missing == 0:
            return MacroResult(True, operation_latency(start), f"Already at {current} drones")

        issued = 0
        while missing > 0:
            if bot.supply_cap >= 200:
                break
            if bot.supply_left <= 0 and already_pending_count(bot, UnitTypeId.OVERLORD) == 0:
                if not issue_train_from_larva(bot, UnitTypeId.OVERLORD):
                    break
            if not bot.can_afford(UnitTypeId.DRONE):
                break
            if not issue_train_from_larva(bot, UnitTypeId.DRONE):
                break
            issued += 1
            missing -= 1

        new_total = bot.workers.amount + already_pending_count(bot, UnitTypeId.DRONE)
        success = new_total >= target
        details = f"Queued {issued} drones; total (including eggs): {new_total}"
        return MacroResult(success, operation_latency(start), details)
    except Exception as exc:  # pragma: no cover - defensive
        return MacroResult(False, operation_latency(start), f"drone_5 error: {exc}")


async def overlord(bot: BotAI, supply_buffer: int = 2) -> MacroResult:
    """
    Queue an Overlord when supply is tight.
    """
    start = time.perf_counter()
    try:
        if bot.supply_cap >= 200:
            return MacroResult(True, operation_latency(start), "At max supply")
        if bot.supply_left > supply_buffer:
            return MacroResult(True, operation_latency(start), f"Supply left {bot.supply_left} > buffer {supply_buffer}")
        if already_pending_count(bot, UnitTypeId.OVERLORD) > 0:
            return MacroResult(True, operation_latency(start), "Overlord already pending")
        success = issue_train_from_larva(bot, UnitTypeId.OVERLORD)
        details = "Queued Overlord" if success else "Unable to queue Overlord"
        return MacroResult(success, operation_latency(start), details)
    except Exception as exc:  # pragma: no cover - defensive
        return MacroResult(False, operation_latency(start), f"overlord error: {exc}")


async def queen_inject(bot: BotAI) -> MacroResult:
    """
    Command idle queens with enough energy to inject the nearest hatchery.
    """
    start = time.perf_counter()
    try:
        queens = select_units(bot, UnitTypeId.QUEEN, ready_only=True)
        if not queens:
            return MacroResult(False, operation_latency(start), "No queens available")
        townhalls = bot.townhalls.of_type(_TOWNHALL_IDS)
        if not townhalls:
            return MacroResult(False, operation_latency(start), "No hatcheries to inject")

        injections = 0
        for queen in queens.idle:
            if queen.energy < 25:
                continue
            target = _pick_inject_target(queen, townhalls)
            if not target:
                continue
            queen(AbilityId.EFFECT_INJECTLARVA, target)
            injections += 1

        details = f"Issued {injections} injects" if injections else "No inject issued"
        return MacroResult(injections > 0, operation_latency(start), details)
    except Exception as exc:  # pragma: no cover - defensive
        return MacroResult(False, operation_latency(start), f"queen_inject error: {exc}")


async def expand(bot: BotAI, max_distance: float = 8.0) -> MacroResult:
    """
    Take the next expansion if resources allow and none is already pending.
    """
    start = time.perf_counter()
    try:
        if already_pending_count(bot, UnitTypeId.HATCHERY) > 0:
            return MacroResult(True, operation_latency(start), "Expansion already pending")

        townhalls = bot.townhalls.of_type(_TOWNHALL_IDS)
        if not townhalls:
            return MacroResult(False, operation_latency(start), "No active townhall to expand from")

        location = await bot.get_next_expansion()
        if location is None:
            return MacroResult(False, operation_latency(start), "No viable expansion location")

        worker = bot.select_build_worker(location)
        if worker is None:
            return MacroResult(False, operation_latency(start), "No worker available for expansion")

        placement = await find_placement(bot, UnitTypeId.HATCHERY, near=location, max_distance=max_distance)
        if placement is None:
            placement = location
        if not issue_build_structure(bot, worker, UnitTypeId.HATCHERY, placement):
            return MacroResult(False, operation_latency(start), "Failed to issue hatchery build command")

        return MacroResult(True, operation_latency(start), "Expansion command issued")
    except Exception as exc:  # pragma: no cover - defensive
        return MacroResult(False, operation_latency(start), f"expand error: {exc}")


async def ling_rush(bot: BotAI, target_zerglings: int = 6) -> MacroResult:
    """
    Prototype Zergling rush macro: ensure a Spawning Pool and morph early lings.
    """
    start = time.perf_counter()
    try:
        status: list[str] = []
        pool_ready = bool(select_structures(bot, UnitTypeId.SPAWNINGPOOL, ready_only=True))
        if not pool_ready and already_pending_count(bot, UnitTypeId.SPAWNINGPOOL) == 0:
            pool_ready = await _start_spawning_pool(bot, status)

        morphs = 0
        if pool_ready:
            morphs = _train_zerglings(bot, target_zerglings)
            if morphs:
                status.append(f"Queued {morphs} lings")
        else:
            status.append("Spawning Pool pending")

        existing = bot.units(UnitTypeId.ZERGLING).amount
        pending = already_pending_count(bot, UnitTypeId.ZERGLING)
        total = existing + pending
        success = total >= target_zerglings or morphs > 0
        status.append(f"Ling total (incl. eggs): {total}")

        return MacroResult(success, operation_latency(start), "; ".join(status))
    except Exception as exc:  # pragma: no cover - defensive
        return MacroResult(False, operation_latency(start), f"ling_rush error: {exc}")


def _pick_inject_target(queen: Unit, townhalls: Units) -> Optional[Unit]:
    # Prefer hatcheries without an active inject buff, fallback to any ready hatchery.
    injectable = townhalls.filter(lambda townhall: not townhall.has_buff(BuffId.QUEENSPAWNLARVATIMER))
    target_group = injectable if injectable else townhalls.ready
    if not target_group:
        return None
    return target_group.closest_to(queen)


async def _start_spawning_pool(bot: BotAI, status: list[str]) -> bool:
    townhalls = bot.townhalls.of_type(_TOWNHALL_IDS).ready
    if not townhalls:
        status.append("No base for spawning pool")
        return False
    if not bot.can_afford(UnitTypeId.SPAWNINGPOOL):
        status.append("Insufficient resources for spawning pool")
        return False

    base = townhalls.first
    candidate = base.position.towards(bot.game_info.map_center, 6)
    placement = await find_placement(bot, UnitTypeId.SPAWNINGPOOL, candidate, max_distance=4.0)
    if placement is None:
        status.append("Could not find placement for spawning pool")
        return False

    worker = bot.select_build_worker(placement)
    if worker is None:
        status.append("No worker for spawning pool")
        return False

    if not issue_build_structure(bot, worker, UnitTypeId.SPAWNINGPOOL, placement):
        status.append("Failed to issue spawning pool build")
        return False

    status.append("Spawning Pool started")
    return True


def _train_zerglings(bot: BotAI, target_zerglings: int) -> int:
    existing = bot.units(UnitTypeId.ZERGLING).amount
    pending = already_pending_count(bot, UnitTypeId.ZERGLING)
    missing = max(0, target_zerglings - (existing + pending))
    if missing <= 0:
        return 0

    morphed = 0
    for _ in range((missing + 1) // 2):
        if bot.supply_left < 2 and already_pending_count(bot, UnitTypeId.OVERLORD) == 0:
            issue_train_from_larva(bot, UnitTypeId.OVERLORD)
            if bot.supply_left < 2:
                break
        if not bot.can_afford(UnitTypeId.ZERGLING):
            break
        larvae = available_larva(bot)
        if not larvae:
            break
        larvae.random.train(UnitTypeId.ZERGLING)
        morphed += 2

    return morphed
