from sc2 import maps
from sc2.player import Bot, Computer
from sc2.main import run_game
from sc2.data import Race, Difficulty
from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId

from bots.zerg_bot import WorkerRushBot

run_game(maps.get("AcropolisLE"), [
    Bot(Race.Zerg, WorkerRushBot()),
    Computer(Race.Protoss, Difficulty.Medium)
], realtime=False)

