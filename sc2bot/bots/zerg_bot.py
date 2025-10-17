from sc2 import maps
from sc2.player import Bot, Computer
from sc2.main import run_game
from sc2.data import Race, Difficulty
from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId

#
# <Bot>
#
class WorkerRushBot(BotAI):
    async def on_step(self, iteration: int):
        for loop_larva in self.larva:
            if self.can_afford(UnitTypeId.DRONE):
                loop_larva.train(UnitTypeId.DRONE)
                # Add break statement here if you only want to train one
            else:
                # Can't afford drones anymore
                break

#
# </Bot>
#