## Zerg Macro Proof of Concept

The `sc2bot.macros.zerg_basic` module exposes a handful of asynchronous macro helpers that integrate with python-sc2's `BotAI`:

- `drone_5` – morph drones until five workers (including eggs) are secured.
- `overlord` – queue an Overlord when supply is tight.
- `queen_inject` – command idle queens with enough energy to inject nearby hatcheries.
- `expand` – start a new Hatchery at the next expansion point if no expansion is pending.
- `ling_rush` – start a Spawning Pool if needed and morph an initial batch of Zerglings.

Each macro returns a `MacroResult` object (success flag, latency in seconds, and human‑readable details) so they can be orchestrated or logged easily.
