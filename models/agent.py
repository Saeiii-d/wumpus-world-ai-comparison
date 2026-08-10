import random
from dataclasses import dataclass, field
from enum import Enum

Position = tuple[int, int]


class Direction(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

    @classmethod
    def all(cls) -> tuple["Direction", ...]:
        return cls.UP, cls.DOWN, cls.LEFT, cls.RIGHT


@dataclass
class AgentState:
    pos: Position
    health: int
    has_gold: bool = False
    steps_taken: int = 0
    is_alive: bool = True
    path: list[Position] = field(default_factory=list)
    pit_entries: int = 0
    death_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.path:
            self.path = [self.pos]

    def copy(self) -> "AgentState":
        return AgentState(
            pos=self.pos,
            health=self.health,
            has_gold=self.has_gold,
            steps_taken=self.steps_taken,
            is_alive=self.is_alive,
            path=self.path[:],
            pit_entries=self.pit_entries,
            death_reason=self.death_reason,
        )


GENE_NAMES = (
    "explore",
    "known_safe",
    "toward_frontier",
    "toward_exit",
    "avoid_wumpus_risk",
    "avoid_pit_risk",
    "avoid_revisit",
    "avoid_known_pit",
    "avoid_backtrack",
    "information_gain",
    "health_reserve",
)


@dataclass
class GeneticAgent:
    """I store the heuristic weights of one genetic chromosome here."""

    weights: dict[str, float]

    @classmethod
    def random_agent(cls, rng: random.Random) -> "GeneticAgent":
        weights = {name: rng.uniform(0.05, 1.0) for name in GENE_NAMES}
        agent = cls(weights)
        agent.normalize()
        return agent

    def copy(self) -> "GeneticAgent":
        return GeneticAgent(self.weights.copy())

    def normalize(self) -> None:
        for name in GENE_NAMES:
            self.weights[name] = max(0.001, float(self.weights.get(name, 0.001)))
        total = sum(self.weights.values())
        for name in GENE_NAMES:
            self.weights[name] /= total
