from dataclasses import dataclass, field
from collections.abc import Iterable

Position = tuple[int, int]


@dataclass(frozen=True)
class Percept:
    stench: bool = False
    breeze: bool = False
    bump: bool = False

    def __str__(self) -> str:
        values: list[str] = []
        if self.stench:
            values.append("Stench")
        if self.breeze:
            values.append("Breeze")
        if self.bump:
            values.append("Bump")
        return "[" + ", ".join(values) + "]"


@dataclass
class Observation:
    percept: Percept
    neighbors: set[Position]


@dataclass
class KnowledgeBase:
    """I store the observations and apply my logical rules repeatedly.

    I use ``safe_cells`` only for cells that have neither a pit nor a Wumpus.
    A known pit can be crossed, but I do not mark it as safe.
    """

    visited: set[Position] = field(default_factory=set)
    traversable: set[Position] = field(default_factory=set)
    safe_cells: set[Position] = field(default_factory=set)
    walls: set[Position] = field(default_factory=set)
    known_pits: set[Position] = field(default_factory=set)
    known_wumpus: set[Position] = field(default_factory=set)
    possible_pits: set[Position] = field(default_factory=set)
    possible_wumpus: set[Position] = field(default_factory=set)
    not_pit: set[Position] = field(default_factory=set)
    not_wumpus: set[Position] = field(default_factory=set)
    observations: dict[Position, Observation] = field(default_factory=dict)
    visit_counts: dict[Position, int] = field(default_factory=dict)

    def record_visit(
        self,
        pos: Position,
        percept: Percept,
        neighbors: Iterable[Position],
        *,
        entered_pit: bool = False,
    ) -> None:
        neighbor_set = set(neighbors)
        self.visited.add(pos)
        self.traversable.add(pos)
        self.visit_counts[pos] = self.visit_counts.get(pos, 0) + 1
        self.not_wumpus.add(pos)

        if entered_pit:
            self.known_pits.add(pos)
            self.not_pit.discard(pos)
            self.safe_cells.discard(pos)
        else:
            self.not_pit.add(pos)

        self.observations[pos] = Observation(percept=percept, neighbors=neighbor_set)
        self._infer_until_stable()

    def mark_wall(self, pos: Position) -> None:
        self.walls.add(pos)
        self.safe_cells.discard(pos)
        self.possible_pits.discard(pos)
        self.possible_wumpus.discard(pos)
        self.known_pits.discard(pos)
        self.known_wumpus.discard(pos)
        self._infer_until_stable()

    def mark_wumpus(self, pos: Position) -> None:
        self.known_wumpus.add(pos)
        self.not_wumpus.discard(pos)
        self.safe_cells.discard(pos)
        self.possible_wumpus.add(pos)
        self._infer_until_stable()

    def mark_pit(self, pos: Position) -> None:
        self.known_pits.add(pos)
        self.not_pit.discard(pos)
        self.safe_cells.discard(pos)
        self.possible_pits.add(pos)
        self._infer_until_stable()

    def _infer_until_stable(self) -> None:
        """I repeat the inference rules until they stop producing new facts."""

        while True:
            before = self._snapshot()
            self._single_inference_pass()
            if before == self._snapshot():
                return

    def _snapshot(self) -> tuple[frozenset[Position], ...]:
        return (
            frozenset(self.safe_cells),
            frozenset(self.known_pits),
            frozenset(self.known_wumpus),
            frozenset(self.possible_pits),
            frozenset(self.possible_wumpus),
            frozenset(self.not_pit),
            frozenset(self.not_wumpus),
        )

    def _single_inference_pass(self) -> None:
        pit_constraints: list[set[Position]] = []
        wumpus_constraints: list[set[Position]] = []

        for observation in self.observations.values():
            candidates = observation.neighbors - self.walls

            if observation.percept.breeze:
                pit_candidates = candidates - self.not_pit - self.known_wumpus
                if not (pit_candidates & self.known_pits):
                    pit_constraints.append(pit_candidates)
            else:
                self.not_pit.update(candidates)

            if observation.percept.stench:
                wumpus_candidates = candidates - self.not_wumpus - self.known_pits
                if not (wumpus_candidates & self.known_wumpus):
                    wumpus_constraints.append(wumpus_candidates)
            else:
                self.not_wumpus.update(candidates)

        self.not_pit.difference_update(self.known_pits)
        self.not_wumpus.difference_update(self.known_wumpus)

        for constraint in pit_constraints:
            remaining = constraint - self.walls - self.not_pit
            if len(remaining) == 1:
                self.known_pits.update(remaining)

        for constraint in wumpus_constraints:
            remaining = constraint - self.walls - self.not_wumpus
            if len(remaining) == 1:
                self.known_wumpus.update(remaining)

        self.possible_pits = set().union(*pit_constraints) if pit_constraints else set()
        self.possible_wumpus = set().union(*wumpus_constraints) if wumpus_constraints else set()
        self.possible_pits.update(self.known_pits)
        self.possible_wumpus.update(self.known_wumpus)

        candidates = self.not_pit & self.not_wumpus
        self.safe_cells.update(
            candidates - self.walls - self.known_pits - self.known_wumpus
        )
        self.safe_cells.update(
            self.traversable - self.known_pits - self.known_wumpus
        )
        self.safe_cells.difference_update(self.walls | self.known_pits | self.known_wumpus)

    def risk_score(self, pos: Position) -> tuple[float, float, float]:
        """I estimate pit, Wumpus and combined risk for one unknown cell.

        These values are heuristic scores in ``[0, 1]``, not exact world
        probabilities. I calculate them only from observations already stored
        in the knowledge base, and I always prefer proven facts over estimates.
        """

        if pos in self.walls:
            return 1.0, 1.0, 1.0
        if pos in self.safe_cells or pos in self.traversable:
            return 0.0, 0.0, 0.0

        adjacent_observations = [
            observation
            for observation in self.observations.values()
            if pos in observation.neighbors
        ]
        if not adjacent_observations:
            # I give unsupported cells a small risk instead of treating them as safe.
            return 0.15, 0.10, 0.235

        pit_positive = sum(obs.percept.breeze for obs in adjacent_observations)
        pit_negative = sum(not obs.percept.breeze for obs in adjacent_observations)
        wumpus_positive = sum(obs.percept.stench for obs in adjacent_observations)
        wumpus_negative = sum(not obs.percept.stench for obs in adjacent_observations)

        if pos in self.not_pit:
            pit_risk = 0.0
        elif pos in self.known_pits:
            pit_risk = 1.0
        else:
            pit_risk = (0.18 + 0.24 * pit_positive) / (1.0 + 0.35 * pit_negative)

        if pos in self.not_wumpus:
            wumpus_risk = 0.0
        elif pos in self.known_wumpus:
            wumpus_risk = 1.0
        else:
            wumpus_risk = (0.10 + 0.42 * wumpus_positive) / (1.0 + 0.50 * wumpus_negative)

        pit_risk = min(1.0, max(0.0, pit_risk))
        wumpus_risk = min(1.0, max(0.0, wumpus_risk))
        combined = 1.0 - (1.0 - pit_risk) * (1.0 - wumpus_risk)
        return pit_risk, wumpus_risk, combined

    def danger_support(self, pos: Position) -> tuple[int, int]:
        """I count the nearby danger evidence while keeping proven negatives."""

        pit_support = 0
        wumpus_support = 0
        for observation in self.observations.values():
            if pos not in observation.neighbors:
                continue
            if observation.percept.breeze:
                pit_support += 1
            if observation.percept.stench:
                wumpus_support += 1

        if pos in self.not_pit:
            pit_support = 0
        if pos in self.not_wumpus:
            wumpus_support = 0

        if pos in self.known_pits:
            pit_support += 5
        if pos in self.known_wumpus:
            wumpus_support += 20
        return pit_support, wumpus_support
