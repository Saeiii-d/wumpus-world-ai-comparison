from dataclasses import dataclass

from data.config import GRID_SIZE, GameConfig
from models.agent import AgentState, Direction
from models.percepts import Percept

Position = tuple[int, int]


@dataclass(frozen=True)
class MoveOutcome:
    valid: bool
    event: str
    old_pos: Position
    new_pos: Position
    health_before: int
    health_after: int


class GameEnvironment:
    """I keep the real map here and expose only the allowed game operations.

    For the partially observable agents, I only use ``observe``,
    ``candidate_neighbors`` and ``apply_move``.
    """

    def __init__(self, config: GameConfig):
        self.config = config.copy()
        self.agent = AgentState(pos=self.config.start_pos, health=self.config.initial_health)
        self.game_over = False
        self.won = False
        self.termination_reason: str | None = None
        self.last_outcome: MoveOutcome | None = None

        if self.get_cell(self.agent.pos) == "G":
            self.agent.has_gold = True

        # I check this special case here because the agent may start on gold and exit.
        if self.agent.pos == self.config.exit_pos and self.agent.has_gold:
            self.won = True
            self.game_over = True
            self.termination_reason = "won"

    def in_bounds(self, pos: Position) -> bool:
        row, col = pos
        return 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE

    def get_cell(self, pos: Position) -> str | None:
        if not self.in_bounds(pos):
            return None
        row, col = pos
        return self.config.grid[row][col]

    def candidate_neighbors(self, pos: Position) -> list[Position]:
        """I return the four in-bounds positions without revealing their contents."""

        row, col = pos
        result: list[Position] = []
        for direction in Direction.all():
            dr, dc = direction.value
            candidate = row + dr, col + dc
            if self.in_bounds(candidate):
                result.append(candidate)
        return result

    def get_neighbors(self, pos: Position) -> list[Position]:
        """I return real traversable neighbors only for the fully observable search."""

        return [candidate for candidate in self.candidate_neighbors(pos) if self.get_cell(candidate) != "D"]

    def observe(self, pos: Position | None = None) -> Percept:
        location = self.agent.pos if pos is None else pos
        true_neighbors = self.get_neighbors(location)
        return Percept(
            stench=any(self.get_cell(neighbor) == "W" for neighbor in true_neighbors),
            breeze=any(self.get_cell(neighbor) == "P" for neighbor in true_neighbors),
            bump=bool(self.last_outcome and not self.last_outcome.valid),
        )

    @staticmethod
    def _adjacent(first: Position, second: Position) -> bool:
        return abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1

    def apply_move(self, new_pos: Position) -> MoveOutcome:
        if self.game_over:
            raise RuntimeError("Cannot move after the game is over")

        old_pos = self.agent.pos
        health_before = self.agent.health

        if not self._adjacent(old_pos, new_pos) or not self.in_bounds(new_pos):
            outcome = MoveOutcome(False, "invalid_move", old_pos, old_pos, health_before, health_before)
            self.last_outcome = outcome
            return outcome

        if self.get_cell(new_pos) == "D":
            outcome = MoveOutcome(False, "bump", old_pos, old_pos, health_before, health_before)
            self.last_outcome = outcome
            return outcome

        self.agent.health -= 1
        self.agent.steps_taken += 1
        self.agent.pos = new_pos
        self.agent.path.append(new_pos)

        event = "moved"
        if self.agent.health <= 0:
            self._kill("health_depleted")
            event = "health_depleted"
        else:
            cell = self.get_cell(new_pos)
            if cell == "W":
                self._kill("wumpus")
                event = "wumpus"
            elif cell == "P":
                # I keep the pit formula exactly as defined for this project.
                # health = (health // 2) - pit_penalty
                self.agent.health = (self.agent.health // 2) - self.config.pit_penalty
                self.agent.pit_entries += 1
                event = "pit"
                if self.agent.health <= 0:
                    self._kill("pit")
                    event = "pit_death"
            elif cell == "G" and not self.agent.has_gold:
                self.agent.has_gold = True
                event = "gold_collected"

            if self.agent.is_alive and new_pos == self.config.exit_pos:
                if self.agent.has_gold:
                    self.won = True
                    self.game_over = True
                    self.termination_reason = "won"
                    event = "won"
                elif event == "moved":
                    event = "exit_without_gold"

        outcome = MoveOutcome(
            True,
            event,
            old_pos,
            self.agent.pos,
            health_before,
            self.agent.health,
        )
        self.last_outcome = outcome
        return outcome

    def _kill(self, reason: str) -> None:
        self.agent.is_alive = False
        self.agent.death_reason = reason
        self.game_over = True
        self.termination_reason = reason

    def get_final_score(self) -> int:
        if not self.won:
            return 0
        return max(0, self.agent.health) + self.config.gold_score

    def copy(self) -> "GameEnvironment":
        clone = GameEnvironment(self.config)
        clone.agent = self.agent.copy()
        clone.game_over = self.game_over
        clone.won = self.won
        clone.termination_reason = self.termination_reason
        clone.last_outcome = self.last_outcome
        return clone

    def adopt_state(self, other: "GameEnvironment") -> None:
        """I copy the runtime state into this environment after an evaluation."""

        self.agent = other.agent.copy()
        self.game_over = other.game_over
        self.won = other.won
        self.termination_reason = other.termination_reason
        self.last_outcome = other.last_outcome
