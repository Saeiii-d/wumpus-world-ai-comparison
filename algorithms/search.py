import heapq
from dataclasses import dataclass

from models.environment import GameEnvironment
from utils import manhattan_distance

Position = tuple[int, int]


@dataclass(frozen=True)
class SearchState:
    """I keep every value that can change the result of an A* state."""

    pos: Position
    has_gold: bool
    health: int


class AStarSearch:
    """
    I use a fully observable A* and try to preserve as much health as possible.
    Since the gold score is fixed, keeping more health also gives a better final score.
    """

    def __init__(self, env: GameEnvironment):
        self.env = env
        self.expanded_nodes = 0
        self.planned_path: list[Position] = []

    def _heuristic(self, state: SearchState) -> int:
        if state.has_gold:
            return manhattan_distance(state.pos, self.env.config.exit_pos)
        return (
            manhattan_distance(state.pos, self.env.config.gold_pos)
            + manhattan_distance(self.env.config.gold_pos, self.env.config.exit_pos)
        )

    def _transition(self, state: SearchState, destination: Position) -> SearchState | None:
        cell = self.env.get_cell(destination)
        if cell in {"D", "W", None}:
            return None

        health = state.health - 1
        if health <= 0:
            return None
        if cell == "P":
            health = (health // 2) - self.env.config.pit_penalty
            if health <= 0:
                return None

        return SearchState(
            pos=destination,
            has_gold=state.has_gold or cell == "G",
            health=health,
        )

    def find_path(self) -> list[Position] | None:
        start = SearchState(
            self.env.agent.pos,
            self.env.agent.has_gold,
            self.env.agent.health,
        )
        sequence = 0
        frontier: list[tuple[int, int, SearchState]] = []
        heapq.heappush(frontier, (self._heuristic(start), sequence, start))

        parent: dict[SearchState, SearchState | None] = {start: None}
        best_health: dict[tuple[Position, bool], int] = {
            (start.pos, start.has_gold): start.health
        }

        while frontier:
            _, _, current = heapq.heappop(frontier)
            if current.health < best_health.get((current.pos, current.has_gold), -1):
                continue

            self.expanded_nodes += 1
            if current.pos == self.env.config.exit_pos and current.has_gold:
                states: list[SearchState] = []
                cursor: SearchState | None = current
                while cursor is not None:
                    states.append(cursor)
                    cursor = parent[cursor]
                states.reverse()
                self.planned_path = [state.pos for state in states]
                return self.planned_path

            for neighbor in self.env.get_neighbors(current.pos):
                successor = self._transition(current, neighbor)
                if successor is None:
                    continue

                key = successor.pos, successor.has_gold
                if successor.health <= best_health.get(key, -1):
                    continue
                best_health[key] = successor.health
                parent[successor] = current
                sequence += 1
                health_loss = self.env.config.initial_health - successor.health
                priority = health_loss + self._heuristic(successor)
                heapq.heappush(frontier, (priority, sequence, successor))

        return None

    def execute(self) -> bool:
        path = self.find_path()
        if not path:
            self.env.termination_reason = "unreachable"
            return False
        for destination in path[1:]:
            outcome = self.env.apply_move(destination)
            if not outcome.valid or (self.env.game_over and not self.env.won):
                return False
        return self.env.won
