import heapq

from models.environment import GameEnvironment
from models.percepts import KnowledgeBase
from utils import manhattan_distance

Position = tuple[int, int]
HealthState = tuple[Position, int]


class LogicBasedAgent:
    """
    I solve the map with rules and only use knowledge collected online.

    First I infer safe and risky cells, then I plan on the known graph, 
    and when no safe move exists I choose the least risky option while avoiding simple loops and fatal pit entries.
    """

    def __init__(self, env: GameEnvironment, max_decisions: int = 700):
        self.env = env
        self.kb = KnowledgeBase()
        self.max_decisions = max_decisions
        self.decisions = 0

    def _observe(self, entered_pit: bool = False) -> None:
        self.kb.record_visit(
            self.env.agent.pos,
            self.env.observe(),
            self.env.candidate_neighbors(self.env.agent.pos),
            entered_pit=entered_pit,
        )

    def _allowed_graph(self) -> set[Position]:
        """I return the cells that I currently know I can intentionally cross."""

        return (
            self.kb.safe_cells
            | self.kb.traversable
            | self.kb.known_pits
            | {self.env.agent.pos}
        ) - self.kb.walls - self.kb.known_wumpus

    def _health_after_known_move(self, health: int, destination: Position) -> int | None:
        remaining = health - 1
        if remaining <= 0:
            return None
        if destination in self.kb.known_pits:
            remaining = (remaining // 2) - self.env.config.pit_penalty
            if remaining <= 0:
                return None
        return remaining

    def _required_health_reserve(self, destination: Position) -> int:
        """
        I estimate the health I should keep after entering ``destination``.

        After finding gold I use the known exit distance. Before that, I keep a
        small reserve so I do not enter a pit with too little health to continue.
        """

        if self.env.agent.has_gold:
            return manhattan_distance(destination, self.env.config.exit_pos) + 2

        safe_frontier = self.kb.safe_cells - self.kb.visited - self.kb.walls
        if safe_frontier:
            nearest = min(manhattan_distance(destination, cell) for cell in safe_frontier)
            return max(6, nearest + 4)
        return 8

    def _known_move_is_viable(self, health: int, destination: Position) -> bool:
        remaining = self._health_after_known_move(health, destination)
        if remaining is None:
            return False
        if destination in self.kb.known_pits:
            return remaining > self._required_health_reserve(destination)
        return True

    def _path_to(self, goals: set[Position]) -> list[Position] | None:
        """I find a path on my known graph that keeps the most health."""

        if self.env.agent.pos in goals:
            return [self.env.agent.pos]

        allowed = self._allowed_graph() | goals
        start: HealthState = (self.env.agent.pos, self.env.agent.health)
        frontier: list[tuple[int, int, HealthState]] = [(-self.env.agent.health, 0, start)]
        parent: dict[HealthState, HealthState | None] = {start: None}
        best_health: dict[Position, int] = {self.env.agent.pos: self.env.agent.health}
        sequence = 0

        while frontier:
            _, _, state = heapq.heappop(frontier)
            pos, health = state
            if health < best_health.get(pos, -1):
                continue
            if pos in goals:
                path: list[Position] = []
                cursor: HealthState | None = state
                while cursor is not None:
                    path.append(cursor[0])
                    cursor = parent[cursor]
                path.reverse()
                return path

            for neighbor in self.env.candidate_neighbors(pos):
                if neighbor not in allowed:
                    continue
                next_health = self._health_after_known_move(health, neighbor)
                if next_health is None:
                    continue
                if neighbor in self.kb.known_pits:
                    reserve = (
                        manhattan_distance(neighbor, self.env.config.exit_pos) + 2
                        if self.env.agent.has_gold
                        else 5
                    )
                    if next_health <= reserve:
                        continue
                if next_health <= best_health.get(neighbor, -1):
                    continue
                next_state = (neighbor, next_health)
                best_health[neighbor] = next_health
                parent[next_state] = state
                sequence += 1
                heapq.heappush(frontier, (-next_health, sequence, next_state))
        return None

    def _unknown_neighbors(self, cell: Position) -> list[Position]:
        return [
            neighbor
            for neighbor in self.env.candidate_neighbors(cell)
            if neighbor not in self.kb.visited
            and neighbor not in self.kb.walls
            and neighbor not in self.kb.known_wumpus
        ]

    def _risk_value(self, pos: Position) -> float:
        pit_risk, wumpus_risk, combined = self.kb.risk_score(pos)
        risk = combined * 100.0 + wumpus_risk * 220.0 + pit_risk * 35.0

        if pos in self.kb.known_pits:
            predicted = self._health_after_known_move(self.env.agent.health, pos)
            if predicted is None or predicted <= self._required_health_reserve(pos):
                return 10_000.0
            health_loss = self.env.agent.health - predicted
            risk += 80.0 + health_loss
        if pos in self.kb.known_wumpus:
            return 100_000.0
        if pos in self.kb.safe_cells:
            return 0.0
        return risk

    def _best_boundary(self) -> Position | None:
        """I choose a reachable known cell next to useful unexplored space."""

        allowed = self._allowed_graph()
        boundaries = [cell for cell in allowed if self._unknown_neighbors(cell)]
        if not boundaries:
            return None

        current = self.env.agent.pos

        def boundary_key(cell: Position) -> tuple[float, int, int, Position]:
            unknowns = self._unknown_neighbors(cell)
            minimum_risk = min(self._risk_value(pos) for pos in unknowns)
            information_gain = len(unknowns)
            return (
                minimum_risk,
                manhattan_distance(current, cell),
                -information_gain,
                cell,
            )

        return min(boundaries, key=boundary_key)

    def _safe_action(self) -> Position | None:
        current = self.env.agent.pos

        # After I find the gold, I stop exploring and focus on the known route to the exit.
        if self.env.agent.has_gold:
            path = self._path_to({self.env.config.exit_pos})
            if path and len(path) > 1:
                return path[1]

        safe_frontier = self.kb.safe_cells - self.kb.visited - self.kb.walls
        if safe_frontier:
            path = self._path_to(safe_frontier)
            if path and len(path) > 1:
                return path[1]

        best_boundary = self._best_boundary()
        if best_boundary is not None and current != best_boundary:
            path = self._path_to({best_boundary})
            if path and len(path) > 1:
                return path[1]
        return None

    def _least_risky_action(self) -> Position | None:
        current = self.env.agent.pos
        candidates = [
            pos
            for pos in self.env.candidate_neighbors(current)
            if pos not in self.kb.walls
            and pos not in self.kb.known_wumpus
            and self._known_move_is_viable(self.env.agent.health, pos)
        ]
        if not candidates:
            return None

        unvisited = [pos for pos in candidates if pos not in self.kb.visited]
        if unvisited:
            candidates = unvisited
        else:
            # I avoid a two-cell loop when another move is almost as safe.
            # I still allow backtracking when it is really needed.
            previous = self.env.agent.path[-2] if len(self.env.agent.path) >= 2 else None
            if previous in candidates and len(candidates) > 1:
                previous_risk = self._risk_value(previous)
                alternatives = [pos for pos in candidates if pos != previous]
                acceptable = [
                    pos for pos in alternatives
                    if self._risk_value(pos) <= previous_risk + 20.0
                ]
                if acceptable:
                    candidates = acceptable

        def risk_key(pos: Position) -> tuple[float, int, int, float, int, Position]:
            risk = self._risk_value(pos)
            visits = self.kb.visit_counts.get(pos, 0)
            loop_penalty = visits * visits
            if self.env.agent.has_gold:
                target_distance = float(manhattan_distance(pos, self.env.config.exit_pos))
                # After finding gold, I give progress toward the exit more importance.
                # Strong Wumpus evidence still overrides a shorter route.
                primary_cost = risk + target_distance * 9.0 + loop_penalty * 4.0
            else:
                # Before finding gold, I use the board center only as a tie-breaker.
                # This does not reveal the hidden gold position.
                center = (3.5, 3.5)
                target_distance = (
                    abs(pos[0] - center[0]) + abs(pos[1] - center[1])
                )
                primary_cost = risk
            information_gain = len(self._unknown_neighbors(pos))
            return (
                primary_cost,
                loop_penalty,
                visits,
                target_distance,
                -information_gain,
                pos,
            )

        return min(candidates, key=risk_key)

    def select_action(self) -> Position | None:
        return self._safe_action() or self._least_risky_action()

    def execute(self) -> bool:
        self._observe()

        while not self.env.game_over and self.decisions < self.max_decisions:
            destination = self.select_action()
            if destination is None:
                self.env.termination_reason = "no_survivable_action"
                break

            outcome = self.env.apply_move(destination)
            self.decisions += 1

            if not outcome.valid:
                self.kb.mark_wall(destination)
                continue
            if outcome.event == "wumpus":
                self.kb.mark_wumpus(destination)
                break
            if outcome.event in {"pit", "pit_death"}:
                self.kb.mark_pit(destination)

            if self.env.agent.is_alive:
                self._observe(entered_pit=outcome.event == "pit")

        if (
            not self.env.won
            and self.env.termination_reason is None
            and self.decisions >= self.max_decisions
        ):
            self.env.termination_reason = "max_decisions"
        return self.env.won
