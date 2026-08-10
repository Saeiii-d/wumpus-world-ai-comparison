import json
import random
import time
from dataclasses import dataclass
from collections.abc import Sequence

from data.config import (
    GameConfig,
    create_ga_training_maps,
    create_ga_validation_maps,
)
from models.agent import GENE_NAMES, GeneticAgent
from models.environment import GameEnvironment
from models.percepts import KnowledgeBase
from utils import manhattan_distance

Position = tuple[int, int]


@dataclass(frozen=True)
class ValidationSummary:
    success_rate: float
    gold_rate: float
    death_rate: float
    mean_score: float
    mean_health: float

    @property
    def selection_key(self) -> tuple[float, float, float, float, float]:
        """I rank validation results so success is more important than raw score."""

        return (
            self.success_rate,
            self.gold_rate,
            -self.death_rate,
            self.mean_score,
            self.mean_health,
        )


class GeneticAlgorithm:
    """
    I evolve an online policy with separate train, validation and test data.

    Each chromosome contains understandable heuristic weights. I train on 150
    saved maps, choose the final model on 30 validation maps, and never use the
    held-out test maps for model selection.
    """

    def __init__(
        self,
        env: GameEnvironment,
        *,
        population_size: int = 16,
        generations: int = 12,
        mutation_rate: float = 0.20,
        seed: int = 42,
        max_steps: int = 240,
        training_batch_size: int = 36,
        validation_candidates: int = 5,
        training_configs: Sequence[GameConfig] | None = None,
        validation_configs: Sequence[GameConfig] | None = None,
    ):
        if population_size < 4:
            raise ValueError("Population size must be at least 4")
        if generations < 1:
            raise ValueError("generations must be at least 1")
        if training_batch_size < 1:
            raise ValueError("training_batch_size must be at least 1")
        if validation_candidates < 1:
            raise ValueError("validation_candidates must be at least 1")

        self.env = env
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.max_steps = max_steps
        self.training_batch_size = training_batch_size
        self.validation_candidates = validation_candidates
        self.seed = seed
        self.rng = random.Random(seed)
        self.population = [
            GeneticAgent.random_agent(self.rng)
            for _ in range(population_size)
        ]

        self.training_configs = list(
            training_configs if training_configs is not None else create_ga_training_maps()
        )
        self.validation_configs = list(
            validation_configs
            if validation_configs is not None
            else create_ga_validation_maps()
        )
        if not self.training_configs:
            raise ValueError("At least one GA training map is required")
        if not self.validation_configs:
            raise ValueError("At least one GA validation map is required")

        self.best_agent: GeneticAgent | None = None
        self.loaded_model = False
        self.best_fitness = float("-inf")
        self.validation_summary: ValidationSummary | None = None
        self.training_time = 0.0
        self.evaluation_time = 0.0
        self.trials_used = 0
        self.total_learning_steps = 0
        self.training_deaths = 0
        self.training_pit_entries = 0
        self._active_training_batch: list[GameConfig] | None = None
        self._hall_of_fame: list[tuple[float, GeneticAgent]] = []

    @staticmethod
    def _record_current(
        kb: KnowledgeBase,
        env: GameEnvironment,
        entered_pit: bool = False,
    ) -> None:
        kb.record_visit(
            env.agent.pos,
            env.observe(),
            env.candidate_neighbors(env.agent.pos),
            entered_pit=entered_pit,
        )

    @staticmethod
    def _difficulty(config: GameConfig) -> str:
        lowered = config.name.lower()
        for difficulty in ("easy", "medium", "hard"):
            if difficulty in lowered:
                return difficulty
        return "medium"

    def _rotating_sample(
        self,
        pool: Sequence[GameConfig],
        count: int,
        generation: int,
        offset: int,
    ) -> list[GameConfig]:
        if not pool:
            return []
        start = (generation * count + offset) % len(pool)
        return [pool[(start + index) % len(pool)] for index in range(count)]

    def _curriculum_batch(self, generation: int) -> list[GameConfig]:
        """I start with easy maps, add harder ones, and use all 150 at the end."""

        if generation == self.generations - 1:
            return list(self.training_configs)

        groups: dict[str, list[GameConfig]] = {
            "easy": [],
            "medium": [],
            "hard": [],
        }
        for config in self.training_configs:
            groups[self._difficulty(config)].append(config)

        first_boundary = max(1, self.generations // 3)
        second_boundary = max(first_boundary + 1, 2 * self.generations // 3)

        if generation < first_boundary:
            return self._rotating_sample(
                groups["easy"],
                self.training_batch_size,
                generation,
                0,
            )

        if generation < second_boundary:
            easy_count = self.training_batch_size // 2
            medium_count = self.training_batch_size - easy_count
            return (
                self._rotating_sample(groups["easy"], easy_count, generation, 7)
                + self._rotating_sample(groups["medium"], medium_count, generation, 19)
            )

        easy_count = self.training_batch_size // 3
        medium_count = self.training_batch_size // 3
        hard_count = self.training_batch_size - easy_count - medium_count
        return (
            self._rotating_sample(groups["easy"], easy_count, generation, 11)
            + self._rotating_sample(groups["medium"], medium_count, generation, 23)
            + self._rotating_sample(groups["hard"], hard_count, generation, 37)
        )

    def _nearest_frontier_distance(
        self,
        destination: Position,
        kb: KnowledgeBase,
    ) -> int:
        frontier = kb.safe_cells - kb.visited - kb.walls
        if not frontier:
            return 0
        return min(manhattan_distance(destination, target) for target in frontier)

    def _information_gain(self, destination: Position, env: GameEnvironment, kb: KnowledgeBase) -> int:
        return sum(
            neighbor not in kb.visited and neighbor not in kb.walls
            for neighbor in env.candidate_neighbors(destination)
        )

    def _health_after_known_move(
        self,
        env: GameEnvironment,
        kb: KnowledgeBase,
        destination: Position,
    ) -> int | None:
        remaining = env.agent.health - 1
        if remaining <= 0:
            return None
        if destination in kb.known_pits:
            remaining = (remaining // 2) - env.config.pit_penalty
            if remaining <= 0:
                return None
        return remaining

    def _required_reserve(
        self,
        env: GameEnvironment,
        kb: KnowledgeBase,
        destination: Position,
    ) -> int:
        if env.agent.has_gold:
            return manhattan_distance(destination, env.config.exit_pos) + 2
        frontier = kb.safe_cells - kb.visited - kb.walls
        if frontier:
            nearest = min(manhattan_distance(destination, cell) for cell in frontier)
            return max(5, nearest + 3)
        return 7

    def _action_score(
        self,
        chromosome: GeneticAgent,
        env: GameEnvironment,
        kb: KnowledgeBase,
        destination: Position,
    ) -> float:
        w = chromosome.weights
        score = 0.0

        if destination not in kb.visited:
            score += 16.0 * w["explore"]
        if destination in kb.safe_cells:
            score += 14.0 * w["known_safe"]

        frontier_distance = self._nearest_frontier_distance(destination, kb)
        score -= frontier_distance * 3.0 * w["toward_frontier"]

        if env.agent.has_gold:
            exit_distance = manhattan_distance(destination, env.config.exit_pos)
            score -= exit_distance * 5.0 * w["toward_exit"]

        pit_risk, wumpus_risk, _ = kb.risk_score(destination)
        score -= wumpus_risk * 180.0 * w["avoid_wumpus_risk"]
        score -= pit_risk * 70.0 * w["avoid_pit_risk"]

        visits = kb.visit_counts.get(destination, 0)
        score -= (visits * visits) * 9.0 * w["avoid_revisit"]

        previous = env.agent.path[-2] if len(env.agent.path) >= 2 else None
        if destination == previous:
            score -= 18.0 * w["avoid_backtrack"]

        information_gain = self._information_gain(destination, env, kb)
        score += information_gain * 5.0 * w["information_gain"]

        predicted_health = self._health_after_known_move(env, kb, destination)
        if predicted_health is None:
            return -1_000_000.0
        reserve = self._required_reserve(env, kb, destination)
        reserve_margin = predicted_health - reserve
        score += min(30.0, reserve_margin) * 0.7 * w["health_reserve"]

        if destination in kb.known_pits:
            if predicted_health <= reserve:
                return -500_000.0
            health_loss = env.agent.health - predicted_health
            score -= (80.0 + health_loss * 2.0) * w["avoid_known_pit"]
        if destination in kb.known_wumpus:
            return -1_000_000.0
        return score

    def choose_action(
        self,
        chromosome: GeneticAgent,
        env: GameEnvironment,
        kb: KnowledgeBase,
        rng: random.Random,
        *,
        stochastic: bool,
    ) -> Position | None:
        candidates: list[Position] = []
        for pos in env.candidate_neighbors(env.agent.pos):
            if pos in kb.walls or pos in kb.known_wumpus:
                continue
            predicted = self._health_after_known_move(env, kb, pos)
            if predicted is None:
                continue
            if pos in kb.known_pits and predicted <= self._required_reserve(env, kb, pos):
                continue
            candidates.append(pos)
        if not candidates:
            return None

        scored = [
            (self._action_score(chromosome, env, kb, pos), pos)
            for pos in candidates
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))

        # I avoid undoing the previous move when another action has a close score.
        # Necessary backtracking is still allowed.
        previous = env.agent.path[-2] if len(env.agent.path) >= 2 else None
        if previous is not None and len(scored) > 1 and scored[0][1] == previous:
            alternative = next((item for item in scored if item[1] != previous), None)
            if alternative is not None and alternative[0] >= scored[0][0] - 2.5:
                scored.remove(alternative)
                scored.insert(0, alternative)

        if stochastic and len(scored) > 1 and rng.random() < 0.05:
            top = scored[: min(3, len(scored))]
            weights = [3, 2, 1][: len(top)]
            return rng.choices([item[1] for item in top], weights=weights, k=1)[0]
        return scored[0][1]

    def rollout(
        self,
        chromosome: GeneticAgent,
        config: GameConfig,
        *,
        seed: int,
        stochastic: bool,
    ) -> GameEnvironment:
        env = GameEnvironment(config)
        kb = KnowledgeBase()
        rng = random.Random(seed)
        self._record_current(kb, env)
        decisions = 0

        while not env.game_over and env.agent.steps_taken < self.max_steps:
            destination = self.choose_action(
                chromosome,
                env,
                kb,
                rng,
                stochastic=stochastic,
            )
            if destination is None:
                env.termination_reason = "no_survivable_action"
                break

            outcome = env.apply_move(destination)
            decisions += 1
            if not outcome.valid:
                kb.mark_wall(destination)
                if decisions > self.max_steps * 4:
                    env.termination_reason = "max_decisions"
                    break
                continue
            if outcome.event == "wumpus":
                kb.mark_wumpus(destination)
                break
            if outcome.event in {"pit", "pit_death"}:
                kb.mark_pit(destination)

            if env.agent.is_alive:
                self._record_current(kb, env, entered_pit=outcome.event == "pit")

        if not env.won and env.termination_reason is None:
            env.termination_reason = (
                "max_steps"
                if env.agent.steps_taken >= self.max_steps
                else "stopped"
            )
        return env

    @staticmethod
    def _run_fitness(env: GameEnvironment) -> float:
        unique_cells = len(set(env.agent.path))
        revisits = max(0, len(env.agent.path) - unique_cells)
        fitness = 0.0

        if env.won:
            fitness += 10_000.0 + env.get_final_score() * 5.0
        elif env.agent.has_gold:
            fitness += 2_500.0
        else:
            fitness -= 600.0

        fitness += max(0, env.agent.health) * 4.0
        fitness += min(unique_cells, 32) * 5.0
        fitness -= env.agent.steps_taken * 3.0
        fitness -= revisits * 12.0
        fitness -= env.agent.pit_entries * 250.0

        if env.agent.death_reason == "wumpus":
            fitness -= 6_000.0
        elif not env.agent.is_alive:
            fitness -= 3_000.0
        elif not env.won:
            fitness -= 1_200.0
        return fitness

    def _evaluate_training_batch(
        self,
        chromosome: GeneticAgent,
        configs: Sequence[GameConfig],
    ) -> float:
        values: list[float] = []
        successes = 0
        golds = 0
        deaths = 0

        for map_index, config in enumerate(configs):
            rollout_seed = self.seed + 70_001 + map_index * 101
            run = self.rollout(
                chromosome,
                config,
                seed=rollout_seed,
                stochastic=True,
            )
            values.append(self._run_fitness(run))
            successes += int(run.won)
            golds += int(run.agent.has_gold)
            deaths += int(not run.agent.is_alive)

            self.trials_used += 1
            self.total_learning_steps += run.agent.steps_taken
            self.training_deaths += int(not run.agent.is_alive)
            self.training_pit_entries += run.agent.pit_entries

        count = len(configs)
        average = sum(values) / count
        success_rate = successes / count
        gold_rate = golds / count
        death_rate = deaths / count
        # I make one complete win more valuable than several incomplete episodes.
        return (
            average
            + success_rate * 8_000.0
            + gold_rate * 2_500.0
            - death_rate * 3_000.0
        )

    def fitness(self, chromosome: GeneticAgent, chromosome_index: int = 0) -> float:
        """I evaluate every chromosome on the same active maps and seeds."""

        del chromosome_index
        configs = self._active_training_batch
        if configs is None:
            configs = self.training_configs[: min(12, len(self.training_configs))]
        return self._evaluate_training_batch(chromosome, configs)

    def _validation_metrics(self, chromosome: GeneticAgent) -> ValidationSummary:
        wins = 0
        golds = 0
        deaths = 0
        scores: list[int] = []
        health_values: list[int] = []

        for map_index, config in enumerate(self.validation_configs):
            run = self.rollout(
                chromosome,
                config,
                seed=800_003 + map_index,
                stochastic=False,
            )
            wins += int(run.won)
            golds += int(run.agent.has_gold)
            deaths += int(not run.agent.is_alive)
            scores.append(run.get_final_score())
            health_values.append(max(0, run.agent.health))

        count = len(self.validation_configs)
        return ValidationSummary(
            success_rate=wins / count,
            gold_rate=golds / count,
            death_rate=deaths / count,
            mean_score=sum(scores) / count,
            mean_health=sum(health_values) / count,
        )

    def _tournament(
        self,
        ranked: Sequence[tuple[float, GeneticAgent]],
    ) -> GeneticAgent:
        contestants = self.rng.sample(list(ranked), k=min(4, len(ranked)))
        return max(contestants, key=lambda item: item[0])[1]

    def crossover(
        self,
        first: GeneticAgent,
        second: GeneticAgent,
    ) -> GeneticAgent:
        child = {}
        for name in GENE_NAMES:
            alpha = self.rng.random()
            child[name] = (
                alpha * first.weights[name]
                + (1.0 - alpha) * second.weights[name]
            )
        result = GeneticAgent(child)
        result.normalize()
        return result

    def mutate(self, chromosome: GeneticAgent) -> None:
        for name in GENE_NAMES:
            if self.rng.random() < self.mutation_rate:
                chromosome.weights[name] += self.rng.gauss(0.0, 0.07)
        chromosome.normalize()

    @staticmethod
    def _weight_signature(chromosome: GeneticAgent) -> tuple[float, ...]:
        return tuple(round(chromosome.weights[name], 8) for name in GENE_NAMES)

    def _remember_candidates(
        self,
        ranked: Sequence[tuple[float, GeneticAgent]],
    ) -> None:
        combined = self._hall_of_fame + [
            (fitness, agent.copy())
            for fitness, agent in ranked[: self.validation_candidates]
        ]
        unique: dict[tuple[float, ...], tuple[float, GeneticAgent]] = {}
        for fitness, agent in combined:
            signature = self._weight_signature(agent)
            previous = unique.get(signature)
            if previous is None or fitness > previous[0]:
                unique[signature] = (fitness, agent)
        self._hall_of_fame = sorted(
            unique.values(),
            key=lambda item: item[0],
            reverse=True,
        )[: max(self.validation_candidates * 3, 10)]

    def evolve(self) -> GeneticAgent:
        started = time.perf_counter()

        for generation in range(self.generations):
            self._active_training_batch = self._curriculum_batch(generation)
            ranked = [
                (self.fitness(chromosome, index), chromosome)
                for index, chromosome in enumerate(self.population)
            ]
            ranked.sort(key=lambda item: item[0], reverse=True)
            self._remember_candidates(ranked)

            if ranked[0][0] > self.best_fitness:
                self.best_fitness = ranked[0][0]

            if generation == self.generations - 1:
                break

            elite_count = max(2, self.population_size // 6)
            new_population = [
                agent.copy()
                for _, agent in ranked[:elite_count]
            ]
            while len(new_population) < self.population_size:
                parent_a = self._tournament(ranked)
                parent_b = self._tournament(ranked)
                child = self.crossover(parent_a, parent_b)
                self.mutate(child)
                new_population.append(child)
            self.population = new_population

        # I take strong training candidates, then choose the final one on validation data.
        candidates = [agent for _, agent in self._hall_of_fame]
        if not candidates:
            raise RuntimeError("Evolution did not produce validation candidates")

        selected_agent: GeneticAgent | None = None
        selected_summary: ValidationSummary | None = None
        for candidate in candidates[: max(self.validation_candidates, 1)]:
            summary = self._validation_metrics(candidate)
            if selected_summary is None or summary.selection_key > selected_summary.selection_key:
                selected_agent = candidate.copy()
                selected_summary = summary

        if selected_agent is None or selected_summary is None:
            raise RuntimeError("Validation did not select a chromosome")

        self.best_agent = selected_agent
        self.validation_summary = selected_summary
        self.training_time = time.perf_counter() - started
        self._active_training_batch = None
        return self.best_agent

    @staticmethod
    def load_model(path: str) -> GeneticAgent:
        """I load a validated chromosome so I can run the presentation quickly."""

        with open(path, "r", encoding="utf-8") as model_file:
            payload = json.load(model_file)
        weights = payload.get("weights", payload)
        missing = [name for name in GENE_NAMES if name not in weights]
        if missing:
            raise ValueError(f"GA model is missing weights: {missing}")
        agent = GeneticAgent({name: float(weights[name]) for name in GENE_NAMES})
        agent.normalize()
        return agent

    def save_model(self, path: str) -> None:
        if self.best_agent is None:
            raise RuntimeError("No trained chromosome is available to save")
        payload = {
            "seed": self.seed,
            "genes": list(GENE_NAMES),
            "weights": self.best_agent.weights,
            "validation": (
                None
                if self.validation_summary is None
                else {
                    "success_rate": self.validation_summary.success_rate,
                    "gold_rate": self.validation_summary.gold_rate,
                    "death_rate": self.validation_summary.death_rate,
                    "mean_score": self.validation_summary.mean_score,
                    "mean_health": self.validation_summary.mean_health,
                }
            ),
        }
        with open(path, "w", encoding="utf-8") as model_file:
            json.dump(payload, model_file, indent=2, sort_keys=True)
            model_file.write("\n")

    def execute(self) -> bool:
        # I may preload ``best_agent`` for a quick demo. In a normal run I train it here.
        best = self.best_agent.copy() if self.best_agent is not None else self.evolve()
        started = time.perf_counter()
        final_run = self.rollout(
            best,
            self.env.config,
            seed=999_983,
            stochastic=False,
        )
        self.evaluation_time = time.perf_counter() - started
        self.env.adopt_state(final_run)
        return self.env.won