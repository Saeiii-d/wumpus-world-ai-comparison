from dataclasses import dataclass, field
from statistics import mean
from collections.abc import Sequence

Position = tuple[int, int]


def manhattan_distance(first: Position, second: Position) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


@dataclass
class GameResult:
    algorithm_name: str
    map_name: str = ""
    success: bool = False
    final_score: int = 0
    steps_taken: int = 0
    health_remaining: int = 0
    has_gold: bool = False
    is_alive: bool = True
    pit_entries: int = 0
    death_reason: str | None = None
    termination_reason: str | None = None
    execution_time: float = 0.0
    training_time: float = 0.0
    training_trials: int = 0
    training_steps: int = 0
    training_deaths: int = 0
    training_pit_entries: int = 0
    path_taken: list[Position] = field(default_factory=list)
    error: str | None = None
    note: str = ""

    @property
    def status(self) -> str:
        if self.error:
            return "ERROR"
        return "SUCCESS" if self.success else "FAILED"

    @classmethod
    def from_environment(
        cls,
        algorithm_name: str,
        env,
        *,
        execution_time: float,
        training_time: float = 0.0,
        training_trials: int = 0,
        training_steps: int = 0,
        training_deaths: int = 0,
        training_pit_entries: int = 0,
        error: str | None = None,
        note: str = "",
    ) -> "GameResult":
        return cls(
            algorithm_name=algorithm_name,
            map_name=env.config.name,
            success=env.won,
            final_score=env.get_final_score(),
            steps_taken=env.agent.steps_taken,
            health_remaining=env.agent.health,
            has_gold=env.agent.has_gold,
            is_alive=env.agent.is_alive,
            pit_entries=env.agent.pit_entries,
            death_reason=env.agent.death_reason,
            termination_reason=getattr(env, "termination_reason", None),
            execution_time=execution_time,
            training_time=training_time,
            training_trials=training_trials,
            training_steps=training_steps,
            training_deaths=training_deaths,
            training_pit_entries=training_pit_entries,
            path_taken=env.agent.path[:],
            error=error,
            note=note,
        )

    def __str__(self) -> str:
        reason = self.error or self.termination_reason or self.death_reason or self.note or "-"
        return (
            f"{self.algorithm_name:<30} | {self.status:<7} | "
            f"score={self.final_score:<4} | steps={self.steps_taken:<3} | "
            f"terminal_health={self.health_remaining:<4} | gold={'yes' if self.has_gold else 'no ':<3} | "
            f"pits={self.pit_entries:<2} | run={self.execution_time:.5f}s | "
            f"train={self.training_time:.5f}s | {reason}"
        )


@dataclass(frozen=True)
class BenchmarkSummary:
    algorithm_name: str
    runs: int
    success_rate: float
    gold_collection_rate: float
    death_rate: float
    wumpus_death_rate: float
    pit_death_rate: float
    mean_score: float
    mean_health: float
    mean_steps: float
    mean_pit_entries: float
    mean_runtime: float
    total_training_time: float
    total_training_trials: int
    total_training_steps: int
    total_training_deaths: int

    @classmethod
    def from_results(cls, name: str, results: Sequence[GameResult]) -> "BenchmarkSummary":
        if not results:
            raise ValueError("Cannot summarize an empty result list")
        count = len(results)
        return cls(
            algorithm_name=name,
            runs=count,
            success_rate=sum(result.success for result in results) / count,
            gold_collection_rate=sum(result.has_gold for result in results) / count,
            death_rate=sum(not result.is_alive for result in results) / count,
            wumpus_death_rate=sum(result.death_reason == "wumpus" for result in results) / count,
            pit_death_rate=sum(result.death_reason == "pit" for result in results) / count,
            mean_score=mean(result.final_score for result in results),
            mean_health=mean(result.health_remaining for result in results),
            mean_steps=mean(result.steps_taken for result in results),
            mean_pit_entries=mean(result.pit_entries for result in results),
            mean_runtime=mean(result.execution_time for result in results),
            total_training_time=sum(result.training_time for result in results),
            total_training_trials=sum(result.training_trials for result in results),
            total_training_steps=sum(result.training_steps for result in results),
            total_training_deaths=sum(result.training_deaths for result in results),
        )


def compare_results(results: Sequence[GameResult]) -> str:
    lines = ["", "SINGLE-MAP ANALYSIS", "-" * 112]
    successful = [result for result in results if result.success]
    errors = [result for result in results if result.error]
    lines.append(f"Successful methods: {len(successful)}/{len(results)}; runtime errors: {len(errors)}")

    if successful:
        best_score = max(successful, key=lambda item: item.final_score)
        fewest_steps = min(successful, key=lambda item: item.steps_taken)
        lines.append(
            f"Best score: {best_score.algorithm_name} ({best_score.final_score}); "
            f"fewest successful steps: {fewest_steps.algorithm_name} "
            f"({fewest_steps.steps_taken})."
        )
    else:
        lines.append("No method completed the full objective (gold followed by exit).")

    for result in results:
        if result.error:
            explanation = f"runtime error: {result.error}"
        elif result.success:
            explanation = (
                f"completed the episode with {result.health_remaining} health and "
                f"{result.pit_entries} pit entries"
            )
        elif not result.is_alive:
            explanation = f"died because of {result.death_reason}"
        elif result.has_gold:
            explanation = (
                "found the gold but did not reach the exit "
                f"({result.termination_reason or 'stopped'})"
            )
        else:
            explanation = (
                "stopped before collecting the gold "
                f"({result.termination_reason or 'stopped'})"
            )
        lines.append(f"- {result.algorithm_name}: {explanation}.")

    lines.append(
        "Fairness note: A* is a full-map oracle baseline, while the logic-based "
        "agent and the genetic policy act under partial observability."
    )
    return "\n".join(lines)


def compare_benchmark(grouped_results: dict[str, Sequence[GameResult]]) -> str:
    summaries = [
        BenchmarkSummary.from_results(name, results)
        for name, results in grouped_results.items()
    ]
    lines = [
        "EVALUATION BENCHMARK (DISJOINT FROM GA TRAINING)",
        "=" * 164,
        (
            f"{'Algorithm':<30} {'Runs':>5} {'Success':>9} {'Gold':>8} "
            f"{'Death':>8} {'W-death':>9} {'P-death':>9} {'Score':>9} "
            f"{'Terminal H':>10} {'Steps':>9} {'Pits':>7} "
            f"{'Mean Run(s)':>12} {'Total Train(s)':>14}"
        ),
        "-" * 164,
    ]
    for summary in summaries:
        lines.append(
            f"{summary.algorithm_name:<30} {summary.runs:>5} "
            f"{summary.success_rate:>8.1%} {summary.gold_collection_rate:>7.1%} "
            f"{summary.death_rate:>7.1%} {summary.wumpus_death_rate:>8.1%} "
            f"{summary.pit_death_rate:>8.1%} {summary.mean_score:>9.2f} "
            f"{summary.mean_health:>10.2f} {summary.mean_steps:>9.2f} "
            f"{summary.mean_pit_entries:>7.2f} {summary.mean_runtime:>12.5f} "
            f"{summary.total_training_time:>14.5f}"
        )

    best = max(summaries, key=lambda item: (item.success_rate, item.mean_score))
    partially_observable = [
        summary
        for summary in summaries
        if not summary.algorithm_name.startswith("A*")
    ]

    lines.append(
        f"\nBest overall result by success rate then score: {best.algorithm_name} "
        f"({best.success_rate:.1%}, mean score {best.mean_score:.2f})."
    )
    if partially_observable:
        best_partial = max(
            partially_observable,
            key=lambda item: (item.success_rate, item.mean_score),
        )
        lines.append(
            f"Best partially observable method: {best_partial.algorithm_name} "
            f"({best_partial.success_rate:.1%}, mean score {best_partial.mean_score:.2f})."
        )
    else:
        lines.append("No partially observable method was included in this benchmark.")
    map_names = {
        result.map_name
        for results in grouped_results.values()
        for result in results
    }
    if map_names and all(name.startswith("Random ") for name in map_names):
        evaluation_note = (
            "All evaluation layouts were generated from reported seeds, contain exactly one "
            "Wumpus, and were accepted only when the full objective was physically solvable."
        )
    else:
        evaluation_note = (
            "Default and Challenge are public project maps; the remaining four are additional "
            "held-out evaluation layouts."
        )
    lines.append(
        "A* reads the complete map and is an oracle/upper-bound baseline. "
        "The GA uses 150 training maps and 30 separate validation maps; every reported "
        f"test layout is held out from model selection. {evaluation_note}"
    )
    return "\n".join(lines)


def format_benchmark_details(grouped_results: dict[str, Sequence[GameResult]]) -> str:
    """I return every map result so the aggregate averages can be checked."""

    lines = [
        "",
        "PER-MAP RESULTS",
        "=" * 166,
        (
            f"{'Map':<36} {'Algorithm':<30} {'Status':<8} {'Score':>7} "
            f"{'Terminal H':>10} {'Steps':>7} {'Pits':>6} "
            f"{'Termination':<20} {'Variant':<16} {'Run(s)':>10}"
        ),
        "-" * 166,
    ]
    ordered = []
    for algorithm_name, results in grouped_results.items():
        for result in results:
            ordered.append((result.map_name, algorithm_name, result))
    ordered.sort(key=lambda item: (item[0], item[1], item[2].note))

    for map_name, algorithm_name, result in ordered:
        reason = result.error or result.termination_reason or result.death_reason or "-"
        variant = result.note if result.note.startswith("ga_seed=") else "-"
        lines.append(
            f"{map_name:<36} {algorithm_name:<30} {result.status:<8} "
            f"{result.final_score:>7} {result.health_remaining:>10} "
            f"{result.steps_taken:>7} {result.pit_entries:>6} "
            f"{reason:<20} {variant:<16} {result.execution_time:>10.5f}"
        )
    return "\n".join(lines)


def _difficulty_from_map_name(name: str) -> str | None:
    lowered = name.lower()
    for difficulty in ("easy", "medium", "hard"):
        if difficulty in lowered:
            return difficulty
    return None


def compare_difficulty_levels(
    grouped_results: dict[str, Sequence[GameResult]],
) -> str:
    """I summarize the benchmark separately for easy, medium and hard maps."""

    rows = []
    for algorithm_name, results in grouped_results.items():
        for difficulty in ("easy", "medium", "hard"):
            selected = [
                result
                for result in results
                if _difficulty_from_map_name(result.map_name) == difficulty
            ]
            if selected:
                rows.append(
                    (difficulty, BenchmarkSummary.from_results(algorithm_name, selected))
                )

    if not rows:
        return ""

    lines = [
        "",
        "RESULTS BY DIFFICULTY",
        "=" * 102,
        (
            f"{'Difficulty':<12} {'Algorithm':<30} {'Runs':>5} "
            f"{'Success':>9} {'Gold':>8} {'Death':>8} "
            f"{'Score':>9} {'Steps':>9}"
        ),
        "-" * 102,
    ]
    order = {"easy": 0, "medium": 1, "hard": 2}
    rows.sort(key=lambda item: (order[item[0]], item[1].algorithm_name))
    for difficulty, summary in rows:
        lines.append(
            f"{difficulty.title():<12} {summary.algorithm_name:<30} "
            f"{summary.runs:>5} {summary.success_rate:>8.1%} "
            f"{summary.gold_collection_rate:>7.1%} {summary.death_rate:>7.1%} "
            f"{summary.mean_score:>9.2f} {summary.mean_steps:>9.2f}"
        )
    return "\n".join(lines)
