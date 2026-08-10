import argparse
import sys
import time
from pathlib import Path
from collections.abc import Callable, Sequence

from algorithms.genetic import GeneticAlgorithm
from algorithms.logic import LogicBasedAgent
from algorithms.search import AStarSearch
from data.config import (
    GameConfig,
    create_balanced_random_maps,
    create_challenge_map,
    create_default_map,
    create_ga_test_maps,
    create_random_map,
    create_random_test_maps,
    create_test_maps,
)
from models.environment import GameEnvironment
from utils import (
    GameResult,
    compare_benchmark,
    compare_difficulty_levels,
    compare_results,
    format_benchmark_details,
)


def load_config(
    selection: str,
    *,
    random_seed: int = 42,
    random_pits: int = 6,
    random_walls: int = 0,
    initial_health: int = 120,
    gold_score: int = 500,
    pit_penalty: int = 15,
) -> GameConfig:
    if selection == "default":
        return create_default_map()
    if selection == "challenge":
        return create_challenge_map()
    if selection == "random":
        return create_random_map(
            seed=random_seed,
            pit_count=random_pits,
            wall_count=random_walls,
            initial_health=initial_health,
            gold_score=gold_score,
            pit_penalty=pit_penalty,
        )
    return GameConfig.from_file(selection)


def run_algorithm(
    name: str,
    config: GameConfig,
    factory: Callable[[GameEnvironment], object],
) -> GameResult:
    env = GameEnvironment(config)
    algorithm = None
    started = time.perf_counter()
    error = None

    try:
        algorithm = factory(env)
        algorithm.execute()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - started
    training_time = getattr(algorithm, "training_time", 0.0) if algorithm else 0.0
    explicit_evaluation = getattr(algorithm, "evaluation_time", None) if algorithm else None
    execution_only = (
        explicit_evaluation
        if explicit_evaluation is not None
        else max(0.0, elapsed - training_time)
    )

    note = ""
    if isinstance(algorithm, AStarSearch):
        note = f"full-map baseline; expanded={algorithm.expanded_nodes}"
    elif isinstance(algorithm, GeneticAlgorithm):
        summary = algorithm.validation_summary
        if algorithm.loaded_model:
            note = "loaded validated pretrained chromosome"
        elif summary is None:
            note = "trained on separate train/validation maps"
        else:
            note = (
                "trained on 150 maps; selected on 30 validation maps; "
                f"validation_success={summary.success_rate:.1%}"
            )

    return GameResult.from_environment(
        name,
        env,
        execution_time=execution_only,
        training_time=training_time,
        training_trials=getattr(algorithm, "trials_used", 0) if algorithm else 0,
        training_steps=getattr(algorithm, "total_learning_steps", 0) if algorithm else 0,
        training_deaths=getattr(algorithm, "training_deaths", 0) if algorithm else 0,
        training_pit_entries=getattr(algorithm, "training_pit_entries", 0) if algorithm else 0,
        error=error,
        note=note,
    )


def run_all(
    config: GameConfig,
    seed: int,
    ga_model: str | None = None,
) -> list[GameResult]:
    def genetic_factory(env: GameEnvironment) -> GeneticAlgorithm:
        algorithm = GeneticAlgorithm(env, seed=seed)
        if ga_model:
            algorithm.best_agent = GeneticAlgorithm.load_model(ga_model)
            algorithm.loaded_model = True
        return algorithm

    return [
        run_algorithm("A* (offline oracle)", config, lambda env: AStarSearch(env)),
        run_algorithm("Logic-based", config, lambda env: LogicBasedAgent(env)),
        run_algorithm(
            "Genetic policy",
            config,
            genetic_factory,
        ),
    ]


def run_benchmark_configs(
    configs: Sequence[GameConfig],
    seed: int,
    ga_runs: int,
) -> dict[str, list[GameResult]]:
    if ga_runs < 1:
        raise ValueError("ga_runs must be at least 1")
    if not configs:
        raise ValueError("At least one evaluation map is required")

    grouped: dict[str, list[GameResult]] = {
        "A* (offline oracle)": [],
        "Logic-based": [],
        "Genetic policy": [],
    }

    for config in configs:
        grouped["A* (offline oracle)"].append(
            run_algorithm("A* (offline oracle)", config, lambda env: AStarSearch(env))
        )
        grouped["Logic-based"].append(
            run_algorithm("Logic-based", config, lambda env: LogicBasedAgent(env))
        )

    # I train each GA seed once, freeze its chromosome, and test it on every map.
    for run_index in range(ga_runs):
        run_seed = seed + run_index * 9_973
        trainer = GeneticAlgorithm(GameEnvironment(configs[0]), seed=run_seed)
        best = trainer.evolve()
        for map_index, config in enumerate(configs):
            started = time.perf_counter()
            evaluation = trainer.rollout(
                best,
                config,
                seed=900_001 + map_index,
                stochastic=False,
            )
            elapsed = time.perf_counter() - started
            validation = trainer.validation_summary
            validation_text = (
                f"; validation_success={validation.success_rate:.1%}"
                if validation is not None
                else ""
            )
            grouped["Genetic policy"].append(
                GameResult.from_environment(
                    "Genetic policy",
                    evaluation,
                    execution_time=elapsed,
                    training_time=trainer.training_time if map_index == 0 else 0.0,
                    training_trials=trainer.trials_used if map_index == 0 else 0,
                    training_steps=trainer.total_learning_steps if map_index == 0 else 0,
                    training_deaths=trainer.training_deaths if map_index == 0 else 0,
                    training_pit_entries=(
                        trainer.training_pit_entries if map_index == 0 else 0
                    ),
                    note=f"ga_seed={run_seed}{validation_text}",
                )
            )

    return grouped


def run_benchmark(seed: int, ga_runs: int) -> dict[str, list[GameResult]]:
    return run_benchmark_configs(create_test_maps(), seed, ga_runs)


def run_random_benchmark(
    *,
    seed: int,
    ga_runs: int,
    map_count: int,
    map_seed: int,
    pit_count: int,
    wall_count: int,
    initial_health: int,
    gold_score: int,
    pit_penalty: int,
    difficulty: str = "custom",
) -> tuple[dict[str, list[GameResult]], list[GameConfig]]:
    if difficulty == "custom":
        configs = create_random_test_maps(
            map_count,
            seed=map_seed,
            pit_count=pit_count,
            wall_count=wall_count,
            initial_health=initial_health,
            gold_score=gold_score,
            pit_penalty=pit_penalty,
        )
    else:
        difficulties = ("easy", "medium", "hard") if difficulty == "mixed" else (difficulty,)
        configs = create_balanced_random_maps(
            map_count,
            seed=map_seed,
            split_name="benchmark",
            difficulties=difficulties,
        )
    return run_benchmark_configs(configs, seed, ga_runs), configs


def run_ga_test_benchmark(seed: int, ga_runs: int) -> dict[str, list[GameResult]]:
    """I evaluate the methods on 50 maps that were not used for model selection."""

    return run_benchmark_configs(create_ga_test_maps(), seed, ga_runs)


def format_map(config: GameConfig) -> str:
    rows = ["".join(row) for row in config.grid]
    rows.append(
        f"start=({config.start_pos[0] + 1},{config.start_pos[1] + 1}) "
        f"exit=({config.exit_pos[0] + 1},{config.exit_pos[1] + 1})"
    )
    return "\n".join(rows)


def save_random_maps(configs: Sequence[GameConfig], directory: str) -> None:
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = ["file\tmap_name\texit_one_based"]
    for index, config in enumerate(configs, start=1):
        filename = f"random_map_{index:03d}.txt"
        config.to_file(output_dir / filename)
        manifest.append(
            f"{filename}\t{config.name}\t"
            f"({config.exit_pos[0] + 1},{config.exit_pos[1] + 1})"
        )
    (output_dir / "manifest.txt").write_text(
        "\n".join(manifest) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wumpus World AI comparison")
    parser.add_argument(
        "--map",
        default=None,
        help="default, challenge, random, or a path to an input text file",
    )
    parser.add_argument("--seed", type=int, default=42, help="Genetic algorithm seed")
    parser.add_argument(
        "--ga-model",
        help="Load a saved GA chromosome and skip training (useful for a live demo)",
    )
    parser.add_argument(
        "--show-paths",
        action="store_true",
        help="Print the complete path taken by each method",
    )
    benchmark_group = parser.add_mutually_exclusive_group()
    benchmark_group.add_argument(
        "--benchmark",
        action="store_true",
        help="Run all methods on the fixed held-out test maps",
    )
    benchmark_group.add_argument(
        "--random-benchmark",
        action="store_true",
        help="Generate solvable random maps and evaluate all methods on them",
    )
    benchmark_group.add_argument(
        "--ga-test-benchmark",
        action="store_true",
        help="Evaluate all methods on the saved 50-map held-out GA test split",
    )
    parser.add_argument(
        "--ga-runs",
        type=int,
        default=1,
        help="Number of independent GA training seeds",
    )
    parser.add_argument(
        "--random-maps",
        type=int,
        default=20,
        help="Number of maps in random benchmark mode",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Base seed used to generate random maps",
    )
    parser.add_argument(
        "--random-pits",
        type=int,
        default=6,
        help="Number of pits in every random map",
    )
    parser.add_argument(
        "--random-walls",
        type=int,
        default=0,
        help="Number of walls in every random map",
    )
    parser.add_argument(
        "--random-difficulty",
        choices=("custom", "easy", "medium", "hard", "mixed"),
        default="custom",
        help=(
            "Difficulty profile for random benchmark. 'custom' uses --random-pits, "
            "--random-walls, --health and --pit-penalty."
        ),
    )
    parser.add_argument(
        "--health",
        type=int,
        default=120,
        help="Initial health for random maps",
    )
    parser.add_argument(
        "--gold-score",
        type=int,
        default=500,
        help="Gold score for random maps",
    )
    parser.add_argument(
        "--pit-penalty",
        type=int,
        default=15,
        help="Pit penalty for random maps",
    )
    parser.add_argument(
        "--show-random-maps",
        action="store_true",
        help="Print every generated random layout",
    )
    parser.add_argument(
        "--save-random-maps",
        metavar="DIRECTORY",
        help="Save generated random maps as reproducible input files",
    )
    return parser


def choose_interactively() -> str:
    print("1. Default map")
    print("2. Challenge map")
    print("3. Random solvable map")
    print("4. Custom input file")
    choice = input("Select map [1]: ").strip() or "1"
    if choice == "2":
        return "challenge"
    if choice == "3":
        return "random"
    if choice == "4":
        return input("Input-file path: ").strip()
    return "default"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.benchmark or args.random_benchmark or args.ga_test_benchmark:
        try:
            if args.random_benchmark:
                grouped, random_configs = run_random_benchmark(
                    seed=args.seed,
                    ga_runs=args.ga_runs,
                    map_count=args.random_maps,
                    map_seed=args.random_seed,
                    pit_count=args.random_pits,
                    wall_count=args.random_walls,
                    initial_health=args.health,
                    gold_score=args.gold_score,
                    pit_penalty=args.pit_penalty,
                    difficulty=args.random_difficulty,
                )
                if args.random_difficulty == "custom":
                    settings = (
                        f"pits={args.random_pits}, walls={args.random_walls}, "
                        f"health={args.health}, pit_penalty={args.pit_penalty}"
                    )
                else:
                    settings = f"difficulty={args.random_difficulty}"
                print(
                    "RANDOM MAP SETTINGS: "
                    f"maps={args.random_maps}, map_seed={args.random_seed}, "
                    f"wumpus=1, {settings}; only solvable layouts are accepted."
                )
                if args.save_random_maps:
                    save_random_maps(random_configs, args.save_random_maps)
                    print(f"Saved random maps to: {args.save_random_maps}")
                if args.show_random_maps:
                    for config in random_configs:
                        print(f"\n[{config.name}]\n{format_map(config)}")
            elif args.ga_test_benchmark:
                grouped = run_ga_test_benchmark(args.seed, args.ga_runs)
            else:
                grouped = run_benchmark(args.seed, args.ga_runs)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"Input error: {exc}", file=sys.stderr)
            return 2
        print(compare_benchmark(grouped))
        difficulty_report = compare_difficulty_levels(grouped)
        if difficulty_report:
            print(difficulty_report)
        print(format_benchmark_details(grouped))
        return 0

    selection = args.map
    if selection is None:
        selection = choose_interactively() if sys.stdin.isatty() else "default"

    try:
        config = load_config(
            selection,
            random_seed=args.random_seed,
            random_pits=args.random_pits,
            random_walls=args.random_walls,
            initial_health=args.health,
            gold_score=args.gold_score,
            pit_penalty=args.pit_penalty,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2

    print("=" * 120)
    print("WUMPUS WORLD AI SOLVER")
    print("=" * 120)
    print(
        f"Map: {config.name} | internal start={config.start_pos} | "
        f"internal exit={config.exit_pos} | health={config.initial_health} | "
        f"gold score={config.gold_score} | pit penalty={config.pit_penalty}"
    )
    print("Objective: collect the gold and then reach the exit while alive.\n")
    if selection == "random":
        print(format_map(config))
        print()
        if args.save_random_maps:
            save_random_maps([config], args.save_random_maps)
            print(f"Saved random map to: {args.save_random_maps}\n")

    results = run_all(config, args.seed, ga_model=args.ga_model)
    for result in results:
        print(result)
        if args.show_paths:
            print(f"  path: {result.path_taken}")

    print(compare_results(results))
    return 0 if any(result.success for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())