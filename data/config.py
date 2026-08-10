from dataclasses import dataclass
from collections import deque
from pathlib import Path
import random
from collections.abc import Iterable, Sequence

Position = tuple[int, int]
GRID_SIZE = 8
VALID_SYMBOLS = {"*", "W", "P", "G", "D"}


@dataclass
class GameConfig:
    """
    I keep all settings of one game in this class.

    Inside the code I use zero-based coordinates, so ``(0, 0)`` is the top-left
    cell. In map files I use one-based coordinates as required by the project.
    """

    grid: list[list[str]]
    initial_health: int
    gold_score: int
    pit_penalty: int
    exit_pos: Position
    start_pos: Position = (0, 0)
    name: str = "Custom Map"

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if len(self.grid) != GRID_SIZE or any(len(row) != GRID_SIZE for row in self.grid):
            raise ValueError(f"Grid must be exactly {GRID_SIZE}x{GRID_SIZE}")

        invalid = {
            symbol
            for row in self.grid
            for symbol in row
            if symbol not in VALID_SYMBOLS
        }
        if invalid:
            raise ValueError(f"Invalid map symbols: {sorted(invalid)}")

        if self.initial_health <= 0:
            raise ValueError("Initial health must be positive")
        if self.gold_score < 0:
            raise ValueError("Gold score cannot be negative")
        if self.pit_penalty < 0:
            raise ValueError("Pit penalty cannot be negative")

        self._validate_position(self.start_pos, "Start")
        self._validate_position(self.exit_pos, "Exit")

        if self.cell(self.start_pos) in {"D", "W", "P"}:
            raise ValueError("Start position must be an empty or gold cell")
        if self.cell(self.exit_pos) in {"D", "W", "P"}:
            raise ValueError("Exit position cannot contain a wall, Wumpus, or pit")

        gold_count = sum(cell == "G" for row in self.grid for cell in row)
        if gold_count != 1:
            raise ValueError(f"The map must contain exactly one gold cell; found {gold_count}")

    @staticmethod
    def _validate_position(pos: Position, label: str) -> None:
        row, col = pos
        if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
            raise ValueError(f"{label} position {pos} is outside the grid")

    def cell(self, pos: Position) -> str:
        row, col = pos
        return self.grid[row][col]

    @property
    def gold_pos(self) -> Position:
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if self.grid[row][col] == "G":
                    return row, col
        raise RuntimeError("Validated map unexpectedly has no gold")

    def copy(self) -> "GameConfig":
        return GameConfig(
            grid=[row[:] for row in self.grid],
            initial_health=self.initial_health,
            gold_score=self.gold_score,
            pit_penalty=self.pit_penalty,
            exit_pos=self.exit_pos,
            start_pos=self.start_pos,
            name=self.name,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "GameConfig":
        """I read one map file and convert it to a ``GameConfig``."""

        file_path = Path(path)
        raw_lines = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines()]
        lines = [line for line in raw_lines if line]
        if len(lines) != 12:
            raise ValueError(
                f"Expected 12 non-empty lines (8 map rows + 4 parameters), found {len(lines)}"
            )

        grid = [list(line) for line in lines[:GRID_SIZE]]
        initial_health = int(lines[8])
        gold_score = int(lines[9])
        pit_penalty = int(lines[10])

        exit_text = lines[11].replace("(", "").replace(")", "").replace(",", " ")
        parts = exit_text.split()
        if len(parts) != 2:
            raise ValueError("Exit position must contain exactly two integers")
        file_row, file_col = int(parts[0]), int(parts[1])
        if not (1 <= file_row <= GRID_SIZE and 1 <= file_col <= GRID_SIZE):
            raise ValueError(
                f"Exit file coordinates must be between 1 and {GRID_SIZE}; "
                f"received ({file_row}, {file_col})"
            )
        exit_pos = file_row - 1, file_col - 1

        return cls(
            grid=grid,
            initial_health=initial_health,
            gold_score=gold_score,
            pit_penalty=pit_penalty,
            exit_pos=exit_pos,
            name=file_path.stem,
        )

    def to_file(self, path: str | Path) -> None:
        rows = ["".join(row) for row in self.grid]
        payload = rows + [
            str(self.initial_health),
            str(self.gold_score),
            str(self.pit_penalty),
            f"{self.exit_pos[0] + 1} {self.exit_pos[1] + 1}",
        ]
        Path(path).write_text("\n".join(payload) + "\n", encoding="utf-8")


def _config_from_rows(
    rows: Sequence[str],
    *,
    name: str,
    initial_health: int,
    gold_score: int,
    pit_penalty: int,
    exit_pos: Position = (7, 7),
) -> GameConfig:
    return GameConfig(
        grid=[list(row) for row in rows],
        initial_health=initial_health,
        gold_score=gold_score,
        pit_penalty=pit_penalty,
        exit_pos=exit_pos,
        name=name,
    )


def create_default_map() -> GameConfig:
    """I use this as the default map with a safe route and risky shortcuts."""

    return _config_from_rows(
        [
            "*DP***W*",
            "****P***",
            "D******P",
            "*P**D***",
            "***G*D**",
            "P******D",
            "**D*P***",
            "********",
        ],
        name="Default Map",
        initial_health=100,
        gold_score=500,
        pit_penalty=15,
    )


def create_challenge_map() -> GameConfig:
    """I use this harder map while keeping at least one safe solution."""

    return _config_from_rows(
        [
            "**PW*P**",
            "**D*****",
            "*****G*W",
            "*D**D**P",
            "P*****D*",
            "**P*W***",
            "*D****P*",
            "***P****",
        ],
        name="Challenge Map",
        initial_health=140,
        gold_score=500,
        pit_penalty=20,
    )


def create_training_maps() -> list[GameConfig]:
    """I keep these small fixed maps only for compatibility and basic training.

    I do not reuse any of them in ``create_test_maps``.
    """

    definitions = [
        (
            "Training A",
            [
                "********",
                "**D*****",
                "****P***",
                "***G****",
                "*****D**",
                "*P******",
                "******W*",
                "********",
            ],
            110,
            12,
        ),
        (
            "Training B",
            [
                "***D****",
                "********",
                "*P***W**",
                "***D****",
                "*****G**",
                "**P*****",
                "****D***",
                "********",
            ],
            120,
            15,
        ),
        (
            "Training C",
            [
                "**P*****",
                "****D***",
                "*D******",
                "*****W**",
                "**G*****",
                "****P***",
                "*D******",
                "********",
            ],
            115,
            13,
        ),
        (
            "Training D",
            [
                "********",
                "*P**D***",
                "***W****",
                "D*******",
                "*****G**",
                "**D*****",
                "****P***",
                "********",
            ],
            130,
            17,
        ),
        (
            "Training E",
            [
                "***D****",
                "*P******",
                "*****W**",
                "**D*****",
                "***G****",
                "******P*",
                "D*******",
                "********",
            ],
            125,
            14,
        ),
        (
            "Training Required Pit",
            [
                "*PG*****",
                "DDD*****",
                "********",
                "********",
                "********",
                "********",
                "********",
                "********",
            ],
            100,
            5,
        ),
    ]

    return [
        _config_from_rows(
            rows,
            name=name,
            initial_health=health,
            gold_score=500,
            pit_penalty=pit_penalty,
        )
        for name, rows, health, pit_penalty in definitions
    ]

def create_test_maps() -> list[GameConfig]:
    """I keep these fixed maps aside for a fair multi-map evaluation.

    None of the extra layouts is used in ``create_training_maps``. I include
    the public default and challenge maps and add unseen hazard arrangements.
    """

    return [
        create_default_map(),
        create_challenge_map(),
        _config_from_rows(
            [
                "********",
                "*D*P****",
                "***D****",
                "**G*****",
                "****W***",
                "*P******",
                "*****D**",
                "********",
            ],
            name="Held-out A",
            initial_health=115,
            gold_score=500,
            pit_penalty=12,
        ),
        _config_from_rows(
            [
                "**D*****",
                "****P***",
                "*W******",
                "***D*G**",
                "P*******",
                "****D***",
                "**P*****",
                "********",
            ],
            name="Held-out B",
            initial_health=125,
            gold_score=500,
            pit_penalty=16,
        ),
        _config_from_rows(
            [
                "********",
                "**P*D***",
                "****W***",
                "*D******",
                "****G***",
                "***P****",
                "D*****P*",
                "********",
            ],
            name="Held-out C",
            initial_health=135,
            gold_score=500,
            pit_penalty=18,
        ),
        _config_from_rows(
            [
                "*P*G****",
                "D*D*****",
                "********",
                "***W****",
                "*P***D**",
                "********",
                "****P***",
                "********",
            ],
            name="Held-out Pit Route",
            initial_health=150,
            gold_score=500,
            pit_penalty=10,
        ),
    ]



def _has_solvable_objective(config: GameConfig) -> bool:
    """I check whether at least one legal route can collect gold and exit.

    I use this only while generating random maps. It is separate from the three
    agents, so it does not give them a hidden solution.
    """

    start_has_gold = config.start_pos == config.gold_pos
    frontier = deque([(config.start_pos, start_has_gold, config.initial_health)])
    best_health: dict[tuple[Position, bool], int] = {
        (config.start_pos, start_has_gold): config.initial_health
    }

    while frontier:
        position, has_gold, health = frontier.popleft()
        if position == config.exit_pos and has_gold:
            return True

        row, col = position
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            destination = row + dr, col + dc
            if not (0 <= destination[0] < GRID_SIZE and 0 <= destination[1] < GRID_SIZE):
                continue

            cell = config.cell(destination)
            if cell in {"D", "W"}:
                continue

            next_health = health - 1
            if next_health <= 0:
                continue
            if cell == "P":
                next_health = (next_health // 2) - config.pit_penalty
                if next_health <= 0:
                    continue

            next_has_gold = has_gold or cell == "G"
            key = destination, next_has_gold
            if next_health <= best_health.get(key, -1):
                continue
            best_health[key] = next_health
            frontier.append((destination, next_has_gold, next_health))

    return False


def create_random_map(
    *,
    seed: int,
    pit_count: int = 6,
    wall_count: int = 0,
    initial_health: int = 120,
    gold_score: int = 500,
    pit_penalty: int = 15,
    max_attempts: int = 2_000,
    name: str | None = None,
) -> GameConfig:
    """I create one reproducible and solvable random 8x8 test map.

    I place the start, exit, gold, one Wumpus, pits and walls in separate cells.
    I reject each layout until a legal route can collect the gold and exit alive.
    """

    if pit_count < 0:
        raise ValueError("pit_count cannot be negative")
    if wall_count < 0:
        raise ValueError("wall_count cannot be negative")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    # I reserve separate cells for the exit, gold, Wumpus, pits and walls.
    # The exit itself is stored separately and has no grid symbol.
    required_non_start_cells = 3 + pit_count + wall_count
    available_non_start_cells = GRID_SIZE * GRID_SIZE - 1
    if required_non_start_cells > available_non_start_cells:
        raise ValueError(
            "Too many requested objects: exit, gold, one Wumpus, pits, and "
            "walls must fit in distinct non-start cells"
        )

    rng = random.Random(seed)
    start_pos = (0, 0)
    cells = [
        (row, col)
        for row in range(GRID_SIZE)
        for col in range(GRID_SIZE)
        if (row, col) != start_pos
    ]

    for attempt in range(1, max_attempts + 1):
        selected = rng.sample(cells, required_non_start_cells)
        exit_pos = selected[0]
        gold_pos = selected[1]
        wumpus_pos = selected[2]
        pit_positions = set(selected[3 : 3 + pit_count])
        wall_positions = set(selected[3 + pit_count :])

        grid = [["*" for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        grid[gold_pos[0]][gold_pos[1]] = "G"
        grid[wumpus_pos[0]][wumpus_pos[1]] = "W"
        for row, col in pit_positions:
            grid[row][col] = "P"
        for row, col in wall_positions:
            grid[row][col] = "D"

        config = GameConfig(
            grid=grid,
            initial_health=initial_health,
            gold_score=gold_score,
            pit_penalty=pit_penalty,
            exit_pos=exit_pos,
            start_pos=start_pos,
            name=name or f"Random Map seed={seed}",
        )
        if _has_solvable_objective(config):
            return config

    raise RuntimeError(
        f"Could not generate a solvable random map after {max_attempts} attempts. "
        "Reduce hazards/walls or increase initial health."
    )


def create_random_test_maps(
    count: int,
    *,
    seed: int = 42,
    pit_count: int = 6,
    wall_count: int = 0,
    initial_health: int = 120,
    gold_score: int = 500,
    pit_penalty: int = 15,
) -> list[GameConfig]:
    """I create unique and reproducible random maps for aggregate tests."""

    if count < 1:
        raise ValueError("count must be at least 1")

    maps: list[GameConfig] = []
    layouts: set[tuple[tuple[str, ...], Position]] = set()
    # I keep temporary random benchmarks separate from the GA development data.
    forbidden_sources = list(create_training_maps())
    try:
        forbidden_sources.extend(create_ga_training_maps())
        forbidden_sources.extend(create_ga_validation_maps())
    except NameError:
        # I use this fallback only if the later helper functions are not defined yet.
        pass
    forbidden_grids = {
        tuple("".join(row) for row in config.grid)
        for config in forbidden_sources
    }
    candidate_index = 0
    max_candidates = max(100, count * 50)

    while len(maps) < count and candidate_index < max_candidates:
        map_seed = seed + candidate_index * 1_000_003
        config = create_random_map(
            seed=map_seed,
            pit_count=pit_count,
            wall_count=wall_count,
            initial_health=initial_health,
            gold_score=gold_score,
            pit_penalty=pit_penalty,
            name=f"Random {len(maps) + 1:03d} seed={map_seed}",
        )
        grid_signature = tuple("".join(row) for row in config.grid)
        signature = grid_signature, config.exit_pos
        candidate_index += 1
        if grid_signature in forbidden_grids or signature in layouts:
            continue
        layouts.add(signature)
        maps.append(config)

    if len(maps) != count:
        raise RuntimeError(f"Could create only {len(maps)} unique random maps")
    return maps


# I define the reproducible random datasets for the genetic algorithm here.

DIFFICULTY_PROFILES = {
    "easy": {
        "pit_range": (2, 3),
        "wall_range": (0, 1),
        "initial_health": 115,
        "pit_penalty": 10,
    },
    "medium": {
        "pit_range": (4, 6),
        "wall_range": (2, 4),
        "initial_health": 125,
        "pit_penalty": 15,
    },
    "hard": {
        "pit_range": (7, 10),
        "wall_range": (4, 8),
        "initial_health": 145,
        "pit_penalty": 18,
    },
}


def map_signature(config: GameConfig) -> tuple[tuple[str, ...], Position]:
    """I build a stable signature to detect overlap between dataset splits."""

    return tuple("".join(row) for row in config.grid), config.exit_pos


def create_difficulty_map(
    *,
    seed: int,
    difficulty: str,
    gold_score: int = 500,
    name: str | None = None,
) -> GameConfig:
    """I create one solvable map from the selected difficulty profile."""

    if difficulty not in DIFFICULTY_PROFILES:
        raise ValueError(
            f"Unknown difficulty {difficulty!r}; expected one of "
            f"{sorted(DIFFICULTY_PROFILES)}"
        )

    profile = DIFFICULTY_PROFILES[difficulty]
    rng = random.Random(seed ^ 0x5F3759DF)
    pit_count = rng.randint(*profile["pit_range"])
    wall_count = rng.randint(*profile["wall_range"])
    return create_random_map(
        seed=seed,
        pit_count=pit_count,
        wall_count=wall_count,
        initial_health=profile["initial_health"],
        gold_score=gold_score,
        pit_penalty=profile["pit_penalty"],
        name=name or f"{difficulty.title()} Random seed={seed}",
    )


def create_balanced_random_maps(
    count: int,
    *,
    seed: int,
    split_name: str,
    difficulties: Sequence[str] = ("easy", "medium", "hard"),
    forbidden: Iterable[tuple[tuple[str, ...], Position]] = (),
) -> list[GameConfig]:
    """I create a unique, balanced and reproducible collection of random maps.

    I assign difficulties in round-robin order and use ``forbidden`` signatures
    to keep training, validation and test sets separate.
    """

    if count < 1:
        raise ValueError("count must be at least 1")
    if not difficulties:
        raise ValueError("At least one difficulty is required")
    for difficulty in difficulties:
        if difficulty not in DIFFICULTY_PROFILES:
            raise ValueError(f"Unsupported difficulty: {difficulty}")

    blocked = set(forbidden)
    seen = set(blocked)
    configs: list[GameConfig] = []
    candidate_index = 0
    max_candidates = max(500, count * 100)

    while len(configs) < count and candidate_index < max_candidates:
        difficulty = difficulties[len(configs) % len(difficulties)]
        map_seed = seed + candidate_index * 1_000_003
        candidate_index += 1
        config = create_difficulty_map(
            seed=map_seed,
            difficulty=difficulty,
            name=(
                f"{split_name}-{len(configs) + 1:03d}-"
                f"{difficulty}-seed={map_seed}"
            ),
        )
        signature = map_signature(config)
        if signature in seen:
            continue
        seen.add(signature)
        configs.append(config)

    if len(configs) != count:
        raise RuntimeError(
            f"Could create only {len(configs)} unique maps for {split_name}"
        )
    return configs


def save_config_dataset(configs: Sequence[GameConfig], directory: str | Path) -> None:
    """I save one dataset split and its small reproducibility manifest."""

    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob("*.txt"):
        old_file.unlink()

    manifest = ["file\tmap_name\tdifficulty\texit_one_based"]
    for index, config in enumerate(configs, start=1):
        lowered = config.name.lower()
        difficulty = next(
            (item for item in DIFFICULTY_PROFILES if item in lowered),
            "custom",
        )
        filename = f"map_{index:03d}_{difficulty}.txt"
        config.to_file(output_dir / filename)
        manifest.append(
            f"{filename}\t{config.name}\t{difficulty}\t"
            f"({config.exit_pos[0] + 1},{config.exit_pos[1] + 1})"
        )
    (output_dir / "manifest.tsv").write_text(
        "\n".join(manifest) + "\n",
        encoding="utf-8",
    )


def load_config_dataset(directory: str | Path) -> list[GameConfig]:
    """I load all map files from a dataset directory in numeric order."""

    directory_path = Path(directory)
    if not directory_path.is_dir():
        return []
    return [
        GameConfig.from_file(path)
        for path in sorted(directory_path.glob("map_*.txt"))
    ]


def _ga_dataset_root() -> Path:
    return Path(__file__).resolve().parent / "ga_datasets"


def create_ga_dataset_splits(
    *,
    training_count: int = 150,
    validation_count: int = 30,
    test_count: int = 50,
    seed: int = 403_243_046,
) -> tuple[list[GameConfig], list[GameConfig], list[GameConfig]]:
    """I generate separate training, validation and test maps for the GA."""

    training = create_balanced_random_maps(
        training_count,
        seed=seed,
        split_name="train",
    )
    used = {map_signature(config) for config in training}
    validation = create_balanced_random_maps(
        validation_count,
        seed=seed + 100_000_007,
        split_name="validation",
        forbidden=used,
    )
    used.update(map_signature(config) for config in validation)
    test = create_balanced_random_maps(
        test_count,
        seed=seed + 200_000_033,
        split_name="test",
        forbidden=used,
    )
    return training, validation, test


def materialize_ga_datasets(
    root: str | Path | None = None,
    *,
    training_count: int = 150,
    validation_count: int = 30,
    test_count: int = 50,
    seed: int = 403_243_046,
) -> tuple[list[GameConfig], list[GameConfig], list[GameConfig]]:
    """I generate and save all reproducible GA dataset splits."""

    destination = Path(root) if root is not None else _ga_dataset_root()
    training, validation, test = create_ga_dataset_splits(
        training_count=training_count,
        validation_count=validation_count,
        test_count=test_count,
        seed=seed,
    )
    save_config_dataset(training, destination / "training")
    save_config_dataset(validation, destination / "validation")
    save_config_dataset(test, destination / "test")
    return training, validation, test


def _load_or_generate_ga_split(split: str, expected_count: int) -> list[GameConfig]:
    configs = load_config_dataset(_ga_dataset_root() / split)
    if len(configs) == expected_count:
        return configs

    training, validation, test = create_ga_dataset_splits()
    generated = {
        "training": training,
        "validation": validation,
        "test": test,
    }
    return generated[split]


def create_ga_training_maps() -> list[GameConfig]:
    return _load_or_generate_ga_split("training", 150)


def create_ga_validation_maps() -> list[GameConfig]:
    return _load_or_generate_ga_split("validation", 30)


def create_ga_test_maps() -> list[GameConfig]:
    return _load_or_generate_ga_split("test", 50)
