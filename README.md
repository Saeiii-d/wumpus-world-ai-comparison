# Wumpus World AI Comparison

[![Tests](https://github.com/Saeiii-d/wumpus-world-ai-comparison/actions/workflows/tests.yml/badge.svg)](https://github.com/Saeiii-d/wumpus-world-ai-comparison/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Python implementation of an 8×8 Wumpus World environment for comparing three different AI approaches: a fully observable A\* baseline, a partially observable knowledge-based agent, and a genetic algorithm that evolves decision-policy weights.

The project focuses on how different assumptions about available information affect navigation, risk management, and decision-making in the same environment.

## Overview

The agent starts at the top-left cell of an 8×8 grid and must:

1. Find and collect the gold.
2. Survive hazards such as Wumpus cells and pits.
3. Reach the exit while still alive.

Each valid move costs one health point. Entering a Wumpus cell causes immediate death, while entering a pit reduces health according to the configured pit penalty.

The environment provides local percepts:

- **Stench** near a Wumpus
- **Breeze** near a pit
- **Bump** after attempting to move into a wall

The three approaches intentionally operate under different information assumptions.

## AI Approaches

| Method                    | Observability | Main Idea                                                                                                   |
| ------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------- |
| **A\* Offline Oracle**    | Full map      | Uses complete knowledge of the map as a fully observable search baseline.                                   |
| **Knowledge-Based Agent** | Partial       | Builds a knowledge base from local percepts and infers safe, dangerous, and uncertain cells.                |
| **Genetic Policy**        | Partial       | Evolves heuristic weights that guide an online policy using only information discovered during exploration. |

### A\* Offline Oracle

The A\* implementation has access to the complete map, including hazards, walls, gold, and exit location.

Its state includes:

- Current position
- Gold possession
- Remaining health

Because it receives information unavailable to the other two agents, it is treated as an **oracle baseline**, not as a directly equivalent partially observable competitor.

It is useful for evaluating what is achievable when the environment is fully known.

### Knowledge-Based Agent

The knowledge-based agent does not inspect hidden map cells.

Instead, it maintains a knowledge base containing:

- Visited and traversable cells
- Known safe cells
- Known and possible pits
- Known and possible Wumpus locations
- Proven negative facts such as `not_pit` and `not_wumpus`
- Percepts observed during exploration

Inference rules are repeatedly applied until no new facts can be derived.

When no proven-safe move is available, the agent estimates heuristic risk from the evidence collected so far and chooses among uncertain alternatives.

### Genetic Policy

The genetic algorithm evolves an online decision policy rather than a fixed path.

Each chromosome contains 11 normalized heuristic weights:

- `explore`
- `known_safe`
- `toward_frontier`
- `toward_exit`
- `avoid_wumpus_risk`
- `avoid_pit_risk`
- `avoid_revisit`
- `avoid_known_pit`
- `avoid_backtrack`
- `information_gain`
- `health_reserve`

The default GA configuration uses:

- Population size: **16**
- Generations: **12**
- Mutation rate: **0.20**
- Training batch size: **36**

Training follows a curriculum that gradually introduces harder maps.

The final generation is evaluated on the complete training split, and strong candidates are then compared on a separate validation split.

## Dataset Separation

The genetic-policy experiments use three independent map splits:

| Split      | Maps | Purpose                          |
| ---------- | ---: | -------------------------------- |
| Training   |  150 | Evolution and fitness evaluation |
| Validation |   30 | Final chromosome selection       |
| Test       |   50 | Held-out evaluation only         |

The held-out test maps are not used for training or model selection.

All map files are included in the repository so experiments can be reproduced.

A pretrained chromosome generated with seed `42` is also included:

```text
data/pretrained_ga_seed42.json
```

This allows the genetic agent to be demonstrated without retraining the model.

## Benchmark Results

Evaluated on the **50-map held-out GA test split** (`python main.py --ga-test-benchmark --seed 42 --ga-runs 1`):

<img width="4050" height="2760" alt="Wumpus World Benchmark Dashboard" src="https://github.com/user-attachments/assets/7ebb1840-546c-43c9-8813-1c67a760db2c" />

| Algorithm | Runs | Success | Gold | Death (W / P) | Mean Score | Mean Steps | Mean Run (s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **A\* (offline oracle)** | 50 | 100.0% | 100.0% | 0.0% (0.0% / 0.0%) | 611.1 | 11.6 | 0.0002 |
| **Logic-based** | 50 | 78.0% | 86.0% | 10.0% (2.0% / 8.0%) | 444.1 | 39.4 | 0.0022 |
| **Genetic policy** | 50 | 62.0% | 84.0% | 8.0% (2.0% / 6.0%) | 340.3 | 35.3 | 0.0027 |

> Note: **A\* (offline oracle)** has full-map visibility and serves as an upper-bound reference rather than an information-equivalent competitor (see [Design Notes](#design-notes)).
 
## Project Structure

```text
wumpus-world-ai-comparison/
├── .github/
│   └── workflows/
│       └── tests.yml
├── algorithms/
│   ├── genetic.py
│   ├── logic.py
│   └── search.py
├── data/
│   ├── ga_datasets/
│   │   ├── training/
│   │   ├── validation/
│   │   └── test/
│   ├── challenge_map.txt
│   ├── config.py
│   ├── default_map.txt
│   └── pretrained_ga_seed42.json
├── models/
│   ├── agent.py
│   ├── environment.py
│   └── percepts.py
├── tests/
│   ├── test_algorithms.py
│   ├── test_environment.py
│   └── test_reporting.py
├── .gitignore
├── LICENSE
├── main.py
├── utils.py
└── README.md
```

## Requirements

- Python **3.10+**
- No third-party packages are required

The project uses only the Python standard library.

## Running the Project

Run the default map:

```bash
python main.py --map default
```

Run the challenge map:

```bash
python main.py --map challenge
```

Generate and run a reproducible random map:

```bash
python main.py --map random --random-seed 42
```

If no map argument is provided in an interactive terminal, the program displays a map-selection menu:

```bash
python main.py
```

### Show Complete Paths

```bash
python main.py --map default --show-paths
```

### Use the Pretrained Genetic Model

Training the genetic algorithm can be skipped by loading the included chromosome:

```bash
python main.py \
  --map default \
  --ga-model data/pretrained_ga_seed42.json
```

## Benchmark Modes

### Fixed Benchmark

Evaluate all three approaches on the predefined held-out benchmark maps:

```bash
python main.py --benchmark
```

### Held-Out GA Test Split

Evaluate the approaches on the saved 50-map GA test split:

```bash
python main.py --ga-test-benchmark
```

### Random Benchmark

Generate solvable random maps and evaluate all methods:

```bash
python main.py \
  --random-benchmark \
  --random-maps 20 \
  --random-seed 42
```

Difficulty profiles are also available:

```bash
python main.py \
  --random-benchmark \
  --random-maps 20 \
  --random-difficulty mixed
```

Supported difficulty values are:

```text
custom
easy
medium
hard
mixed
```

Generated maps can also be saved for later reproduction:

```bash
python main.py \
  --random-benchmark \
  --random-maps 20 \
  --save-random-maps generated_maps
```

## Map Format

Custom maps contain 12 non-empty lines:

1. Eight rows representing the 8×8 grid
2. Initial health
3. Gold score
4. Pit penalty
5. Exit position

Supported symbols:

| Symbol | Meaning    |
| ------ | ---------- |
| `*`    | Empty cell |
| `W`    | Wumpus     |
| `P`    | Pit        |
| `G`    | Gold       |
| `D`    | Wall       |

Example:

```text
*DP***W*
****P***
D******P
*P**D***
***G*D**
P******D
**D*P***
********
100
500
15
8 8
```

Map files use **one-based coordinates** for the exit position, while the internal Python implementation uses zero-based coordinates.

## Game Rules

- Every valid move costs **1 health point**.
- Invalid moves and wall bumps do not consume health.
- Entering a Wumpus cell causes immediate death.
- Entering a pit applies:

```text
health = (health // 2) - pit_penalty
```

- The agent must collect the gold before reaching the exit.
- Reaching the exit without the gold does not count as a win.
- Final score for a successful run is:

```text
remaining_health + gold_score
```

## Testing

The project includes tests for:

- Environment rules and map validation
- Wumpus, pit, wall, and health behavior
- Percept generation
- A\* behavior
- Knowledge-base inference
- Partial-observability boundaries
- Genetic population and training behavior
- Training / validation / test split separation
- Pretrained model validation
- Random-map reproducibility and solvability
- Benchmark and reporting logic

Run the complete test suite with:

```bash
python -m unittest discover -s tests -v
```

Current test suite:

```text
Ran 52 tests
OK
```

## Design Notes

This project intentionally distinguishes between two evaluation settings:

- **Full observability:** A\* can inspect the complete environment and therefore acts as an oracle baseline.
- **Partial observability:** the knowledge-based and genetic agents must make decisions from local observations and accumulated knowledge.

For this reason, raw scores between A\* and the partially observable methods should not be interpreted as a perfectly information-balanced comparison.

The more meaningful comparison is how each approach behaves under its own information model, including survival, exploration, risk management, and ability to complete the objective.

## Key Concepts Demonstrated

- A\* search
- Knowledge representation and iterative inference
- Partial observability
- Heuristic risk estimation
- Genetic algorithms
- Tournament selection
- Crossover and mutation
- Curriculum-based training
- Train / validation / test separation
- Reproducible random-map generation
- Automated testing

## Related Article

For a detailed discussion of the design decisions, observability differences, knowledge-based reasoning, and genetic-policy training used in this project, see:

[**Comparing A\*, Knowledge-Based Reasoning, and a Genetic Policy in Wumpus World**](https://medium.com/@saeiiid.khazaei/comparing-a-knowledge-based-reasoning-and-a-genetic-policy-in-wumpus-world-4e9ea8f71b04)

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
