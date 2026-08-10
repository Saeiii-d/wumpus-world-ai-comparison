import tempfile
import unittest
from pathlib import Path

from algorithms.search import AStarSearch
from data.config import (
    GameConfig,
    create_default_map,
    create_difficulty_map,
    create_ga_test_maps,
    create_ga_training_maps,
    create_ga_validation_maps,
    create_random_map,
    create_random_test_maps,
    create_training_maps,
    map_signature,
)
from models.environment import GameEnvironment


class EnvironmentTests(unittest.TestCase):
    def test_default_map_is_valid_and_exit_is_not_wall(self):
        config = create_default_map()
        self.assertNotEqual(config.cell(config.exit_pos), "D")

    def test_invalid_wall_move_does_not_cost_health(self):
        env = GameEnvironment(create_default_map())
        before = env.agent.health
        outcome = env.apply_move((0, 1))
        self.assertFalse(outcome.valid)
        self.assertEqual(env.agent.health, before)
        self.assertEqual(env.agent.steps_taken, 0)

    def test_pit_formula_is_preserved(self):
        config = GameConfig(
            grid=[
                list("*PG*****"),
                list("********"),
                list("********"),
                list("********"),
                list("********"),
                list("********"),
                list("********"),
                list("********"),
            ],
            initial_health=100,
            gold_score=500,
            pit_penalty=15,
            exit_pos=(0, 3),
        )
        env = GameEnvironment(config)
        outcome = env.apply_move((0, 1))
        self.assertEqual(outcome.event, "pit")
        self.assertEqual(env.agent.health, ((100 - 1) // 2) - 15)

    def test_exit_without_gold_is_not_a_win(self):
        config = GameConfig(
            grid=[
                list("**G*****"),
                list("********"),
                list("********"),
                list("********"),
                list("********"),
                list("********"),
                list("********"),
                list("********"),
            ],
            initial_health=100,
            gold_score=500,
            pit_penalty=15,
            exit_pos=(0, 1),
        )
        env = GameEnvironment(config)
        outcome = env.apply_move((0, 1))
        self.assertEqual(outcome.event, "exit_without_gold")
        self.assertFalse(env.won)
        self.assertFalse(env.game_over)

    def test_winning_requires_gold_then_exit(self):
        config = GameConfig(
            grid=[
                list("*G******"),
                list("********"),
                list("********"),
                list("********"),
                list("********"),
                list("********"),
                list("********"),
                list("********"),
            ],
            initial_health=100,
            gold_score=500,
            pit_penalty=15,
            exit_pos=(0, 2),
        )
        env = GameEnvironment(config)
        env.apply_move((0, 1))
        env.apply_move((0, 2))
        self.assertTrue(env.won)
        self.assertEqual(env.get_final_score(), env.agent.health + 500)

    def test_file_round_trip(self):
        config = create_default_map()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.txt"
            config.to_file(path)
            loaded = GameConfig.from_file(path)
        self.assertEqual(loaded.grid, config.grid)
        self.assertEqual(loaded.exit_pos, config.exit_pos)
        self.assertEqual(loaded.pit_penalty, config.pit_penalty)

    def test_astar_can_use_a_survivable_required_pit(self):
        config = GameConfig(
            grid=[
                list("*PG*****"),
                list("DDD*****"),
                list("********"),
                list("********"),
                list("********"),
                list("********"),
                list("********"),
                list("********"),
            ],
            initial_health=100,
            gold_score=500,
            pit_penalty=5,
            exit_pos=(0, 3),
        )
        env = GameEnvironment(config)
        self.assertTrue(AStarSearch(env).execute())
        self.assertEqual(env.agent.pit_entries, 1)
        self.assertTrue(env.agent.has_gold)


    def test_starting_on_gold_exit_wins_without_a_move(self):
        config = GameConfig(
            grid=[
                list("G*******"),
                list("********"),
                list("********"),
                list("********"),
                list("********"),
                list("********"),
                list("********"),
                list("********"),
            ],
            initial_health=100,
            gold_score=500,
            pit_penalty=15,
            exit_pos=(0, 0),
        )
        env = GameEnvironment(config)
        self.assertTrue(env.won)
        self.assertTrue(env.game_over)
        self.assertEqual(env.agent.steps_taken, 0)
        self.assertEqual(env.get_final_score(), 600)

    def test_input_file_uses_one_based_exit_coordinates(self):
        payload = "\n".join([
            "G*******",
            "********",
            "********",
            "********",
            "********",
            "********",
            "********",
            "********",
            "100",
            "500",
            "15",
            "1 1",
        ]) + "\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "one_based.txt"
            path.write_text(payload, encoding="utf-8")
            config = GameConfig.from_file(path)
        self.assertEqual(config.exit_pos, (0, 0))

    def test_zero_based_exit_in_file_is_rejected(self):
        payload = "\n".join([
            "G*******",
            "********",
            "********",
            "********",
            "********",
            "********",
            "********",
            "********",
            "100",
            "500",
            "15",
            "0 0",
        ]) + "\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid_coordinates.txt"
            path.write_text(payload, encoding="utf-8")
            with self.assertRaises(ValueError):
                GameConfig.from_file(path)

    def test_bump_percept_is_available_after_wall_attempt(self):
        env = GameEnvironment(create_default_map())
        outcome = env.apply_move((0, 1))
        self.assertEqual(outcome.event, "bump")
        self.assertTrue(env.observe().bump)

    def test_combined_stench_and_breeze_with_multiple_hazards(self):
        config = GameConfig(
            grid=[
                list("*W******"),
                list("P*******"),
                list("****W***"),
                list("*****P**"),
                list("********"),
                list("********"),
                list("********"),
                list("*******G"),
            ],
            initial_health=100,
            gold_score=500,
            pit_penalty=10,
            exit_pos=(7, 7),
        )
        percept = GameEnvironment(config).observe()
        self.assertTrue(percept.stench)
        self.assertTrue(percept.breeze)

    def test_astar_returns_false_for_an_unreachable_gold(self):
        config = GameConfig(
            grid=[
                list("*D******"),
                list("D*******"),
                list("********"),
                list("********"),
                list("****G***"),
                list("********"),
                list("********"),
                list("********"),
            ],
            initial_health=100,
            gold_score=500,
            pit_penalty=10,
            exit_pos=(7, 7),
        )
        env = GameEnvironment(config)
        self.assertFalse(AStarSearch(env).execute())
        self.assertFalse(env.won)
        self.assertTrue(env.agent.is_alive)
        self.assertEqual(env.termination_reason, "unreachable")

    def test_move_with_one_health_causes_health_depletion(self):
        config = GameConfig(
            grid=[
                list("*G******"),
                list("********"),
                list("********"),
                list("********"),
                list("********"),
                list("********"),
                list("********"),
                list("********"),
            ],
            initial_health=1,
            gold_score=500,
            pit_penalty=0,
            exit_pos=(0, 1),
        )
        env = GameEnvironment(config)
        outcome = env.apply_move((0, 1))
        self.assertEqual(outcome.event, "health_depleted")
        self.assertFalse(env.agent.is_alive)
        self.assertEqual(env.agent.death_reason, "health_depleted")
        self.assertFalse(env.agent.has_gold)

    def test_random_map_is_reproducible_and_has_requested_objects(self):
        first = create_random_map(seed=1234, pit_count=7, wall_count=3)
        second = create_random_map(seed=1234, pit_count=7, wall_count=3)

        self.assertEqual(first.grid, second.grid)
        self.assertEqual(first.exit_pos, second.exit_pos)
        flattened = [cell for row in first.grid for cell in row]
        self.assertEqual(flattened.count("W"), 1)
        self.assertEqual(flattened.count("P"), 7)
        self.assertEqual(flattened.count("D"), 3)
        self.assertEqual(flattened.count("G"), 1)
        self.assertNotEqual(first.start_pos, first.exit_pos)
        self.assertNotIn(first.cell(first.exit_pos), {"W", "P", "D", "G"})

    def test_every_generated_random_map_is_solvable_by_astar(self):
        for config in create_random_test_maps(8, seed=77, pit_count=8, wall_count=5):
            env = GameEnvironment(config)
            self.assertTrue(AStarSearch(env).execute(), config.name)
            self.assertTrue(env.won)

    def test_random_map_batch_is_unique_and_disjoint_from_ga_training(self):
        configs = create_random_test_maps(10, seed=91, pit_count=6, wall_count=2)
        signatures = {
            (tuple("".join(row) for row in config.grid), config.exit_pos)
            for config in configs
        }
        training_grids = {
            tuple("".join(row) for row in config.grid)
            for config in create_training_maps()
        }
        self.assertEqual(len(signatures), 10)
        self.assertTrue(
            all(tuple("".join(row) for row in config.grid) not in training_grids for config in configs)
        )

    def test_random_map_rejects_objects_that_do_not_fit(self):
        with self.assertRaises(ValueError):
            create_random_map(seed=1, pit_count=61, wall_count=0)


    def test_ga_dataset_splits_have_exact_sizes_and_no_overlap(self):
        training = create_ga_training_maps()
        validation = create_ga_validation_maps()
        test = create_ga_test_maps()
        self.assertEqual((len(training), len(validation), len(test)), (150, 30, 50))
        training_signatures = {map_signature(config) for config in training}
        validation_signatures = {map_signature(config) for config in validation}
        test_signatures = {map_signature(config) for config in test}
        self.assertTrue(training_signatures.isdisjoint(validation_signatures))
        self.assertTrue(training_signatures.isdisjoint(test_signatures))
        self.assertTrue(validation_signatures.isdisjoint(test_signatures))

    def test_ga_dataset_splits_are_balanced_by_difficulty(self):
        expected = {
            "training": (create_ga_training_maps(), {"easy": 50, "medium": 50, "hard": 50}),
            "validation": (create_ga_validation_maps(), {"easy": 10, "medium": 10, "hard": 10}),
        }
        for split_name, (configs, counts) in expected.items():
            actual = {difficulty: 0 for difficulty in counts}
            for config in configs:
                for difficulty in actual:
                    if difficulty in config.name.lower():
                        actual[difficulty] += 1
                        break
            self.assertEqual(actual, counts, split_name)

    def test_difficulty_profiles_generate_solvable_maps(self):
        ranges = {
            "easy": ((2, 3), (0, 1)),
            "medium": ((4, 6), (2, 4)),
            "hard": ((7, 10), (4, 8)),
        }
        for index, (difficulty, (pit_range, wall_range)) in enumerate(ranges.items()):
            config = create_difficulty_map(seed=900 + index, difficulty=difficulty)
            cells = [cell for row in config.grid for cell in row]
            self.assertIn(cells.count("P"), range(pit_range[0], pit_range[1] + 1))
            self.assertIn(cells.count("D"), range(wall_range[0], wall_range[1] + 1))
            env = GameEnvironment(config)
            self.assertTrue(AStarSearch(env).execute(), difficulty)


if __name__ == "__main__":
    unittest.main()
