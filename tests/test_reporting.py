import unittest
from data.config import create_default_map
from main import build_parser, load_config, run_algorithm, run_benchmark
from models.environment import GameEnvironment
from utils import GameResult, compare_benchmark, compare_difficulty_levels, compare_results, format_benchmark_details

class ReportingTests(unittest.TestCase):

    @staticmethod
    def _result(name: str, *, success: bool, seed_note: str='') -> GameResult:
        env = GameEnvironment(create_default_map())
        env.won = success
        if success:
            env.agent.has_gold = True
        return GameResult.from_environment(name, env, execution_time=0.01, note=seed_note)

    def test_parser_uses_one_ga_run_by_default(self):
        args = build_parser().parse_args(['--benchmark'])
        self.assertEqual(args.ga_runs, 1)

    def test_benchmark_rejects_zero_ga_runs_before_training(self):
        with self.assertRaises(ValueError):
            run_benchmark(seed=42, ga_runs=0)

    def test_benchmark_headings_distinguish_mean_and_total_time(self):
        grouped = {'A* (offline oracle)': [self._result('A* (offline oracle)', success=True)], 'Logic-based': [self._result('Logic-based', success=False)]}
        report = compare_benchmark(grouped)
        self.assertIn('Mean Run(s)', report)
        self.assertIn('Total Train(s)', report)
        self.assertIn('oracle/upper-bound baseline', report)

    def test_benchmark_handles_oracle_only_input(self):
        grouped = {'A* (offline oracle)': [self._result('A* (offline oracle)', success=True)]}
        report = compare_benchmark(grouped)
        self.assertIn('No partially observable method', report)

    def test_reports_terminal_health_and_termination_reason(self):
        env = GameEnvironment(create_default_map())
        env.termination_reason = 'unreachable'
        result = GameResult.from_environment('Example', env, execution_time=0.01)
        grouped = {'Example': [result]}
        self.assertIn('Terminal H', compare_benchmark(grouped))
        self.assertIn('unreachable', format_benchmark_details(grouped))

    def test_map_level_report_contains_ga_seed(self):
        grouped = {'Genetic policy': [self._result('Genetic policy', success=True, seed_note='ga_seed=42')]}
        report = format_benchmark_details(grouped)
        self.assertIn('ga_seed=42', report)
        self.assertIn('PER-MAP RESULTS', report)

    def test_single_map_analysis_explains_success_failure_and_error(self):
        success = self._result('Winner', success=True)
        success.health_remaining = 80
        failed = self._result('Explorer', success=False)
        failed.has_gold = True
        failed.termination_reason = 'max_steps'
        error = GameResult(algorithm_name='Broken', error='boom')
        report = compare_results([success, failed, error])
        self.assertIn('Best score: Winner', report)
        self.assertIn('max_steps', report)
        self.assertIn('runtime error: boom', report)
        self.assertIn('terminal_health', str(success))

    def test_run_algorithm_reports_runtime_errors(self):

        class BrokenAlgorithm:

            def __init__(self, env):
                self.env = env

            def execute(self):
                raise RuntimeError('broken')
        result = run_algorithm('Broken', create_default_map(), lambda env: BrokenAlgorithm(env))
        self.assertEqual(result.status, 'ERROR')
        self.assertIn('RuntimeError', result.error or '')

    def test_load_config_named_maps(self):
        self.assertEqual(load_config('default').name, 'Default Map')
        self.assertEqual(load_config('challenge').name, 'Challenge Map')

    def test_parser_exposes_random_benchmark_settings(self):
        args = build_parser().parse_args([
            '--random-benchmark',
            '--random-maps', '25',
            '--random-pits', '9',
            '--random-walls', '4',
            '--random-seed', '123',
        ])
        self.assertTrue(args.random_benchmark)
        self.assertEqual(args.random_maps, 25)
        self.assertEqual(args.random_pits, 9)
        self.assertEqual(args.random_walls, 4)
        self.assertEqual(args.random_seed, 123)

    def test_load_config_supports_one_random_map(self):
        config = load_config(
            'random',
            random_seed=222,
            random_pits=5,
            random_walls=2,
        )
        flattened = [cell for row in config.grid for cell in row]
        self.assertEqual(flattened.count('W'), 1)
        self.assertEqual(flattened.count('P'), 5)
        self.assertEqual(flattened.count('D'), 2)

    def test_parser_exposes_difficulty_and_saved_test_benchmark(self):
        args = build_parser().parse_args([
            '--random-benchmark', '--random-difficulty', 'mixed'
        ])
        self.assertEqual(args.random_difficulty, 'mixed')
        args = build_parser().parse_args(['--ga-test-benchmark'])
        self.assertTrue(args.ga_test_benchmark)

    def test_difficulty_report_separates_levels(self):
        easy = self._result('Logic-based', success=True)
        easy.map_name = 'test-001-easy'
        hard = self._result('Logic-based', success=False)
        hard.map_name = 'test-002-hard'
        report = compare_difficulty_levels({'Logic-based': [easy, hard]})
        self.assertIn('Easy', report)
        self.assertIn('Hard', report)
        self.assertIn('RESULTS BY DIFFICULTY', report)


    def test_parser_accepts_pretrained_ga_model(self):
        args = build_parser().parse_args([
            '--map', 'default', '--ga-model', 'data/pretrained_ga_seed42.json'
        ])
        self.assertEqual(args.ga_model, 'data/pretrained_ga_seed42.json')


if __name__ == '__main__':
    unittest.main()
