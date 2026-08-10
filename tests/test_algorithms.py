import inspect
import unittest
from algorithms.genetic import GeneticAlgorithm, ValidationSummary
from algorithms.logic import LogicBasedAgent
from algorithms.search import AStarSearch
from data.config import GameConfig, create_challenge_map, create_default_map, create_test_maps
from models.environment import GameEnvironment
from models.percepts import KnowledgeBase, Percept
from models.agent import GENE_NAMES

class AlgorithmTests(unittest.TestCase):

    def test_percept_string_is_closed(self):
        self.assertEqual(str(Percept(stench=True, breeze=True)), '[Stench, Breeze]')

    def test_astar_completes_default_objective(self):
        env = GameEnvironment(create_default_map())
        self.assertTrue(AStarSearch(env).execute())
        self.assertTrue(env.agent.has_gold)
        self.assertTrue(env.won)

    def test_logic_agent_runs_without_runtime_error(self):
        env = GameEnvironment(create_default_map())
        result = LogicBasedAgent(env).execute()
        self.assertIsInstance(result, bool)
        self.assertGreaterEqual(env.agent.steps_taken, 1)

    def test_logic_completes_both_included_maps(self):
        for config in (create_default_map(), create_challenge_map()):
            env = GameEnvironment(config)
            self.assertTrue(LogicBasedAgent(env).execute(), config.name)
            self.assertTrue(env.agent.has_gold)

    def test_genetic_population_is_diverse(self):
        ga = GeneticAlgorithm(GameEnvironment(create_default_map()), population_size=6, generations=1, seed=7, max_steps=40)
        signatures = {tuple((round(agent.weights[name], 6) for name in agent.weights)) for agent in ga.population}
        self.assertGreater(len(signatures), 1)

    def test_genetic_execute_updates_caller_environment(self):
        env = GameEnvironment(create_default_map())
        ga = GeneticAlgorithm(env, population_size=6, generations=2, seed=3, max_steps=80)
        result = ga.execute()
        self.assertIsInstance(result, bool)
        self.assertGreaterEqual(len(env.agent.path), 1)
        self.assertGreaterEqual(ga.training_time, 0.0)

    def test_ga_training_does_not_include_selected_test_map(self):
        config = create_default_map()
        ga = GeneticAlgorithm(GameEnvironment(config), population_size=6, generations=1, seed=1, max_steps=20)
        training_rows = {tuple((''.join(row) for row in item.grid)) for item in ga.training_configs}
        selected_rows = tuple((''.join(row) for row in config.grid))
        self.assertNotIn(selected_rows, training_rows)

    def test_ga_fitness_seeds_are_common_across_chromosomes(self):
        ga = GeneticAlgorithm(GameEnvironment(create_default_map()), population_size=6, generations=1, seed=1, max_steps=20)
        seen = []
        original_rollout = ga.rollout

        def recording_rollout(chromosome, config, *, seed, stochastic):
            seen.append((config.name, seed))
            return original_rollout(chromosome, config, seed=seed, stochastic=stochastic)
        ga.rollout = recording_rollout
        ga.fitness(ga.population[0], chromosome_index=0)
        first = seen[:]
        seen.clear()
        ga.fitness(ga.population[1], chromosome_index=99)
        self.assertEqual(first, seen)

    def test_negative_facts_clear_danger_support(self):
        kb = KnowledgeBase()
        target = (0, 1)
        kb.record_visit((0, 0), Percept(breeze=True, stench=True), [target, (1, 0)])
        kb.not_pit.add(target)
        kb.not_wumpus.add(target)
        pit_support, wumpus_support = kb.danger_support(target)
        self.assertEqual((pit_support, wumpus_support), (0, 0))

    def test_logic_can_complete_a_required_survivable_pit_route(self):
        config = GameConfig(grid=[list('*PG*****'), list('DDD*****'), list('********'), list('********'), list('********'), list('********'), list('********'), list('********')], initial_health=100, gold_score=500, pit_penalty=5, exit_pos=(0, 3))
        env = GameEnvironment(config)
        self.assertTrue(LogicBasedAgent(env).execute())
        self.assertEqual(env.agent.pit_entries, 1)

    def test_partially_observable_algorithms_do_not_read_hidden_cells(self):
        for algorithm_class in (LogicBasedAgent, GeneticAlgorithm):
            source = inspect.getsource(algorithm_class)
            self.assertNotIn('.config.grid', source, algorithm_class.__name__)
            self.assertNotIn('.get_cell(', source, algorithm_class.__name__)
            self.assertNotIn('.gold_pos', source, algorithm_class.__name__)

    def test_ga_rejects_a_known_lethal_pit(self):
        config = GameConfig(grid=[list('*PG*****'), list('********'), list('********'), list('********'), list('********'), list('********'), list('********'), list('********')], initial_health=12, gold_score=500, pit_penalty=5, exit_pos=(0, 2))
        env = GameEnvironment(config)
        ga = GeneticAlgorithm(env, population_size=6, generations=1, max_steps=20)
        kb = KnowledgeBase()
        kb.record_visit((0, 0), env.observe(), env.candidate_neighbors((0, 0)))
        kb.mark_pit((0, 1))
        action = ga.choose_action(ga.population[0], env, kb, ga.rng, stochastic=False)
        self.assertEqual(action, (1, 0))

    def test_ga_has_structurally_diverse_training_maps(self):
        ga = GeneticAlgorithm(GameEnvironment(create_default_map()), population_size=6, generations=1, max_steps=20)
        self.assertGreaterEqual(len(ga.training_configs), 6)
        training_layouts = {tuple((''.join(row) for row in config.grid)) for config in ga.training_configs}
        test_layouts = {tuple((''.join(row) for row in config.grid)) for config in create_test_maps()}
        self.assertTrue(training_layouts.isdisjoint(test_layouts))
    def test_risk_score_respects_proven_safe_and_known_hazard_facts(self):
        kb = KnowledgeBase()
        target = (0, 1)
        kb.record_visit((0, 0), Percept(breeze=True, stench=True), [target, (1, 0)])
        pit_risk, wumpus_risk, combined = kb.risk_score(target)
        self.assertGreater(combined, 0.0)
        kb.not_pit.add(target)
        kb.not_wumpus.add(target)
        kb.safe_cells.add(target)
        self.assertEqual(kb.risk_score(target), (0.0, 0.0, 0.0))
        kb.safe_cells.discard(target)
        kb.mark_wumpus(target)
        self.assertEqual(kb.risk_score(target)[1], 1.0)

    def test_ga_curriculum_ends_with_all_training_maps(self):
        ga = GeneticAlgorithm(
            GameEnvironment(create_default_map()),
            population_size=4,
            generations=4,
            training_batch_size=9,
            max_steps=10,
        )
        first_batch = ga._curriculum_batch(0)
        final_batch = ga._curriculum_batch(3)
        self.assertTrue(all("easy" in config.name.lower() for config in first_batch))
        self.assertEqual(len(final_batch), 150)

    def test_validation_selection_key_prioritizes_success(self):
        higher_success = ValidationSummary(0.7, 0.7, 0.3, 100.0, 10.0)
        higher_score = ValidationSummary(0.6, 1.0, 0.0, 600.0, 100.0)
        self.assertGreater(higher_success.selection_key, higher_score.selection_key)


    def test_pretrained_model_contains_all_genes(self):
        agent = GeneticAlgorithm.load_model('data/pretrained_ga_seed42.json')
        self.assertEqual(set(agent.weights), set(GENE_NAMES))
        self.assertAlmostEqual(sum(agent.weights.values()), 1.0, places=7)
        self.assertTrue(all(value > 0 for value in agent.weights.values()))


if __name__ == '__main__':
    unittest.main()
