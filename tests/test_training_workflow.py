"""Verify the real orchestration's completion gate before expensive execution."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from enhancer_benchmark_audit.training_workflow import run_training


class TrainingWorkflowTests(unittest.TestCase):
    def test_activity_authenticates_all_jobs_before_evaluation(self):
        events = []
        state = {'completed': False, 'jobs': 0}

        def train(*job):
            state['jobs'] += 1

        def finish():
            self.assertEqual(state['jobs'], 54)
            state['completed'] = True
            events.append('finish')

        def evaluate():
            self.assertTrue(state['completed'])
            events.append('evaluate')

        s = SimpleNamespace(setup=lambda: events.append('setup'),
                            jobs=lambda: [(i,) for i in range(54)],
                            train_job=train, finish_training=finish)
        result = run_training('activity', training=s,
                              analysis=SimpleNamespace(selection_and_inference=evaluate))
        self.assertEqual(events, ['setup', 'finish', 'evaluate'])
        self.assertEqual(result['jobs'], 54)

    def test_failed_completion_cannot_enter_evaluation(self):
        events = []

        def fail():
            raise ValueError('Artifact authentication failed')

        s = SimpleNamespace(setup=lambda: None, jobs=lambda: [],
                            train_job=lambda *j: None, finish_training=fail)
        with self.assertRaisesRegex(ValueError, 'authentication'):
            run_training('activity', training=s,
                         analysis=SimpleNamespace(selection_and_inference=lambda: events.append('evaluated')))
        self.assertEqual(events, [])

    def test_classification_keeps_all_sixty_jobs(self):
        events = []
        s = SimpleNamespace(MODELS=range(2), STAGES=range(2), SEEDS=range(3), FOLDS=range(5),
                            phase31_setup=lambda: events.append('setup'),
                            train_job=lambda *j: events.append(j),
                            phase31_finish=lambda: events.append('finish'))
        result = run_training('classification', training=s)
        self.assertEqual(result['jobs'], 60)
        self.assertEqual(len(events), 62)
        self.assertEqual(events[-1], 'finish')


if __name__ == '__main__':
    unittest.main()
