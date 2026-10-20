"""Orchestration around immutable study implementations; no recipe changes."""
import importlib
import itertools


def run_training(task, *, evaluate=True, training=None, analysis=None):
    """Run every declared job, authenticate completion, then evaluate if requested."""
    if task == 'classification':
        s = training or importlib.import_module('phase11c_completion')
        s.phase31_setup()
        jobs = itertools.product(s.MODELS, s.STAGES, s.SEEDS, s.FOLDS)
        count = 0
        for job in jobs:
            s.train_job(*job)
            count += 1
        s.phase31_finish()
        return {'task': task, 'jobs': count, 'completion_authenticated': True}
    if task != 'activity':
        raise ValueError('Unknown training task: ' + task)
    s = training or importlib.import_module('mpra_fairness_training')
    s.setup()
    count = 0
    for job in s.jobs():
        s.train_job(*job)
        count += 1
    # The frozen evaluator requires the authenticated completed status set here.
    s.finish_training()
    if evaluate:
        a = analysis or importlib.import_module('mpra_fairness_analysis')
        a.selection_and_inference()
    return {'task': task, 'jobs': count, 'completion_authenticated': True,
            'evaluation_requested': evaluate}
