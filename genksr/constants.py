import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data/')

# task-specific measurement alphabets
KQD_NUM_MMT_OUTCOMES = 6
SQKD_NUM_MMT_OUTCOMES = 2

# Backwards-compatible alias used by KQD-only utilities and timing figures.
NUM_MMT_OUTCOMES = KQD_NUM_MMT_OUTCOMES


def measurement_outcomes_for_task(task):
    if task == 'kqd':
        return KQD_NUM_MMT_OUTCOMES
    if task == 'sqkd':
        return SQKD_NUM_MMT_OUTCOMES
    raise ValueError(f"Unknown task {task!r}; expected 'kqd' or 'sqkd'")

# gcn features
ONE_HOT_FEATURE = 'one_hot'
WDEGREE_FEATURE = 'weighted_degree'

# learning rate schedulers
COSINE_ANNEALING_WARM_RESTARTS_SCHEDULER = 'cosine_annealing_warm_restarts'
COSINE_ANNEALING_SCHEDULER = 'cosine_annealing'
WARMUP_SQRT_SCHEDULER = 'warmup_sqrt'
WARMUP_COSINE_SCHEDULER = 'warmup_cosine'
