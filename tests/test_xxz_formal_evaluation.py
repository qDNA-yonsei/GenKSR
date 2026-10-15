import numpy as np

from scripts.experiments.evaluate_xxz_sparsity import (
    build_condition_library,
    fixed_stream,
    samples_to_fixed_rows,
    select_knn_conditions,
    uniform_control_streams,
)


def test_fixed_stream_uses_nested_dimension_prefixes_and_postselection():
    rows = {
        0: np.array([1, -1, 3, 4]),
        1: np.array([5, 6, -1, 8]),
    }

    low, low_raw = fixed_stream(rows, depth=2, samples_per_dimension=2)
    high, high_raw = fixed_stream(rows, depth=2, samples_per_dimension=4)

    assert low_raw == 4
    assert high_raw == 8
    assert np.array_equal(low, np.array([1, 5, 6]))
    assert set(low).issubset(set(high))


def test_samples_to_fixed_rows_uses_displayed_order_and_rejects_wrong_sector():
    samples = np.array([
        [1, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.uint8)

    rows = samples_to_fixed_rows(samples, n_qubit=4)

    assert rows.tolist() == [10, -1]


def test_uniform_controls_are_reproducible_and_nested():
    sector = np.array([3, 5, 6, 9, 10, 12], dtype=np.int64)
    first = uniform_control_streams(7, 4, 2, 20, sector)
    second = uniform_control_streams(7, 4, 2, 20, sector)

    assert np.array_equal(first['uniform_full_raw'], second['uniform_full_raw'])
    assert np.array_equal(first['uniform_sector'], second['uniform_sector'])
    assert set(first['uniform_sector']).issubset(set(sector))
    assert np.array_equal(
        first['uniform_full_raw'][:5], second['uniform_full_raw'][:5])


def test_knn_condition_selection_uses_normalized_jxy_and_time():
    instances = {
        0: {'J_xy': '0.2', 'dt': '0.5'},
        1: {'J_xy': '0.8', 'dt': '0.25'},
    }
    conditions, scaling = build_condition_library(instances, depth=2)

    _, nearest = select_knn_conditions(
        target_jxy=0.79,
        target_time=0.24,
        conditions=conditions,
        scaling=scaling,
        knn_k=1,
    )

    assert nearest[0]['hamiltonian_id'] == 1
    assert nearest[0]['dimension'] == 1
