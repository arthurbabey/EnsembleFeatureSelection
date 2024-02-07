import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from src.merging_strategy_methods import \
    merging_strategy_union_of_pairwise_intersections


def test_basic_functionality():
    subsets = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    result = merging_strategy_union_of_pairwise_intersections(subsets)
    assert isinstance(result, set)


def test_empty_input():
    subsets = []
    result = merging_strategy_union_of_pairwise_intersections(subsets)
    assert result == set()


def test_single_subset():
    subsets = [[1, 2, 3, 4]]
    result = merging_strategy_union_of_pairwise_intersections(subsets)
    assert result == set()


def test_multiple_subsets():
    subsets = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    result = merging_strategy_union_of_pairwise_intersections(subsets)
    assert result == {2, 3, 4}


def test_symmetry_property():
    subsets_1 = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    subsets_2 = [[3, 4, 5], [2, 3, 4], [1, 2, 3]]
    result_1 = merging_strategy_union_of_pairwise_intersections(subsets_1)
    result_2 = merging_strategy_union_of_pairwise_intersections(subsets_2)
    assert result_1 == result_2


def test_empty_output():
    subsets = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    result = merging_strategy_union_of_pairwise_intersections(subsets)
    assert result == set()
