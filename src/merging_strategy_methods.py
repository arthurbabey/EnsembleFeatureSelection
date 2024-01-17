from itertools import combinations
from ranky import kemeny_young, borda ;


# here major difference is that set based method like union works with subsets of features
# and produces a new set combining the features present in features
# while ranking methods like k-y or borda used rankings so they need ranking for all features
# and here they produces the list of features ranked by importance, we can threshold to select the k-top number of features


def merging_strategy_union_of_pairwise_intersections(subsets):
    # Generate all pairs of indices
    pairs = combinations(range(len(subsets)), 2)

    # Compute intersections between pairs of subsets
    intersections = [set(subsets[i]) & set(subsets[j]) for i, j in pairs]

    # Compute the union of pairwise intersections
    union_prwse_inter = set().union(*intersections)

    return union_prwse_inter


# could seed here
def merging_strategy_kemeny_young(scores, k_features=None, **kwargs):
    scores_merged = kemeny_young(m=scores, axis=0, **kwargs)
    sorted_indices = sorted(range(len(scores_merged)), key=lambda i: scores_merged[i], reverse=True)
    if k_features is not None:
        return sorted_indices[:k_features]
    else:
        return sorted_indices


# need to be pd.df to use method='median'
def merging_strategy_borda(scores, k_features=None, **kwargs):
    scores_merged = borda(m=scores, **kwargs)
    sorted_indices = sorted(range(len(scores_merged)), key=lambda i: scores_merged[i], reverse=True)
    if k_features is not None:
        return sorted_indices[:k_features]
    else:
        return sorted_indices
