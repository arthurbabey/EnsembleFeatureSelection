import pandas as pd


class ParetoAnalysis:
    def __init__(self, data, group_names):
        self.data = data
        self.num_groups = len(data)
        self.num_metrics = len(data[0])
        self.is_dominated_freq = [0] * self.num_metrics
        self.dominated_freq = [0] * self.num_metrics
        self.group_names = group_names
        # 4 : group_id, dom_freq, is_dom_freq, scalar
        self.results = [[0] * 4 for _ in range(self.num_groups)]

    def group_dominate_count(self, group_index):
        group = self.data[group_index]
        dominate_count = 0
        for other_group_index, other_group in enumerate(self.data):
            if other_group_index != group_index:
                # compare group and other group
                # check for all metrics if group < other_group => dom_count +=1
                if all(
                    group[metric_index] >= other_group[metric_index]
                    for metric_index in range(self.num_metrics)
                ) and any(
                    group[metric_index] > other_group[metric_index]
                    for metric_index in range(self.num_metrics)
                ):
                    dominate_count += 1
        return dominate_count

    def group_isdominated_count(self, group_index):
        group = self.data[group_index]
        is_dominated_count = 0
        for other_group_index, other_group in enumerate(self.data):
            if other_group_index != group_index:
                # compare group and other group
                # check for all metrics if group < other_group => dom_count +=1
                if all(
                    group[metric_index] <= other_group[metric_index]
                    for metric_index in range(self.num_metrics)
                ) and any(
                    group[metric_index] < other_group[metric_index]
                    for metric_index in range(self.num_metrics)
                ):
                    is_dominated_count += 1
        return is_dominated_count

    def compute_dominance(self):
        for group_index in range(self.num_groups):
            # group ID
            self.results[group_index][0] = self.group_names[group_index]
            # dominate count
            self.results[group_index][1] = self.group_dominate_count(group_index)
            # is dominated count
            self.results[group_index][2] = self.group_isdominated_count(group_index)
            # scalar
            self.results[group_index][3] = (
                self.results[group_index][1] - self.results[group_index][2]
            )

    def get_results(self):
        self.compute_dominance()
        # Sort the results list based on the "scalar_score" column (index 3)
        sorted_results = sorted(self.results, key=lambda x: x[3], reverse=True)
        return sorted_results
