import pandas as pd

class ParetoAnalysis:
    def __init__(self, data, group_names):
        self.data = data
        self.num_groups = len(data)
        self.num_metrics = len(data[0])
        self.is_dominated_freq = [0] * self.num_metrics
        self.dominated_freq = [0] * self.num_metrics
        self.group_names = group_names
        self.results = [[0] * 4 for _ in range(self.num_groups)]


    def is_dominated(self, group_index, metric_index):
        group = self.data[group_index]
        metric = group[metric_index]
        for other_group_index, other_group in enumerate(self.data):
            if other_group_index != group_index:
                other_metric = other_group[metric_index]
                if metric < other_metric:
                    return True
        return False

    def compute_dominance(self):
        for group_index in range(self.num_groups):
            for metric_index in range(self.num_metrics):
                self.results[group_index][0] = self.group_names[group_index]
                if self.is_dominated(group_index, metric_index):
                    self.results[group_index][1] += 1
                else:
                    self.results[group_index][2] += 1

    def get_results(self):
        self.compute_dominance()
        for row in self.results:
            row[3] = row[1] - row[2]
        headers = ['group_id', 'dominates_freq', 'isDominated_freq', 'scalar_score']
        df = pd.DataFrame(self.results, columns = headers).sort_values('scalar_score', ascending=False)
        return df