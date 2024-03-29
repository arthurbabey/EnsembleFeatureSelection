from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split

from .feature import Feature
from .merging_strategy_methods import *
from .metrics import *
from .pareto import ParetoAnalysis


class FeatureSelectionPipeline:
    def __init__(
        self,
        data,
        fs_methods,
        merging_strategy,
        classifier,
        num_repeats=10,
        threshold=None,
        task="classification",
    ):
        """
        Initializes a FeatureSelectionPipeline object.

        Parameters:
        - data (pandas.DataFrame): The dataset to be used for feature selection and classification.
        - fs_methods (list): List of feature selection methods to be applied.
        - merging_strategy (str): Strategy for merging selected features ('union' or 'intersection').
        - classifier: (str) Ref to the classifier used to compute performance metrics, should be either 'RF',
            'naiveBayes' or 'bagging'.
        - num_repeats (int, optional): Number of repetitions for feature selection. Defaults to 10.
        - threshold (int, optional): number of features to used for set based merging strategy. Defaults to None,
            pipeline will use one tenth of features.
        """
        self.data = data
        self.fs_methods = fs_methods
        self.merging_strategy = merging_strategy
        self.num_repeats = num_repeats
        self.classifier = classifier
        self.threshold = threshold
        self.task = task
        self.subgroup_names = self._generate_subgroup_names()
        self.features = self._extract_features(data)
        self.FS_subsets = {}
        self.merged_features = {}

    def _compute_sbst_and_scores_per_method(self, train_data, idx):
        """
        Computes feature subset and scores for each feature selection method per repeat

        Parameters:
        - train_data (pandas.DataFrame): The training dataset.
        - idx (int): Index for the current iteration.

        Details:
        - Loops through each feature selection method to calculate feature scores and indices.
        - Stores the computed features in the `FS_subsets` dictionary.
        """
        for fs_method in self.fs_methods:
            method_name = fs_method.__name__.replace("feature_selection_", "")
            X_train, y_train = self._get_X_y(train_data)
            threshold = (
                self.threshold
                if self.threshold is not None
                else self.data.shape[1] // 10
            )
            selected_feature_scores, selected_features_indices = fs_method(
                X=X_train,
                y=y_train,
                task=self.task,
                num_features_to_select=threshold,
            )
            self.FS_subsets[(idx, method_name)] = self._compute_features(
                selected_features_indices, selected_feature_scores
            )

    def _compute_merging(self, idx):
        """
        Computes merging of selected features based on the merging strategy per repeat.

        Parameters:
        - idx (int): Index for the current iteration.

        Details:
        - Uses different merging strategies based on the set strategy.
        - Stores the merged features in the `merged_features` dictionary by group name and repeat.
        """

        for group_name in self.subgroup_names:
            if (
                self.merging_strategy
                is merging_strategy_union_of_pairwise_intersections
            ):
                group_features_idx = [
                    [
                        feature.get_name()
                        for feature in self.FS_subsets[(idx, method_name)]
                        if feature.get_selected()
                    ]
                    for method_name in group_name
                ]
                self.merged_features[(idx, group_name)] = self.merging_strategy(
                    group_features_idx
                )

            elif self.merging_strategy in (
                merging_strategy_borda,
                merging_strategy_kemeny_young,
            ):
                group_features_scores = [
                    [
                        feature.get_score()
                        for feature in self.FS_subsets[(idx, method_name)]
                    ]
                    for method_name in group_name
                ]
                features_idx = self.merging_strategy(
                    group_features_scores, k_features=20, workers=-1
                )
                self.merged_features[(idx, group_name)] = self.get_feature_names(
                    features_idx
                )
            else:
                raise ValueError("Unsupported merging strategy function")

    @staticmethod
    def _compute_pareto_analysis(groups, names):
        """
        Performs Pareto analysis to identify best-performing groups or repeats.

        Parameters:
        - groups (list of list): Group of data to perform the pareto analysis on.
        - names (list): Names of the scores for analysis.

        Returns:
        - str: Name of the best-performing group or repeat.

        Details:
        - Utilizes Pareto analysis to determine the best performing group or repeat.
        """
        pareto = ParetoAnalysis(groups, names)
        pareto_results = pareto.get_results()
        best_group_name = pareto_results[0][0]
        return best_group_name

    def _compute_metrics(self, train_data, test_data, result_dicts, idx):
        for group_name in self.subgroup_names:
            results = self.compute_performance(
                list(self.merged_features[(idx, group_name)]),
                self.classifier,
                train_data,
                test_data,
            )

            if self.task == "classification":
                result_dicts[0][(idx, group_name)] = results["accuracy"]
                result_dicts[1][(idx, group_name)] = results["AUROC"]
                result_dicts[2][(idx, group_name)] = results[
                    "MAE"
                ]  # should be minimized
            elif self.task == "regression":
                result_dicts[0][(idx, group_name)] = results[
                    "MAE"
                ]  # should be minimized
                result_dicts[1][(idx, group_name)] = results["R2"]
                result_dicts[2][(idx, group_name)] = results[
                    "RMSE"
                ]  # should be minimized

            features_stability = [
                [
                    feature.get_name()
                    for feature in self.FS_subsets[(idx, method_name)]
                    if feature.get_selected()
                ]
                for method_name in group_name
            ]
            stability = self.compute_stability(features_stability)
            result_dicts[3][(idx, group_name)] = stability[0]

        return result_dicts

    def iterate_pipeline(self):
        """
        Runs the entire feature selection and classification pipeline.

        This method iterates through the pipeline for the specified number of repeats, performing:
        - Data splitting
        - Feature computation and merging
        - Performance evaluation
        - Calculation of mean metrics per repeat
        - Pareto analysis to identify best performing groups and repeats

        Returns:
        - tuple: A tuple containing the merged features, the best repeat, and the best group name.
        """
        # a list of dicts to store the 4 metrics (3 performance and one stability)
        result_dicts = [{} for _ in range(4)]

        # compute feautres subset, merging and store metrics in dict for each repeat
        for i in range(self.num_repeats):
            print(f"Start repeat {i}")
            train_data, test_data = self.get_data_split(test_size=0.20)
            self._compute_sbst_and_scores_per_method(train_data=train_data, idx=i)
            self._compute_merging(idx=i)
            result_dicts = self._compute_metrics(
                train_data=train_data,
                test_data=test_data,
                result_dicts=result_dicts,
                idx=i,
            )

        # this is list of groups where each groups are a list of mean metrics respecting the orders of result_dicts
        # first list of the list is : mean accs for group1, means AUROC for group2 etc
        list_of_means = self._calculate_means(result_dicts, self.subgroup_names)

        # taking the negative to maximize MAE and RMSE
        if self.task == "classification":
            mae_index = 2
            list_of_means = [
                [-mean if idx == mae_index else mean for idx, mean in enumerate(means)]
                for means in list_of_means
            ]
        elif self.task == "regression":
            mae_index = 0
            rmse_index = 2
            list_of_means = [
                [
                    -mean if idx in (mae_index, rmse_index) else mean
                    for idx, mean in enumerate(means)
                ]
                for means in list_of_means
            ]

        # add number of methods to maximize; ensuring diversity
        metrics = [
            mean + [len(group_name)]
            for mean, group_name in zip(list_of_means, self.subgroup_names)
        ]
        metrics_names = self.subgroup_names + ["number of methods"]

        # find the best group using average metrics
        best_group_name = self._compute_pareto_analysis(
            groups=list_of_means, names=metrics_names
        )

        best_group_metrics = self._extract_repeat_metrics(
            best_group_name, *result_dicts
        )

        # taking the negative to maximize MAE and RMSE
        if self.task == "classification":
            mae_index = 2
            best_group_metrics = [
                [
                    -metric if idx == mae_index else metric
                    for idx, metric in enumerate(group_metrics)
                ]
                for group_metrics in best_group_metrics
            ]
        elif self.task == "regression":
            mae_index = 0
            rmse_index = 2
            best_group_metrics = [
                [
                    -metric if idx in (mae_index, rmse_index) else metric
                    for idx, metric in enumerate(group_metrics)
                ]
                for group_metrics in best_group_metrics
            ]

        # find the best repeat using metrics from best group only
        best_repeat = self._compute_pareto_analysis(
            groups=best_group_metrics,
            names=[str(i) for i in range(self.num_repeats)],
        )

        return (
            self.merged_features[(int(best_repeat), best_group_name)],
            best_repeat,
            best_group_name,
        )

    @staticmethod
    def _extract_features(data):
        """
        Extracts features from the given data.
        """
        if isinstance(data, pd.DataFrame):
            columns = [
                col for col in data.columns if col != "target"
            ]  # Exclude 'target' column
            return [Feature(name) for name in columns]  # Create Feature objects
        else:
            raise ValueError(
                "Data format not supported. Please provide a pandas DataFrame."
            )

    def _generate_subgroup_names(self):
        """
        Generates subgroup names based on feature selection methods.
        """
        fs_method_names = [
            fs_method.__name__.replace("feature_selection_", "")
            for fs_method in self.fs_methods
        ]
        subgroup_names = []

        # Generate combinations of FS method names with minimum length 2
        for r in range(2, len(fs_method_names) + 1):
            subgroup_names.extend(combinations(fs_method_names, r))

        return subgroup_names

    def get_feature_names(self, index_list):
        """
        Retrieves feature names based on provided indices.
        """
        return list(self.data.columns[index_list])

    def get_data_split(self, test_size):
        """
        Splits the data into training and testing sets.
        """
        if self.task == "classification":
            train_data, test_data = train_test_split(
                self.data,
                test_size=test_size,
                random_state=1,
                stratify=self.data["target"],
            )
            return train_data, test_data
        elif self.task == "regression":
            train_data, test_data = train_test_split(
                self.data, test_size=test_size, random_state=1
            )
        else:
            raise ValueError("Unsupported task type")
        return train_data, test_data

    @staticmethod
    def _get_X_y(data):
        """
        Extracts features (X) and target (y) from the given data.
        """
        # test to assert data is a dataframe
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                "Data format not supported. Please provide a pandas DataFrame."
            )
        if "target" not in data.columns:
            raise ValueError(
                "'target' column not found in the data. Please rename your dependant variable to 'target'"
            )
        X = data.drop(columns=["target"])
        y = data["target"]
        return X, y

    def _compute_features(self, selected_features_indices, feature_scores):
        """
        Computes features based on selected indices and their scores.
        """
        all_features = []
        feature_names = [col for col in self.data.columns if col != "target"]
        for idx, name in enumerate(feature_names):
            if idx in selected_features_indices:
                if feature_scores is None:
                    feature = Feature(name, score=None, selected=True)
                else:
                    feature = Feature(name, score=feature_scores[idx], selected=True)
            else:
                if feature_scores is None:
                    feature = Feature(name, score=None, selected=False)
                else:
                    feature = Feature(name, score=feature_scores[idx], selected=False)

            all_features.append(feature)

        return all_features

    @staticmethod
    def _calculate_means(list_of_dicts, group_names):
        """
        Calculate means for each group across dictionaries in a list of dictionaries.

        Args:
        - list_of_dicts (list): A list of dictionaries containing key-value pairs.
        - group_names (list): A list of group names, where each group name is a tuple of strings.

        Returns:
        - means_list (list): A list of lists containing means for each group across dictionaries.
        """
        means_list = []

        for group_name in group_names:
            group_means = []
            for d in list_of_dicts:
                group_values = [
                    value for key, value in d.items() if key[1] == group_name
                ]
                group_mean = (
                    np.mean(group_values) if group_values else np.nan
                )  # Use np.nan if no values found
                group_means.append(group_mean)
            means_list.append(group_means)

        return means_list

    @staticmethod
    def _extract_repeat_metrics(group_name, *result_dicts):
        # Find all unique indices
        indices = sorted(set(key[0] for key in result_dicts[0].keys()))
        # Create a 2D array to store the metrics for the fixed group name
        result_array = []
        for idx in indices:
            row = [d.get((idx, group_name)) for d in result_dicts]
            result_array.append(row)

        return result_array

    def compute_performance(self, features, classifier, train_data, test_data):
        if "target" not in train_data.columns:
            raise ValueError(
                "'target' column not found in the data. Please rename your dependant variable to 'target'"
            )

        sliced_train_data = train_data[features + ["target"]]
        sliced_test_data = test_data[features + ["target"]]

        X_train, y_train = self._get_X_y(data=sliced_train_data)
        X_test, y_test = self._get_X_y(data=sliced_test_data)
        return compute_performance_metrics(
            classifier, self.task, X_train, y_train, X_test, y_test
        )

    @staticmethod
    def compute_stability(features_list):
        return compute_stability_metrics(features_list)

    def __str__(self):
        return (
            f"Feature selection pipeline with : "
            f"merging strategy : {self.merging_strategy}, "
            f"feature selection methods: {self.fs_methods}, "
            f"Number of repeats: {self.num_repeats}"
        )
