import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
from src.feature_selection_pipeline import FeatureSelectionPipeline

from src.feature_selection_methods import *
from src.merging_strategy_methods import *

import os

# Restrict the process to use only a specific number of CPU cores
cores_to_use = 72  # Change this number to the desired core count
os.sched_setaffinity(0, range(cores_to_use))


def create_toy_dataset(num_samples=1000, num_features=100, random_seed=42):
    np.random.seed(random_seed)  # Set the random seed for NumPy
    # Generating random data for features and target column
    data = {
        f"feature_{i}": np.random.rand(num_samples) for i in range(num_features)
    }
    data['target'] = np.random.randint(0, 2, size=num_samples)  # Binary classification target column
    # Creating a DataFrame from the generated data
    return pd.DataFrame(data)

def GSE_dataset(csv_path):
    breast_data = pd.read_csv(csv_path)
    data = breast_data.drop(columns=['samples'])
    data.rename(columns={'type': 'target_raw'}, inplace=True)
    label_encoder = LabelEncoder()
    data['target'] = label_encoder.fit_transform(data['target_raw'])

    print("Mapping of original values to encoded integers:")
    for original_value, encoded_value in zip(data['target_raw'], data['target']):
        print(f"{original_value} -> {encoded_value}")
    data = data.drop(columns=['target_raw'])

    return data

def leukemia_dataset(csv_path):
    data = pd.read_csv(csv_path)
    data.rename(columns={'cancer': 'target_raw'}, inplace=True)
    label_encoder = LabelEncoder()
    data['target'] = label_encoder.fit_transform(data['target_raw'])

    # Separating features from the target
    features = data.drop(columns=['target_raw', 'target'])

    # Applying MinMax scaling to the features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_features = pd.DataFrame(scaled_features, columns=features.columns)

    # Combine scaled features with the target and encoded label
    data_processed = pd.concat([scaled_features, data['target']], axis=1)

    print("Mapping of original values to encoded integers:")
    for original_value, encoded_value in zip(data['target_raw'], data['target']):
        print(f"{original_value} -> {encoded_value}")

    return data_processed

if __name__ == "__main__":
    classifier = 'RF'
    fs_methods = [feature_selection_chi2, feature_selection_infogain, feature_selection_xgboost]
    merging_strategy = merging_strategy_union_of_pairwise_intersections
    num_repeats = 5

    print(f"Number of repeat is {num_repeats}")

    # Create a toy dataset
    # dataset = create_toy_dataset()
    dataset = GSE_dataset('./breast_cancer_data/Breast_GSE45827.csv')

    # Create and test the pipeline
    pipeline = FeatureSelectionPipeline(dataset, fs_methods, merging_strategy, classifier, num_repeats)
    print(pipeline)
    best_features, best_repeat, best_group_name = pipeline.iterate_pipeline()

    file_name = "GSE_result.txt"
    result_folder = 'result'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    file_path = os.path.join(result_folder, file_name)

    # Write the results to the file
    with open(file_path, "w") as file:
        file.write(f"The best features are {best_features}\n")
        file.write(f"Best repeat value: {best_repeat}\n")
        file.write(f"Best group name: {best_group_name}\n")

    print(f"Results written to {file_path}")
