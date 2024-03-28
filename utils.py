import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

import src.feature_selection_methods as feature_selection_methods
import src.merging_strategy_methods as merging_strategy_methods


def read_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # get function from strings for fs_methods and merging_strategy_methods
    fs_methods = []
    for method_name in config["fs_methods"]["value"]:
        # Assuming feature selection functions are in a module named 'feature_selection'
        fs_method = getattr(feature_selection_methods, method_name)
        fs_methods.append(fs_method)

    config["fs_methods"]["value"] = fs_methods
    strategy_name = config["merging_strategy"]["value"]
    config["merging_strategy"]["value"] = getattr(
        merging_strategy_methods, strategy_name
    )
    validate_config(config)
    return config


def validate_config(config):
    # Validate classifier
    if config["classifier"]["value"] not in config["classifier"]["valid_values"]:
        raise ValueError("Invalid classifier specified in the configuration.")

    # Validate fs_methods
    if (
        not isinstance(config["fs_methods"]["value"], list)
        or len(config["fs_methods"]["value"]) < 2
    ):
        raise ValueError(
            "At least two feature selection method must be specified in the configuration."
        )
    for method in config["fs_methods"]["value"]:
        if not callable(method):
            raise ValueError(
                "Feature selection method specified in the configuration is not callable."
            )
        if method.__name__ not in config["fs_methods"]["valid_values"]:
            raise ValueError(
                "Invalid feature selection method specified in the configuration."
            )

    # Validate merging_strategy
    if not callable(config["merging_strategy"]["value"]):
        raise ValueError("Merging strategy specified is not callable.")
    if not isinstance(config["merging_strategy"]["value"].__name__, str):
        raise ValueError(
            "A single string should be specified in the configuration file."
        )
    if (
        config["merging_strategy"]["value"].__name__
        not in config["merging_strategy"]["valid_values"]
    ):
        raise ValueError(
            "Invalid merging strategy name specified in the configuration."
        )

    # Validate num_repeats
    if "num_repeats" in config:
        num_repeats = config["num_repeats"]["value"]
        if not isinstance(num_repeats, int) or num_repeats < 1 or num_repeats > 10:
            raise ValueError(
                "Invalid value for num_repeats. It should be an integer between 1 and 10."
            )


def preprocess_exp1(data_file, metadata_file, normalize, task):
    data_df = pd.read_csv(data_file, index_col=0)
    meta_df = pd.read_csv(metadata_file, index_col=0)
    merged_df = pd.merge(data_df, meta_df, on="SAMPLE_ID")

    if task == "regression":
        print(task)
        merged_df.rename(columns={"TARGET_VAR_NUM": "target"}, inplace=True)
        merged_df.drop(
            columns=["SAMPLE_ID", "TARGET_VAR_BIN"],
            inplace=True,
            errors="ignore",
        )
        categorical_cols = ["IND_VAR_1", "IND_VAR_2", "IND_VAR_3"]
    elif task == "classification":
        print(task)
        merged_df.rename(columns={"TARGET_VAR_BIN": "target"}, inplace=True)
        merged_df.drop(
            columns=["SAMPLE_ID", "TARGET_VAR_NUM"],
            inplace=True,
            errors="ignore",
        )
        categorical_cols = ["target", "IND_VAR_1", "IND_VAR_2", "IND_VAR_3"]

    # Drop rows containing NaN values
    merged_df.dropna(inplace=True)
    # print("Number of samples after dropping NaN values:", len(merged_df))
    # print("Resulting shape of the DataFrame (samples, features):", merged_df.shape[0], merged_df.shape[1] - 1)

    # Label encode categorical variables
    label_encoder = LabelEncoder()
    for col in categorical_cols:
        if col in merged_df.columns:
            merged_df[col] = label_encoder.fit_transform(merged_df[col])

    # Standard scale non-categorical variables
    if normalize:
        non_categorical_cols = merged_df.columns.difference(categorical_cols)
        non_categorical_cols = non_categorical_cols.drop("target", errors="ignore")
        scaler = StandardScaler()
        merged_df[non_categorical_cols] = scaler.fit_transform(
            merged_df[non_categorical_cols]
        )

    return merged_df


def create_toy_dataset(num_samples=1000, num_features=100, random_seed=42):
    np.random.seed(random_seed)  # Set the random seed for NumPy
    # Generating random data for features and target column
    data = {f"feature_{i}": np.random.rand(num_samples) for i in range(num_features)}
    data["target"] = np.random.randint(
        0, 2, size=num_samples
    )  # Binary classification target column
    # Creating a DataFrame from the generated data
    return pd.DataFrame(data)


def GSE_dataset(csv_path):
    breast_data = pd.read_csv(csv_path)
    data = breast_data.drop(columns=["samples"])
    data.rename(columns={"type": "target_raw"}, inplace=True)
    label_encoder = LabelEncoder()
    data["target"] = label_encoder.fit_transform(data["target_raw"])

    print("Mapping of original values to encoded integers:")
    for original_value, encoded_value in zip(data["target_raw"], data["target"]):
        print(f"{original_value} -> {encoded_value}")
    data = data.drop(columns=["target_raw"])

    return data


def leukemia_dataset(csv_path):
    data = pd.read_csv(csv_path)
    data.rename(columns={"cancer": "target_raw"}, inplace=True)
    label_encoder = LabelEncoder()
    data["target"] = label_encoder.fit_transform(data["target_raw"])

    # Separating features from the target
    features = data.drop(columns=["target_raw", "target"])

    # Applying MinMax scaling to the features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_features = pd.DataFrame(scaled_features, columns=features.columns)

    # Combine scaled features with the target and encoded label
    data_processed = pd.concat([scaled_features, data["target"]], axis=1)

    print("Mapping of original values to encoded integers:")
    for original_value, encoded_value in zip(data["target_raw"], data["target"]):
        print(f"{original_value} -> {encoded_value}")

    return data_processed
