import pandas as pd
import yaml


def read_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    validate_config(config)  # Validate the original config dictionary
    return config


def validate_config(config):
    # Validate classifier
    if config['classifier']['value'] not in config['classifier']['valid_values']:
        raise ValueError("Invalid classifier specified in the configuration.")

    # Validate fs_methods
    if not isinstance(config['fs_methods']['value'], list) or len(config['fs_methods']['value']) < 2:
        raise ValueError("At least two feature selection method must be specified in the configuration.")
    for method in config['fs_methods']['value']:
        if method not in config['fs_methods']['valid_values']:
            raise ValueError("Invalid feature selection method specified in the configuration.")

    # Validate merging_strategy
    if config['merging_strategy']['value'] not in config['merging_strategy']['valid_values']:
        raise ValueError("Invalid merging strategy specified in the configuration.")

    # Validate num_repeats
    if 'num_repeats' in config:
        num_repeats = config['num_repeats']['value']
        if not isinstance(num_repeats, int) or num_repeats < 1 or num_repeats > 10:
            raise ValueError("Invalid value for num_repeats. It should be an integer between 1 and 10.")


def preprocess_data(data_file, metadata_file):
    data_path = '/home/arthur.babey/arthurbabey/Data/'
    data_df = pd.read_csv(data_path + data_file)
    meta_df = pd.read_csv(data_path + metadata_file)
    merged_df = pd.merge(data_df, meta_df, on='SAMPLE_ID')
    merged_df.rename(columns={'TARGET_VAR_BIN': 'target'}, inplace=True)

    # drop the continous target to use only categorical one
    if 'TARGET_VAR_NUM' in merged_df.columns:
        merged_df.drop(columns=['TARGET_VAR_NUM'], inplace=True)

    # LabelEncoding for target and other categorical variable
    label_encoder = LabelEncoder()
    for col in merged_df.select_dtypes(include=['object']):
        merged_df[col] = label_encoder.fit_transform(merged_df[col])

    return merged_df

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