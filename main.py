import os
from src.feature_selection_pipeline import FeatureSelectionPipeline
from src.feature_selection_methods import *
from utils import *

# Restrict the process to use only a specific number of CPU cores
cores_to_use = 72  # Change this number to the desired core count
os.sched_setaffinity(0, range(cores_to_use))


if __name__ == "__main__":

    config_file = 'config.yaml'
    params = read_config(config_file)
    classifier = params['classifier']['value']
    fs_methods = params['fs_methods']['value']
    merging_strategy = params['merging_strategy']['value']
    num_repeats = params.get('num_repeats', {'value': 1})['value']  # Default to 1 if not provided
    data_path = params['data_path']['value']
    result_path = params['result_path']['value']
    experiment_name = params['experiment_name']['value']

    dataset = preprocess_data(data_path+'EXP1_TRANSCRIPTOMICS.csv', data_path+'EXP1_METADATA.csv')
    pipeline = FeatureSelectionPipeline(dataset, fs_methods, merging_strategy, classifier, num_repeats)
    best_features, best_repeat, best_group_name = pipeline.iterate_pipeline()

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    file_path = os.path.join(result_path, experiment_name)

    # Write the results to the file
    with open(file_path, "w") as file:
        file.write(f"The best features are {best_features}\n")
        file.write(f"Best repeat value: {best_repeat}\n")
        file.write(f"Best group name: {best_group_name}\n")

    print(f"Results written to {file_path}")
