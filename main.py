import os
import pickle
import shutil
from src.feature_selection_pipeline import FeatureSelectionPipeline
from utils import *

# Restrict the process to use only a specific number of CPU cores
cores_to_use = 72  # Change this number to the desired core count
os.sched_setaffinity(0, range(cores_to_use))


if __name__ == "__main__":

    # parse config file
    config_file = 'config.yaml'
    params = read_config(config_file)
    classifier = params['classifier']['value']
    fs_methods = params['fs_methods']['value']
    merging_strategy = params['merging_strategy']['value']
    num_repeats = params.get('num_repeats', {'value': 1})['value']  # Default to 1 if not provided
    data_path = params['data_path']['value']
    result_path = params['result_path']['value']
    experiment_name = params['experiment_name']['value']

    # create results folders and save config
    experiment_folder = os.path.join(result_path, experiment_name)
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    shutil.copy(config_file, os.path.join(experiment_folder, "config.yaml"))

    # Run pipeline
    dataset = preprocess_data(data_path+'EXP1_PROTEOMICS.csv', data_path+'EXP1_METADATA.csv')
    pipeline = FeatureSelectionPipeline(dataset, fs_methods, merging_strategy, classifier, num_repeats)
    best_features, best_repeat, best_group_name = pipeline.iterate_pipeline()

    # save results
    result_file_path = os.path.join(experiment_folder, "results.txt")
    with open(result_file_path, "w") as file:
        file.write(f"The best features are {best_features}\n")
        file.write(f"Best repeat value: {best_repeat}\n")
        file.write(f"Best group name: {best_group_name}\n")
    result_pickle_path = os.path.join(experiment_folder, "results.pkl")
    with open(result_pickle_path, "wb") as file:
        pickle.dump(best_features, file)
        pickle.dump(best_repeat, file)
        pickle.dump(best_group_name, file)

    print(f"Results written to {experiment_folder}")
