# EnsembleFeatureSelection
Feature Ensemble selection pipeline

**Introduction:**

Implementation of an Ensemble Feature Selection pipeline.


**Information:**

1. **Installation:**

   - Clone the repository to your local machine:

   - Install the required dependencies


2. **File Structure:**

- **src:** Contains all the files employed by the pipeline.
  - `main.py`: Script to execute the pipeline.
  - `config.yaml`: Configuration file for customization.
  - `requirements.txt` : python dependency
  - `Dockerfile`: Docker file to buid a docker image

3. **Usage:**
   
   - Modify the `config.yaml` file to choose the pipeline parameter
   - If needed overwrite the preprocess function in `utils.py` to create a dataframe from your data, pipeline method accept data as a dataframe and it should have a 'target' column.
   - Run main.py : `python main.py`
   - Results will be stored in a designated folder for further analysis.

