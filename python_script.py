import subprocess
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

# List of files to run in order
files_to_run = [
    "data_collection.py",
    "data_cleaning.py",
    "notebooks/feature_and_eda.ipynb",
    "notebooks/race_winner_model.ipynb",
    "notebooks/lap_time_model.ipynb",
    "notebooks/next_race_predict.ipynb",
    "app.py"
]

def run_python_file(file_path):
    print(f"Running Python script: {file_path}")
    result = subprocess.run(["python", file_path], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

def run_notebook(file_path):
    print(f"Running Notebook: {file_path}")
    with open(file_path) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        try:
            ep.preprocess(nb, {'metadata': {'path': os.path.dirname(file_path) or '.'}})
            print(f"Notebook {file_path} executed successfully.")
        except Exception as e:
            print(f"Error running notebook {file_path}: {e}")

for file in files_to_run:
    if file.endswith(".py"):
        run_python_file(file)
    elif file.endswith(".ipynb"):
        run_notebook(file)
    else:
        print(f"Skipping unsupported file type: {file}")
