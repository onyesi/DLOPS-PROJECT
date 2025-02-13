import mlflow
import subprocess

experiments = mlflow.search_experiments()
experiment = [exp for exp in experiments if exp.name == "dlops_experiment"][0]
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
latest_run = runs.iloc[-1]
loss = latest_run["metrics.loss"]

if loss > 0.5:  # Threshold for retraining
    print("Retraining model...")
    subprocess.run(["python", "model/train.py"])