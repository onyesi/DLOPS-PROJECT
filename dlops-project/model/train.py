import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from model import model  # Importing the model

# Load the model
def train():
    model_instance = model.SimpleNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_instance.parameters(), lr=0.01)

    for epoch in range(10):
        inputs = torch.randn(32, 10)
        targets = torch.randn(32, 1)
        outputs = model_instance(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch [{epoch+1}/10], Loss: {loss.item()}")

    # Log to MLflow
    mlflow.set_experiment("dlops_experiment")
    with mlflow.start_run():
        mlflow.pytorch.log_model(model_instance, "model")
        mlflow.log_metric("loss", loss.item())

if __name__ == "__main__":
    train()
