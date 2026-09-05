import sys
import os

# Allow importing files from backend
sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from features import prepare_dataset
from model import WorldModel


# -----------------------------
# Configuration
# -----------------------------

DATA_PATH = "data/sample_traffic.csv"

SEQUENCE_LENGTH = 10

EPOCHS = 5

BATCH_SIZE = 128

LEARNING_RATE = 0.001


# -----------------------------
# Load data
# -----------------------------

print("Preparing dataset...")

X, y, scaler = prepare_dataset(
    DATA_PATH,
    sequence_length=SEQUENCE_LENGTH
)


# Convert to PyTorch tensors

X = torch.tensor(
    X,
    dtype=torch.float32
)

y = torch.tensor(
    y,
    dtype=torch.float32
)


print("X:", X.shape)
print("y:", y.shape)


# -----------------------------
# Train/Test split
# -----------------------------

split = int(
    len(X) * 0.8
)

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]


# -----------------------------
# DataLoader
# -----------------------------

train_dataset = TensorDataset(
    X_train,
    y_train
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# -----------------------------
# Model
# -----------------------------

model = WorldModel(
    input_size=X.shape[2],
    hidden_size=64
)


# -----------------------------
# Loss
# -----------------------------

risk_loss = nn.BCELoss()

state_loss = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# Training
# -----------------------------

print("\nTraining TRINETRA World Model...\n")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for batch_X, batch_y in train_loader:

        optimizer.zero_grad()

        predicted_state, risk = model(
            batch_X
        )

        # Risk prediction loss
        loss_risk = risk_loss(
            risk.squeeze(),
            batch_y
        )

        # We don't have a separate future
        # feature target yet, so use a
        # small auxiliary objective.
        last_state = batch_X[:, -1, :]

        loss_state = state_loss(
            predicted_state,
            last_state
        )

        loss = (
            loss_risk +
            0.1 * loss_state
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = (
        total_loss /
        len(train_loader)
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Loss: {avg_loss:.4f}"
    )


# -----------------------------
# Save model
# -----------------------------

os.makedirs(
    "models",
    exist_ok=True
)

torch.save(
    model.state_dict(),
    "models/world_model.pt"
)

print("\n✅ MODEL TRAINING COMPLETE")

print(
    "Saved: models/world_model.pt"
)