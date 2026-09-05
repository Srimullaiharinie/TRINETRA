import torch
import torch.nn as nn


class WorldModel(nn.Module):

    def __init__(
        self,
        input_size=16,
        hidden_size=64
    ):
        super().__init__()

        # Convert input traffic features
        # into a learned representation
        self.encoder = nn.Linear(
            input_size,
            hidden_size
        )

        # Learn temporal network behaviour
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Predict the next network state
        self.state_predictor = nn.Linear(
            hidden_size,
            input_size
        )

        # Predict malicious/infiltration risk
        self.risk_predictor = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        # Feature encoding
        x = torch.relu(
            self.encoder(x)
        )

        # Temporal modelling
        output, hidden = self.gru(x)

        # Last observed network state
        last_state = output[:, -1, :]

        # Predicted future state
        future_state = self.state_predictor(
            last_state
        )

        # Risk probability
        risk = torch.sigmoid(
            self.risk_predictor(
                last_state
            )
        )

        return future_state, risk