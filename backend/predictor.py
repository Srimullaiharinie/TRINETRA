import os
import sys
import numpy as np
import torch

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from model import WorldModel
from features import FEATURE_COLUMNS


class AttackPredictor:

    def __init__(
        self,
        model_path="models/world_model.pt",
        sequence_length=10
    ):

        self.sequence_length = sequence_length

        # Load checkpoint
        checkpoint = torch.load(
            model_path,
            map_location="cpu"
        )

        # Detect number of features used during training
        input_size = checkpoint[
            "encoder.weight"
        ].shape[1]

        print(
            f"Model expects {input_size} features"
        )

        self.input_size = input_size

        # Create model
        self.model = WorldModel(
            input_size=input_size,
            hidden_size=64
        )

        # Load trained weights
        self.model.load_state_dict(
            checkpoint
        )

        self.model.eval()

    # ==================================================
    # BASIC RISK PREDICTION
    # ==================================================

    def predict_risk(self, sequence):

        x = torch.tensor(
            sequence,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():

            future_state, risk = self.model(x)

        return (
            future_state.numpy()[0],
            float(risk.numpy()[0][0])
        )

        # ==================================================
    # GRADIENT-BASED FEATURE ATTRIBUTION
    # ==================================================

    def explain_prediction(self, sequence):

        """
        Calculates gradient-based feature attribution.

        Higher absolute gradient magnitude indicates
        stronger influence of a feature on the predicted
        infiltration risk.
        """

        # Convert sequence to tensor
        x = torch.tensor(
            sequence,
            dtype=torch.float32
        )

        # Add batch dimension
        x = x.unsqueeze(0)

        # Enable gradient calculation
        x.requires_grad_(True)

        # Clear previous gradients
        self.model.zero_grad()

        # Forward pass
        future_state, risk = self.model(x)

        # Select scalar risk prediction
        risk_value = risk[0, 0]

        # Backpropagate
        risk_value.backward()

        # Get gradients
        gradients = x.grad

        if gradients is None:
            raise RuntimeError(
                "Gradient calculation failed."
            )

        # Calculate mean absolute gradient
        # across all time steps
        feature_attribution = (
            gradients.abs()
            .mean(dim=1)
            .squeeze(0)
            .detach()
            .numpy()
        )

        # Get feature names
        feature_names = FEATURE_COLUMNS[:self.input_size]

        # Fallback names if required
        if len(feature_names) < self.input_size:

            feature_names = [
                f"Feature_{i}"
                for i in range(self.input_size)
            ]

        # Create feature-importance pairs
        feature_importance = list(
            zip(
                feature_names,
                feature_attribution
            )
        )

        # Sort highest to lowest
        feature_importance.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return feature_importance

    # ==================================================
    # K-STEP FORECAST
    # ==================================================

    def forecast(
        self,
        sequence,
        steps=5
    ):

        current_sequence = sequence.copy()

        risks = []
        future_states = []

        for step in range(steps):

            future_state, risk = self.predict_risk(
                current_sequence
            )

            risks.append(risk)

            future_states.append(
                future_state
            )

            # Shift temporal window
            current_sequence = np.vstack(
                [
                    current_sequence[1:],
                    future_state
                ]
            )

        return risks, future_states


# ==================================================
# ATTACK STAGE MAPPING
# ==================================================

def get_attack_stage(risk):

    if risk < 0.20:

        return "Normal"

    elif risk < 0.40:

        return "Reconnaissance"

    elif risk < 0.60:

        return "Initial Access"

    elif risk < 0.80:

        return "Lateral Movement"

    else:

        return "Command & Control"


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    print(
        "Loading TRINETRA World Model..."
    )

    predictor = AttackPredictor()

    print(
        "Model loaded successfully."
    )

    print(
        "Predictor ready."
    )