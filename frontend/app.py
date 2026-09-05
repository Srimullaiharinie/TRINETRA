import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import torch
import matplotlib.pyplot as plt


# ==================================================
# ADD BACKEND TO PYTHON PATH
# ==================================================

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "backend"
        )
    )
)

from features import (
    FEATURE_COLUMNS,
    clean_data,
    create_binary_label,
    normalize_features,
    create_sequences
)

from predictor import AttackPredictor, get_attack_stage


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="TRINETRA",
    page_icon="🛡️",
    layout="wide"
)


# ==================================================
# HEADER
# ==================================================

st.title("🛡️ TRINETRA")

st.subheader(
    "AI-Based Network Attack Forecasting & Proactive Cyber Defence"
)

st.markdown(
    """
    **Temporal World Model for Predicting Attacker Progression**
    """
)

st.divider()


# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_predictor():

    return AttackPredictor(
        model_path="models/world_model.pt",
        sequence_length=10
    )


try:

    predictor = load_predictor()

    model_status = "🟢 Model Loaded"

except Exception as e:

    st.error(
        f"Model loading failed: {e}"
    )

    st.stop()


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header("TRINETRA Controls")

st.sidebar.success(model_status)

uploaded_file = st.sidebar.file_uploader(
    "Upload Network Traffic CSV",
    type=["csv"]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    ### World Model

    **Temporal sequence:** 10 flows

    **Forecast horizon:** 5 steps

    **Model:** GRU

    **Mode:** Offline inference
    """
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Input:** Network traffic CSV

    **Output:**
    - Infiltration risk
    - Future state forecast
    - Attack stage
    - Feature attribution
    - Defender recommendation
    """
)


# ==================================================
# DATA PROCESSING FUNCTION
# ==================================================

def process_uploaded_data(file):

    df = pd.read_csv(file)

    original_shape = df.shape

    # Replace infinite values
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Select only features required by the trained model
    available = [
        col
        for col in FEATURE_COLUMNS
        if col in df.columns
    ]

    if len(available) == 0:

        raise ValueError(
            "None of the required model features were found in the uploaded CSV."
        )

    df = df[
        available
    ].copy()

    # Convert to numeric
    for col in available:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Missing values
    df = df.fillna(0)

    # Standardize features
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    X = scaler.fit_transform(df)

    X = np.nan_to_num(
        X,
        nan=0,
        posinf=0,
        neginf=0
    )

    return df, X, original_shape


# ==================================================
# INITIAL SCREEN
# ==================================================

if uploaded_file is None:

    st.info(
        "Upload a network traffic CSV from the sidebar to start forecasting."
    )

    st.markdown("### Prototype Pipeline")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "1",
        "Traffic"
    )

    col2.metric(
        "2",
        "Features"
    )

    col3.metric(
        "3",
        "Temporal Model"
    )

    col4.metric(
        "4",
        "Forecast"
    )

    col5.metric(
        "5",
        "Defence"
    )

    st.divider()

    st.markdown(
        """
        ### How TRINETRA Works

        **Network Traffic**
        ↓
        **Feature Extraction**
        ↓
        **Temporal GRU World Model**
        ↓
        **K-Step Future Simulation**
        ↓
        **Infiltration Risk Forecast**
        ↓
        **Attack Stage Mapping**
        ↓
        **Defender Decision Support**
        """
    )

    st.stop()


# ==================================================
# PROCESS DATA
# ==================================================

try:

    df_features, X, original_shape = process_uploaded_data(
        uploaded_file
    )

except Exception as e:

    st.error(
        f"Unable to process uploaded CSV: {e}"
    )

    st.stop()


# ==================================================
# DATASET INFORMATION
# ==================================================

st.header("📊 Network Traffic Analysis")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Traffic Records",
    f"{original_shape[0]:,}"
)

c2.metric(
    "Model Features",
    len(df_features.columns)
)

c3.metric(
    "Temporal Window",
    "10"
)

c4.metric(
    "Forecast Horizon",
    "5 Steps"
)


# ==================================================
# TEMPORAL WINDOW
# ==================================================

SEQUENCE_LENGTH = 10

if len(X) < SEQUENCE_LENGTH:

    st.error(
        "Not enough traffic records to create a temporal sequence."
    )

    st.stop()


sequence = X[
    -SEQUENCE_LENGTH:
]


# ==================================================
# FORECAST
# ==================================================

try:

    risks, future_states = predictor.forecast(
        sequence,
        steps=5
    )

except Exception as e:

    st.error(
        f"Prediction failed: {e}"
    )

    st.stop()


# ==================================================
# CURRENT RISK
# ==================================================

current_risk = float(
    risks[0]
)

risk_percent = current_risk * 100


# ==================================================
# ATTACK STAGE
# ==================================================

stage = get_attack_stage(
    current_risk
)


# ==================================================
# RISK LEVEL
# ==================================================

if risk_percent >= 80:

    risk_level = "HIGH"

elif risk_percent >= 60:

    risk_level = "ELEVATED"

elif risk_percent >= 40:

    risk_level = "MODERATE"

else:

    risk_level = "LOW"


# ==================================================
# RISK DISPLAY
# ==================================================

st.divider()

st.subheader(
    "🚨 Infiltration Risk"
)

risk_col, stage_col, level_col = st.columns(3)

with risk_col:

    st.metric(
        "Current Forecast Risk",
        f"{risk_percent:.1f}%"
    )

    st.progress(
        min(
            max(
                current_risk,
                0.0
            ),
            1.0
        )
    )


with stage_col:

    st.metric(
        "Predicted Attack Stage",
        stage
    )


with level_col:

    st.metric(
        "Risk Level",
        risk_level
    )


# ==================================================
# FORECAST CHART
# ==================================================

st.subheader(
    "🔮 K-Step Attack Forecast"
)

forecast_df = pd.DataFrame(
    {
        "Future Step": [
            "T+1",
            "T+2",
            "T+3",
            "T+4",
            "T+5"
        ],
        "Risk": [
            float(r) * 100
            for r in risks
        ]
    }
)


fig, ax = plt.subplots(
    figsize=(10, 4)
)

ax.plot(
    forecast_df["Future Step"],
    forecast_df["Risk"],
    marker="o"
)

ax.set_ylim(
    0,
    100
)

ax.set_ylabel(
    "Predicted Infiltration Risk (%)"
)

ax.set_xlabel(
    "Future Time Window"
)

ax.set_title(
    "Predicted Network Attack Progression"
)

ax.grid(
    True,
    alpha=0.3
)

st.pyplot(
    fig,
    width="stretch"
)

plt.close(fig)


# ==================================================
# FORECAST TABLE
# ==================================================

st.dataframe(
    forecast_df.round(2),
    width="stretch",
    hide_index=True
)


# ==================================================
# FUTURE NETWORK STATES
# ==================================================

st.subheader(
    "🧠 Predicted Future Network States"
)

try:

    state_df = pd.DataFrame(
        future_states,
        columns=FEATURE_COLUMNS
    )

    state_df.index = [
        "T+1",
        "T+2",
        "T+3",
        "T+4",
        "T+5"
    ]

    st.dataframe(
        state_df.round(3),
        width="stretch"
    )

except Exception as e:

    st.warning(
        f"Unable to display future state features: {e}"
    )


# ==================================================
# MODEL-BASED FEATURE ATTRIBUTION
# ==================================================

st.subheader(
    "🔍 Top Contributing Traffic Features"
)

st.caption(
    "Model-derived feature attribution based on the gradient of the "
    "predicted infiltration risk with respect to the latest traffic state."
)

try:

    feature_importance = predictor.explain_prediction(
        sequence
    )

    # Top 5 features
    top_features = feature_importance[:5]

    feature_df = pd.DataFrame(
        {
            "Feature": [
                item[0]
                for item in top_features
            ],
            "Feature Attribution": [
                float(item[1])
                for item in top_features
            ]
        }
    )

    st.dataframe(
        feature_df.round(4),
        width="stretch",
        hide_index=True
    )

except Exception as e:

    st.warning(
        f"Feature attribution unavailable: {e}"
    )

    # Safe fallback
    latest_values = df_features.iloc[-1]

    feature_scores = (
        latest_values.abs()
        .sort_values(
            ascending=False
        )
        .head(5)
    )

    fallback_df = pd.DataFrame(
        {
            "Feature": feature_scores.index,
            "Feature Magnitude": feature_scores.values
        }
    )

    st.dataframe(
        fallback_df,
        width="stretch",
        hide_index=True
    )


# ==================================================
# ATTACK STAGE INTERPRETATION
# ==================================================

st.subheader(
    "🎯 Predicted Attack Stage"
)

stage_description = {

    "Normal":
        "No strong malicious trajectory detected in the current network state.",

    "Reconnaissance":
        "Traffic behaviour indicates possible scanning or reconnaissance activity.",

    "Initial Access":
        "Traffic behaviour indicates a potential transition toward initial compromise.",

    "Lateral Movement":
        "Predicted trajectory indicates possible movement between internal network resources.",

    "Command & Control":
        "Predicted trajectory indicates behaviour consistent with possible command-and-control activity."
}


st.info(
    f"**{stage}** — {stage_description.get(stage, 'Potential malicious progression detected.')}"
)


# ==================================================
# MITRE ATT&CK-STYLE MAPPING
# ==================================================

st.subheader(
    "🧩 ATT&CK Stage Mapping"
)

attack_stages = [
    "Reconnaissance",
    "Initial Access",
    "Lateral Movement",
    "Command & Control",
    "Exfiltration"
]

stage_table = pd.DataFrame(
    {
        "Attack Stage": attack_stages,
        "Status": [
            "Predicted"
            if s == stage
            else "Not currently predicted"
            for s in attack_stages
        ]
    }
)

st.dataframe(
    stage_table,
    width="stretch",
    hide_index=True
)


# ==================================================
# FLAGGED TRAFFIC
# ==================================================

st.subheader(
    "🚩 Latest Traffic Features"
)

st.dataframe(
    df_features.tail(10),
    width="stretch"
)


# ==================================================
# DEFENDER RECOMMENDATION
# ==================================================

st.subheader(
    "🛡️ Defender Decision Support"
)

if risk_percent >= 80:

    st.error(
        "HIGH RISK — Investigate active connections, "
        "isolate suspicious hosts and review command-and-control traffic."
    )

elif risk_percent >= 60:

    st.warning(
        "ELEVATED RISK — Investigate suspicious flows, "
        "port activity and lateral communication."
    )

elif risk_percent >= 40:

    st.warning(
        "MODERATE RISK — Increase monitoring of the affected traffic."
    )

else:

    st.success(
        "LOW RISK — No strong future infiltration trajectory detected."
    )


# ==================================================
# PROTOTYPE SUMMARY
# ==================================================

st.divider()

st.subheader(
    "📋 Forecast Summary"
)

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

summary_col1.metric(
    "Current Risk",
    f"{risk_percent:.1f}%"
)

summary_col2.metric(
    "Trend",
    "Increasing"
    if len(risks) > 1 and risks[-1] > risks[0]
    else "Stable / Decreasing"
)

summary_col3.metric(
    "Predicted Stage",
    stage
)

summary_col4.metric(
    "Model",
    "GRU World Model"
)


# ==================================================
# FOOTER
# ==================================================

st.caption(
    "TRINETRA — Offline AI Network Attack Forecasting Prototype"
)