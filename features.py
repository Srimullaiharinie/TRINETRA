import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "SYN Flag Count",
    "RST Flag Count",
    "ACK Flag Count",
    "Average Packet Size",
    "Down/Up Ratio",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "Active Mean"
]


def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Loaded dataset:", df.shape)
    return df


def clean_data(df):

    df = df.replace([np.inf, -np.inf], np.nan)

    available = [
        col for col in FEATURE_COLUMNS
        if col in df.columns
    ]

    df = df[available + ["Label"]].copy()

    for col in available:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df[available] = df[available].fillna(0)

    df[available] = df[available].clip(
        -1e10,
        1e10
    )

    return df


def create_binary_label(df):

    df["Target"] = (
        df["Label"]
        .astype(str)
        .str.upper()
        .str.strip()
        .ne("BENIGN")
        .astype(int)
    )

    return df


def normalize_features(df):

    features = [
        col for col in FEATURE_COLUMNS
        if col in df.columns
    ]

    scaler = StandardScaler()

    X = scaler.fit_transform(
        df[features]
    )

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    y = df["Target"].values

    return X, y, scaler


def create_sequences(X, y, sequence_length=10):

    sequences = []
    targets = []

    for i in range(len(X) - sequence_length):

        sequence = X[
            i:i + sequence_length
        ]

        target = y[
            i + sequence_length
        ]

        sequences.append(sequence)
        targets.append(target)

    return (
        np.array(sequences),
        np.array(targets)
    )


def prepare_dataset(
    file_path,
    sequence_length=10
):

    print("Loading data...")

    df = load_data(file_path)

    print("Cleaning data...")

    df = clean_data(df)

    print("Creating labels...")

    df = create_binary_label(df)

    print("\nLabel distribution:")
    print(df["Target"].value_counts())

    print("\nNormalizing features...")

    X, y, scaler = normalize_features(df)

    print("Creating temporal sequences...")

    X_seq, y_seq = create_sequences(
        X,
        y,
        sequence_length
    )

    print("\nSequence shape:", X_seq.shape)
    print("Target shape:", y_seq.shape)

    return X_seq, y_seq, scaler


if __name__ == "__main__":

    DATA_PATH = "data/sample_traffic.csv"

    X, y, scaler = prepare_dataset(
        DATA_PATH,
        sequence_length=10
    )

    print("\nSUCCESS!")
    print("X shape:", X.shape)
    print("y shape:", y.shape)