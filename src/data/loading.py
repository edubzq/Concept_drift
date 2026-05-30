import pandas as pd

def load_blocks(csv_path):
    df = pd.read_csv(csv_path)

    feature_cols = [
        c for c in df.columns
        if c not in ["target", "block", "timestamp", "concept"]
    ]

    chunks = []
    for block in sorted(df["block"].unique()):
        block_df = df[df["block"] == block]
        X = block_df[feature_cols]
        y = block_df["target"].values
        chunks.append((X, y))

    return chunks