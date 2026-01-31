import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Sandy’s Law — Phase Coherence (Toy 3)",
    layout="wide",
)

st.title("Sandy’s Law — Phase Coherence Instrument")
st.caption(
    "Phase-events only • No time axis • Shared-time emerges via square crowding"
)

# =====================================================
# UTILITIES
# =====================================================
def normalize_col(c):
    c = c.strip().lower()
    if c in ["sigma", "sig", "Σ", "s"]:
        return "sigma"
    if c in ["z", "trap", "trapstrength"]:
        return "z"
    if c in ["event", "event_id", "id"]:
        return "event_id"
    return c


def load_phase_csv(text):
    df = pd.read_csv(StringIO(text))
    df.columns = [normalize_col(c) for c in df.columns]

    if "z" not in df.columns or "sigma" not in df.columns:
        raise ValueError("CSV must contain columns: z, sigma")

    if "event_id" not in df.columns:
        df["event_id"] = np.arange(len(df))

    df = df[["event_id", "z", "sigma"]]
    df["z"] = pd.to_numeric(df["z"], errors="coerce")
    df["sigma"] = pd.to_numeric(df["sigma"], errors="coerce")
    df = df.dropna()

    df["z"] = df["z"].clip(0, 1)
    df["sigma"] = df["sigma"].clip(0, 1)
    df["event_id"] = df["event_id"].astype(int)

    return df.sort_values("event_id").reset_index(drop=True)


def assign_squares(df, bins):
    eps = 1e-9
    zb = np.floor(df["z"] * (bins - eps)).astype(int)
    sb = np.floor(df["sigma"] * (bins - eps)).astype(int)
    zb = zb.clip(0, bins - 1)
    sb = sb.clip(0, bins - 1)

    out = df.copy()
    out["z_bin"] = zb
    out["s_bin"] = sb
    out["cell"] = zb.astype(str) + "_" + sb.astype(str)
    return out


def coherence_C(occupancy):
    N = occupancy.sum()
    if N <= 1:
        return 0.0
    return (np.sum(occupancy ** 2) - N) / (N * (N - 1))


def classify_regime(C):
    if C >= 0.75:
        return "Strong macroscopic coherence"
    if C >= 0.50:
        return "Emergent coherence"
    if C >= 0.25:
        return "Weak coherence"
    return "Independent regime (exhaust-free)"


def run_length_stats(states):
    runs = []
    cur = 1
    for i in range(1, len(states)):
        if states[i] == states[i - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return np.mean(runs), np.max(runs)


def build_states(df_binned, min_occ):
    counts = df_binned["cell"].value_counts()
    occ = df_binned["cell"].map(counts).values

    states = np.full(len(occ), -1)
    states[(occ >= min_occ) & (occ < 2 * min_occ)] = 0
    states[occ >= 2 * min_occ] = 1
    return states


# =====================================================
# CSV INPUT
# =====================================================
st.header("1️⃣ Paste Phase-Event CSV")

st.markdown(
    """
**Expected format (NO time column):**
```csv
event_id,z,sigma
0,0.62,0.18
1,0.58,0.21
...