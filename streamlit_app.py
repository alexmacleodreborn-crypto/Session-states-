import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =========================
# SANDY'S LAW REGIMES
# =========================
K_LOW = 1.03
K_FREEZE = 1.51

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Zeno Cell – Stable Flow Simulator", layout="wide")

st.title("Zeno Cell / Everlasting Battery Simulator")
st.caption("Regime-controlled energy system (event-space dynamics)")

col1, col2 = st.columns(2)

with col1:
    LOAD = st.slider("Load (energy drain)", 0.0005, 0.01, 0.002, step=0.0005)
    Z_LOSS = st.slider("Loss Z", 0.002, 0.03, 0.01, step=0.002)
    SIGMA = st.slider("Entropy Σ (noise)", 0.0, 0.02, 0.006, step=0.002)

with col2:
    PHI_MAX = st.slider("Max injection Φₘₐₓ", 0.002, 0.05, 0.02, step=0.002)
    T_TOTAL = st.slider("Simulation time", 50, 400, 200, step=50)

# =========================
# CONTROLLER
# =========================
DT = 0.05
E_TARGET = 0.75
ENTROPY_GRAD = 0.02

def square_to_K(Z, Sigma):
    base = 1.1
    K = base + 12*Z + 8*Sigma
    return float(np.clip(K, K_LOW, K_FREEZE - 0.02))

def K_to_fm(K, Z):
    return (K * Z) / ENTROPY_GRAD

def simulate():
    n = int(T_TOTAL / DT)
    t = np.linspace(0, T_TOTAL, n)
    E = np.zeros(n)
    K_series = np.zeros(n)
    f_m = np.zeros(n)

    E[0] = 1.0
    rng = np.random.default_rng(7)

    for i in range(1, n):
        K = square_to_K(Z_LOSS, SIGMA)
        K_series[i] = K
        f = K_to_fm(K, Z_LOSS)
        f_m[i] = f

        # Regime
        if K < K_FREEZE:
            lam = Z_LOSS * 0.85
        else:
            lam = Z_LOSS * 0.12

        required = LOAD + lam * E[i-1]
        feedback = 0.02 * (E_TARGET - E[i-1])
        raw = required + feedback

        phi_cap = PHI_MAX * (K - K_LOW) / (K_FREEZE - K_LOW)
        phi_cap = max(phi_cap, 0.0)

        injection = min(max(raw, 0.0), phi_cap)

        noise = rng.normal(0.0, SIGMA) * np.sqrt(DT)

        dE = -lam * E[i-1] * DT + injection - LOAD + noise
        E[i] = np.clip(E[i-1] + dE, 0.0, 1.3)

    return t, E, K_series, f_m

t, E, K_series, f_m = simulate()

# =========================
# PLOTS
# =========================
fig, axs = plt.subplots(3, 1, figsize=(8, 9))

axs[0].plot(t, E)
axs[0].axhline(E_TARGET, linestyle="--")
axs[0].set_title("Energy under load")

axs[1].plot(t, K_series)
axs[1].axhline(K_LOW, linestyle="--")
axs[1].axhline(K_FREEZE, linestyle="--")
axs[1].set_title("K (regime control)")

axs[2].plot(t, f_m)
axs[2].set_title("Event density fₘ")

for ax in axs:
    ax.grid(True)

st.pyplot(fig)

# =========================
# REGIME STATUS
# =========================
if E[-1] < 0.1:
    st.error("❌ System expired (dissipative collapse)")
elif np.mean(K_series[-50:]) > K_FREEZE:
    st.warning("⚠️ Zeno lock-up (frozen)")
else:
    st.success("✅ Stable-flow regime maintained")