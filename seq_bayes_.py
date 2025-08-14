import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.title('Sequential Bayesian Inference Simulation')

# ===== User settings =====
st.sidebar.header('User Settings')
rho = st.sidebar.slider('Correlation (rho)', -1.0, 1.0, 0.2, 0.01)
alpha_true = st.sidebar.slider('True Alpha', 0.0, 2.0, 0.7, 0.01)
beta_true = st.sidebar.slider('True Beta', 0.0, 2.0, 0.3, 0.01)
gamma_true = st.sidebar.slider('True Gamma', 0.0, 2.0, 0.8, 0.01)
sigma = st.sidebar.slider('Sigma', 0.01, 1.0, 0.1, 0.01)
tau = st.sidebar.slider('Prior Std (tau)', 1.0, 10.0, 5.0, 0.1)
n = st.sidebar.slider('Sample Size (n)', 100, 1000, 200, 10)
n_reps = st.sidebar.slider('Number of Replications', 100, 1000, 400, 10)
n_post_samples = st.sidebar.slider('Posterior Samples', 1000, 5000, 3000, 100)
# =========================

np.random.seed(11)

sigma2 = sigma**2
tau2 = tau**2
sum_true = alpha_true + beta_true

alpha_seqFix = np.empty(n_reps); beta_seqFix = np.empty(n_reps); gamma_seqFix = np.empty(n_reps)
alpha_bayes  = np.empty(n_reps); beta_bayes  = np.empty(n_reps); gamma_bayes  = np.empty(n_reps)
sum_seqFix = np.empty(n_reps); sum_bayes = np.empty(n_reps)

posterior_cloud_ab = None
posterior_mean = None
seqFix_point_example = None
true_point_ab = np.array([alpha_true, beta_true])

for r in range(n_reps):
    x1 = np.random.normal(size=n)
    x2 = rho * x1 + np.sqrt(max(0.0, 1 - rho**2)) * np.random.normal(size=n)
    z  = np.random.normal(size=n)
    eps = np.random.normal(scale=sigma, size=n)
    y = alpha_true*x1 + beta_true*x2 + gamma_true*z + eps

    # Stage 1: Bayesian for (alpha,gamma) in y~x1+z
    X1 = np.column_stack([x1, z])
    A1 = (X1.T @ X1) / sigma2 + (1.0/tau2) * np.eye(2)
    cov1 = np.linalg.inv(A1)
    mean1 = cov1 @ (X1.T @ y / sigma2)
    alpha_hat_bayes, gamma_hat_bayes = mean1

    # Stage 2: Fix (alpha,gamma), Bayesian for beta
    resid = y - alpha_hat_bayes * x1 - gamma_hat_bayes * z
    XtX_beta = (x2 @ x2) / sigma2 + (1.0/tau2)
    cov2 = 1.0 / XtX_beta
    mean2 = cov2 * (x2 @ resid) / sigma2
    beta_hat_bayes_seqFix = float(mean2)

    alpha_seqFix[r] = float(alpha_hat_bayes)
    beta_seqFix[r]  = beta_hat_bayes_seqFix
    gamma_seqFix[r] = float(gamma_hat_bayes)
    sum_seqFix[r]   = alpha_seqFix[r] + beta_seqFix[r]

    # Joint Bayesian
    X = np.column_stack([x1, x2, z])
    A = (X.T @ X) / sigma2 + (1.0/tau2) * np.eye(3)
    cov_post = np.linalg.inv(A)
    mean_post = cov_post @ (X.T @ y / sigma2)

    alpha_bayes[r], beta_bayes[r], gamma_bayes[r] = mean_post
    sum_bayes[r] = mean_post[0] + mean_post[1]

    if r == 0:
        posterior_cloud = np.random.multivariate_normal(mean_post, cov_post, size=n_post_samples)
        posterior_cloud_ab = posterior_cloud[:, :2]
        posterior_mean = mean_post.copy()
        seqFix_point_example = np.array([alpha_hat_bayes, beta_hat_bayes_seqFix])

def stats(x, true):
    import numpy as np
    return {"mean": float(np.mean(x)), "std": float(np.std(x, ddof=1)), "bias": float(np.mean(x) - true), "rmse": float(np.sqrt(np.mean((x - true)**2))) }

rows = []
for lbl, est, tru in [
    ("SeqBayesFix", alpha_seqFix, alpha_true),
    ("JointBayes",  alpha_bayes,  alpha_true),
    ("SeqBayesFix", beta_seqFix,  beta_true),
    ("JointBayes",  beta_bayes,   beta_true),
    ("SeqBayesFix", gamma_seqFix, gamma_true),
    ("JointBayes",  gamma_bayes,  gamma_true),]:
    m = "alpha" if tru==alpha_true and (est is alpha_seqFix or est is alpha_bayes) else "beta" if tru==beta_true and (est is beta_seqFix or est is beta_bayes) else "gamma"
    s = stats(est, tru)
    rows.append([lbl, m, s["mean"], s["std"], s["bias"], s["rmse"]])

s_seq_sum = stats(sum_seqFix, sum_true)
s_bay_sum = stats(sum_bayes,  sum_true)
rows.append(["SeqBayesFix", "alpha+beta", s_seq_sum["mean"], s_seq_sum["std"], s_seq_sum["bias"], s_seq_sum["rmse"]])
rows.append(["JointBayes",  "alpha+beta", s_bay_sum["mean"],  s_bay_sum["std"],  s_bay_sum["bias"],  s_bay_sum["rmse"]])

df = pd.DataFrame(rows, columns=["Method","Quantity","Mean","Std","Bias","RMSE"])
st.write(df)

# Scatter Plot
st.header('Posterior Distribution Scatter Plot')
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(posterior_cloud_ab[:,0], posterior_cloud_ab[:,1], s=6, label="joint posterior samples (α,β)")
ax.scatter([posterior_mean[0]], [posterior_mean[1]], marker='D', s=80, label="joint posterior mean")
ax.scatter([seqFix_point_example[0]], [seqFix_point_example[1]], marker='s', s=80, label="SeqBayesFix point (α̂,β̂)")
ax.scatter([alpha_true], [beta_true], marker='*', s=160, label="true (α,β)")
ax.set_xlabel("alpha")
ax.set_ylabel("beta")
ax.set_title(f"Correlation ρ={rho} between x1 and x2")
ax.legend(loc="upper right")
ax.grid(True)
st.pyplot(fig)

# Histograms
st.header('Parameter Estimate Histograms')
for name, seq, bay, tru in [
    ("Alpha", alpha_seqFix, alpha_bayes, alpha_true),
    ("Beta", beta_seqFix, beta_bayes, beta_true),
    ("Gamma", gamma_seqFix, gamma_bayes, gamma_true),
    ("Alpha+Beta", sum_seqFix, sum_bayes, sum_true)
]:
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(seq, bins=30, alpha=0.6, label=f"SeqBayesFix {name}")
    ax.hist(bay, bins=30, alpha=0.6, label=f"JointBayes {name}")
    ax.axvline(tru, linewidth=2, label=f"true {name}")
    ax.set_xlabel(f"{name} estimate")
    ax.set_ylabel("count")
    ax.set_title(f"{name} estimates")
    ax.legend()
    st.pyplot(fig)
