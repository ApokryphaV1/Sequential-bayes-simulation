import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

st.title("Sequential vs. Joint Bayesian Estimation")

# Sidebar inputs
st.sidebar.header("Simulation Parameters")
rho = st.sidebar.slider("Correlation ρ between x1 and x2", 0.0, 1.0, 0.2, 0.05)
alpha_true = st.sidebar.number_input("True α", value=0.7)
beta_true  = st.sidebar.number_input("True β", value=0.3)
gamma_true = st.sidebar.number_input("True γ", value=0.8)
sigma = st.sidebar.number_input("Noise σ", value=0.1)
tau = st.sidebar.number_input("Prior std τ", value=5.0)
n = st.sidebar.number_input("Sample size n", value=200)
n_reps = st.sidebar.number_input("Number of replications", value=400)
n_post_samples = st.sidebar.number_input("Posterior samples for plot", value=3000)

if st.sidebar.button("Run Simulation"):
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
        return {"mean": float(np.mean(x)), "std": float(np.std(x, ddof=1)), 
                "bias": float(np.mean(x) - true), 
                "rmse": float(np.sqrt(np.mean((x - true)**2)))}

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
    st.subheader("Summary Statistics")
    st.dataframe(df)

    # Corner plot
    st.subheader("Joint Posterior (α, β) Corner Plot")
    fig_corner = az.plot_pair({"posterior": {"alpha": posterior_cloud_ab[:,0], "beta": posterior_cloud_ab[:,1]}},
                              marginals=True)
    st.pyplot(fig_corner)

    # Histograms
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
