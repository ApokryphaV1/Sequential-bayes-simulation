import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
from io import StringIO, BytesIO

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(
    page_title="Sequential Bayesian Inference Simulation",
    layout="wide",
)

st.title("Sequential Bayesian Inference Simulation — Enhanced")
st.caption(
    "Compare a two-stage (sequential) Bayes estimator that fixes (α, γ) before estimating β,"
    " against a full joint Bayes estimator. Explore bias, RMSE, coverage, and posterior diagnostics."
)

# ===== Sidebar: User settings =====
st.sidebar.header("Data-Generating Process & Prior")
with st.sidebar:
    colA, colB = st.columns(2)
    with colA:
        rho = st.slider("Correlation ρ = Corr(x₁,x₂)", -1.0, 1.0, 0.2, 0.01)
        sigma = st.slider("Noise sd σ", 0.01, 1.0, 0.1, 0.01)
        n = st.slider("Sample size n", 50, 2000, 200, 10)
        n_reps = st.slider("Replications", 50, 5000, 400, 50)
    with colB:
        alpha_true = st.slider("True α", 0.0, 2.0, 0.7, 0.01)
        beta_true  = st.slider("True β", 0.0, 2.0, 0.3, 0.01)
        gamma_true = st.slider("True γ", 0.0, 2.0, 0.8, 0.01)
        n_post_samples = st.slider("Posterior samples (1st rep)", 500, 10000, 3000, 100)

    st.markdown("---")
    st.subheader("Prior on coefficients")
    prior_col1, prior_col2 = st.columns(2)
    with prior_col1:
        shared_tau = st.checkbox("Use one τ for all (α,β,γ)", value=True)
    with prior_col2:
        seed = st.number_input("Random seed", min_value=0, max_value=10**7, value=11, step=1)

    if shared_tau:
        tau_all = st.slider("Prior sd τ (shared)", 0.1, 20.0, 5.0, 0.1)
        tau_vec = np.array([tau_all, tau_all, tau_all], dtype=float)
    else:
        tau_alpha = st.slider("Prior sd τ_α", 0.1, 20.0, 5.0, 0.1)
        tau_beta  = st.slider("Prior sd τ_β", 0.1, 20.0, 5.0, 0.1)
        tau_gamma = st.slider("Prior sd τ_γ", 0.1, 20.0, 5.0, 0.1)
        tau_vec = np.array([tau_alpha, tau_beta, tau_gamma], dtype=float)

    with st.expander("Advanced prior means (default 0)"):
        mu_alpha = st.number_input("Prior mean μ_α", value=0.0, step=0.1)
        mu_beta  = st.number_input("Prior mean μ_β", value=0.0, step=0.1)
        mu_gamma = st.number_input("Prior mean μ_γ", value=0.0, step=0.1)
        mu_vec = np.array([mu_alpha, mu_beta, mu_gamma], dtype=float)

    st.markdown("---")
    show_progress = st.checkbox("Show progress bar during simulation", value=False)
    show_ols = st.checkbox("Include OLS baseline", value=False)

# ===== Helpers =====

def stats(arr: np.ndarray, true_val: float) -> dict:
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    bias = mean - float(true_val)
    rmse = float(np.sqrt(np.mean((arr - true_val) ** 2)))
    return {"mean": mean, "std": std, "bias": bias, "rmse": rmse}

@st.cache_data(show_spinner=False)
def run_simulation(
    n_reps: int,
    n: int,
    rho: float,
    sigma: float,
    tau_vec: np.ndarray,
    mu_vec: np.ndarray,
    alpha_true: float,
    beta_true: float,
    gamma_true: float,
    seed: int,
    n_post_samples: int,
    show_progress: bool,
    include_ols: bool,
):
    rng = np.random.default_rng(seed)

    alpha_seq = np.empty(n_reps)
    beta_seq = np.empty(n_reps)
    gamma_seq = np.empty(n_reps)

    alpha_joint = np.empty(n_reps)
    beta_joint = np.empty(n_reps)
    gamma_joint = np.empty(n_reps)

    sum_seq = np.empty(n_reps)
    sum_joint = np.empty(n_reps)

    # Optional OLS baseline
    if include_ols:
        alpha_ols = np.empty(n_reps)
        beta_ols  = np.empty(n_reps)
        gamma_ols = np.empty(n_reps)
        sum_ols   = np.empty(n_reps)
    else:
        alpha_ols = beta_ols = gamma_ols = sum_ols = None

    # For coverage (joint only)
    var_alpha_joint = np.empty(n_reps)
    var_beta_joint = np.empty(n_reps)
    var_gamma_joint = np.empty(n_reps)
    var_sum_joint = np.empty(n_reps)

    sigma2 = float(sigma ** 2)
    tau2_vec = tau_vec ** 2
    prior_prec = np.diag(1.0 / tau2_vec)

    posterior_cloud = None
    posterior_cloud_ab = None
    posterior_mean = None
    seqFix_point_example = None

    progress = st.progress(0) if show_progress else None

    for r in range(n_reps):
        # Simulate predictors with desired correlation
        x1 = rng.normal(size=n)
        x2 = rho * x1 + np.sqrt(max(0.0, 1.0 - rho ** 2)) * rng.normal(size=n)
        z  = rng.normal(size=n)
        eps = rng.normal(scale=sigma, size=n)
        y = alpha_true * x1 + beta_true * x2 + gamma_true * z + eps

        # Stage 1: Bayesian for (alpha, gamma) in y ~ x1 + z
        X1 = np.column_stack([x1, z])
        mu1 = np.array([mu_vec[0], mu_vec[2]])
        prior_prec1 = np.diag([1.0 / tau2_vec[0], 1.0 / tau2_vec[2]])
        A1 = (X1.T @ X1) / sigma2 + prior_prec1
        b1 = (X1.T @ y) / sigma2 + prior_prec1 @ mu1
        mean1 = np.linalg.solve(A1, b1)
        alpha_hat, gamma_hat = mean1

        # Stage 2: Fix (alpha, gamma), Bayesian for beta
        resid = y - alpha_hat * x1 - gamma_hat * z
        xtx_beta = (x2 @ x2) / sigma2 + 1.0 / tau2_vec[1]
        b_beta = (x2 @ resid) / sigma2 + mu_vec[1] / tau2_vec[1]
        beta_hat_seq = b_beta / xtx_beta

        alpha_seq[r] = float(alpha_hat)
        beta_seq[r] = float(beta_hat_seq)
        gamma_seq[r] = float(gamma_hat)
        sum_seq[r] = alpha_seq[r] + beta_seq[r]

        # Joint Bayesian on (alpha, beta, gamma)
        X = np.column_stack([x1, x2, z])
        A = (X.T @ X) / sigma2 + prior_prec
        b = (X.T @ y) / sigma2 + prior_prec @ mu_vec
        mean_post = np.linalg.solve(A, b)
        cov_post = np.linalg.inv(A)  # 3x3 — cheap and stable enough here

        alpha_joint[r], beta_joint[r], gamma_joint[r] = mean_post
        sum_joint[r] = mean_post[0] + mean_post[1]

        # Posterior variances and sum variance for coverage
        var_alpha_joint[r] = cov_post[0, 0]
        var_beta_joint[r]  = cov_post[1, 1]
        var_gamma_joint[r] = cov_post[2, 2]
        var_sum_joint[r]   = cov_post[0, 0] + cov_post[1, 1] + 2.0 * cov_post[0, 1]

        # OLS baseline
        if include_ols:
            XtX = X.T @ X
            try:
                beta_hat_ols = np.linalg.solve(XtX, X.T @ y)
            except np.linalg.LinAlgError:
                beta_hat_ols = np.linalg.pinv(XtX) @ (X.T @ y)
            alpha_ols[r], beta_ols[r], gamma_ols[r] = beta_hat_ols
            sum_ols[r] = alpha_ols[r] + beta_ols[r]

        # Save first-rep posterior cloud for visualization
        if r == 0:
            posterior_cloud = rng.multivariate_normal(mean_post, cov_post, size=n_post_samples)
            posterior_cloud_ab = posterior_cloud[:, :2]
            posterior_mean = mean_post.copy()
            seqFix_point_example = np.array([alpha_hat, float(beta_hat_seq)])

        if show_progress and progress is not None:
            progress.progress((r + 1) / n_reps)

    # 95% credible interval coverage (joint only)
    z = 1.96
    cover_alpha = ((alpha_true >= alpha_joint - z * np.sqrt(var_alpha_joint)) &
                   (alpha_true <= alpha_joint + z * np.sqrt(var_alpha_joint))).mean()
    cover_beta  = ((beta_true  >= beta_joint  - z * np.sqrt(var_beta_joint))  &
                   (beta_true  <= beta_joint  + z * np.sqrt(var_beta_joint))).mean()
    cover_gamma = ((gamma_true >= gamma_joint - z * np.sqrt(var_gamma_joint)) &
                   (gamma_true <= gamma_joint + z * np.sqrt(var_gamma_joint))).mean()
    sum_true = alpha_true + beta_true
    cover_sum   = ((sum_true   >= sum_joint   - z * np.sqrt(var_sum_joint))   &
                   (sum_true   <= sum_joint   + z * np.sqrt(var_sum_joint))).mean()

    return dict(
        alpha_seq=alpha_seq, beta_seq=beta_seq, gamma_seq=gamma_seq, sum_seq=sum_seq,
        alpha_joint=alpha_joint, beta_joint=beta_joint, gamma_joint=gamma_joint, sum_joint=sum_joint,
        alpha_ols=alpha_ols, beta_ols=beta_ols, gamma_ols=gamma_ols, sum_ols=sum_ols,
        posterior_cloud=posterior_cloud, posterior_cloud_ab=posterior_cloud_ab,
        posterior_mean=posterior_mean, seqFix_point_example=seqFix_point_example,
        coverage=dict(alpha=float(cover_alpha), beta=float(cover_beta), gamma=float(cover_gamma), sum=float(cover_sum))
    )

# ===== Run simulation =====
res = run_simulation(
    n_reps=n_reps,
    n=n,
    rho=rho,
    sigma=sigma,
    tau_vec=tau_vec,
    mu_vec=mu_vec,
    alpha_true=alpha_true,
    beta_true=beta_true,
    gamma_true=gamma_true,
    seed=seed,
    n_post_samples=n_post_samples,
    show_progress=show_progress,
    include_ols=show_ols,
)

# ===== Summary tables =====
rows = []
for (label, arr, tru, mname) in [
    ("SeqBayesFix", res["alpha_seq"], alpha_true, "alpha"),
    ("JointBayes",  res["alpha_joint"], alpha_true, "alpha"),
    ("SeqBayesFix", res["beta_seq"],  beta_true,  "beta"),
    ("JointBayes",  res["beta_joint"],  beta_true,  "beta"),
    ("SeqBayesFix", res["gamma_seq"], gamma_true, "gamma"),
    ("JointBayes",  res["gamma_joint"], gamma_true, "gamma"),
]:
    s = stats(arr, tru)
    rows.append([label, mname, s["mean"], s["std"], s["bias"], s["rmse"]])

# alpha+beta
sum_true = alpha_true + beta_true
s_seq_sum = stats(res["sum_seq"], sum_true)
s_joint_sum = stats(res["sum_joint"], sum_true)
rows.append(["SeqBayesFix", "alpha+beta", s_seq_sum["mean"], s_seq_sum["std"], s_seq_sum["bias"], s_seq_sum["rmse"]])
rows.append(["JointBayes",  "alpha+beta", s_joint_sum["mean"], s_joint_sum["std"], s_joint_sum["bias"], s_joint_sum["rmse"]])

# Optional OLS rows
if show_ols:
    for (arr, tru, mname) in [
        (res["alpha_ols"], alpha_true, "alpha"),
        (res["beta_ols"],  beta_true,  "beta"),
        (res["gamma_ols"], gamma_true, "gamma"),
    ]:
        s = stats(arr, tru)
        rows.append(["OLS", mname, s["mean"], s["std"], s["bias"], s["rmse"]])
    s_ols_sum = stats(res["sum_ols"], sum_true)
    rows.append(["OLS", "alpha+beta", s_ols_sum["mean"], s_ols_sum["std"], s_ols_sum["bias"], s_ols_sum["rmse"]])

summary_df = pd.DataFrame(rows, columns=["Method", "Quantity", "Mean", "Std", "Bias", "RMSE"]).round(4)

# MSE ratio (Seq / Joint) — handy quick diagnostic
mse_ratio_rows = []
for name, seq_arr, joint_arr in [
    ("alpha", res["alpha_seq"], res["alpha_joint"]),
    ("beta",  res["beta_seq"],  res["beta_joint"]),
    ("gamma", res["gamma_seq"], res["gamma_joint"]),
    ("alpha+beta", res["sum_seq"], res["sum_joint"]),
]:
    mse_seq = float(np.mean((seq_arr - (alpha_true if name=="alpha" else beta_true if name=="beta" else gamma_true if name=="gamma" else sum_true))**2))
    mse_joint = float(np.mean((joint_arr - (alpha_true if name=="alpha" else beta_true if name=="beta" else gamma_true if name=="gamma" else sum_true))**2))
    mse_ratio_rows.append([name, mse_seq / mse_joint if mse_joint > 0 else np.nan])

mse_ratio_df = pd.DataFrame(mse_ratio_rows, columns=["Quantity", "MSE Ratio (Seq/Joint)"]).round(3)

# ===== Layout with tabs =====
summary_tab, diagnostics_tab, viz_tab, ppc_tab, download_tab = st.tabs([
    "📊 Summary", "📐 Diagnostics", "📈 Corner / Scatter", "🔍 Posterior Predictive", "⬇️ Export"
])

with summary_tab:
    st.subheader("Performance summary across replications")
    st.dataframe(summary_df, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**Joint posterior 95% coverage rates (should be near 0.95):**")
        cov = res["coverage"]
        cov_df = pd.DataFrame({
            "Quantity": ["alpha", "beta", "gamma", "alpha+beta"],
            "Coverage95": [cov["alpha"], cov["beta"], cov["gamma"], cov["sum"]],
        }).round(3)
        st.dataframe(cov_df, use_container_width=True)
    with right:
        st.markdown("**MSE ratio (Sequential / Joint)** — values > 1 favor Joint")
        st.dataframe(mse_ratio_df, use_container_width=True)

with diagnostics_tab:
    st.subheader("Estimator distributions")
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.ravel()

    for ax, qty, seq_arr, joint_arr, true_val in [
        (axes[0], "alpha", res["alpha_seq"], res["alpha_joint"], alpha_true),
        (axes[1], "beta",  res["beta_seq"],  res["beta_joint"],  beta_true),
        (axes[2], "gamma", res["gamma_seq"], res["gamma_joint"], gamma_true),
        (axes[3], "alpha+beta", res["sum_seq"], res["sum_joint"], sum_true),
    ]:
        ax.hist(seq_arr, bins=40, alpha=0.6, label="SeqBayesFix")
        ax.hist(joint_arr, bins=40, alpha=0.6, label="JointBayes")
        ax.axvline(true_val, linestyle="--", linewidth=2, label="True" if qty=="alpha" else None)
        ax.set_title(qty)
        ax.grid(True, alpha=0.3)
        ax.legend()

    st.pyplot(fig, clear_figure=True)

with viz_tab:
    st.subheader("Posterior distribution (first replication)")
    st.caption("Blue: Joint Bayesian posterior draws; Red: distribution of sequential estimators across all replications")

    seq_bayes_estimators = np.column_stack([res["alpha_seq"], res["beta_seq"], res["gamma_seq"]])
    fig = corner.corner(
        res["posterior_cloud"],
        labels=["alpha", "beta", "gamma"],
        truths=[alpha_true, beta_true, gamma_true],
        color="blue",
    )
    corner.corner(seq_bayes_estimators, fig=fig, color="red")
    st.pyplot(fig, clear_figure=True)

    st.markdown("---")
    st.subheader("α vs β (clouds)")
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    ax2.scatter(res["posterior_cloud_ab"][:, 0], res["posterior_cloud_ab"][:, 1], s=6, alpha=0.35, label="Joint posterior draws")
    ax2.scatter(res["seqFix_point_example"][0], res["seqFix_point_example"][1], s=60, marker="x", label="Sequential (1st rep)")
    ax2.scatter(np.mean(res["alpha_seq"]), np.mean(res["beta_seq"]), s=80, marker="+", label="Sequential mean (all reps)")
    ax2.axvline(alpha_true, linestyle="--", alpha=0.7)
    ax2.axhline(beta_true, linestyle="--", alpha=0.7)
    ax2.set_xlabel("alpha")
    ax2.set_ylabel("beta")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    st.pyplot(fig2, clear_figure=True)

with ppc_tab:
    st.subheader("Posterior predictive check (first replication)")
    st.caption("Simulate ỹ from θ ~ p(θ|y) and compare to observed y distribution.")
    draws_for_ppc = st.slider("Number of posterior draws for PPC", 50, 1000, 200, 50)

    # Recreate the first replication's data with the same seed offset
    rng_ppc = np.random.default_rng(seed + 12345)
    x1 = rng_ppc.normal(size=n)
    x2 = rho * x1 + np.sqrt(max(0.0, 1.0 - rho ** 2)) * rng_ppc.normal(size=n)
    z  = rng_ppc.normal(size=n)
    eps = rng_ppc.normal(scale=sigma, size=n)
    y_obs = alpha_true * x1 + beta_true * x2 + gamma_true * z + eps

    # Posterior predictive: y_rep = X @ theta_draw + eps_draw
    X_ppc = np.column_stack([x1, x2, z])
    idx = np.random.choice(res["posterior_cloud"].shape[0], size=draws_for_ppc, replace=False)
    thetas = res["posterior_cloud"][idx]

    # To keep memory small, simulate summaries instead of the full matrix of y_rep
    yrep_means = []
    yrep_sds   = []
    for th in thetas:
        mu = X_ppc @ th
        yrep = mu + rng_ppc.normal(scale=sigma, size=n)
        yrep_means.append(float(np.mean(yrep)))
        yrep_sds.append(float(np.std(yrep, ddof=1)))

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.hist(yrep_means, bins=30, alpha=0.7, label="ỹ mean (rep draws)")
    ax3.axvline(np.mean(y_obs), color="k", linestyle="--", label="y mean (obs)")
    ax3.set_title("Posterior predictive of mean(y)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    st.pyplot(fig3, clear_figure=True)

    fig4, ax4 = plt.subplots(figsize=(6, 4))
    ax4.hist(yrep_sds, bins=30, alpha=0.7, label="ỹ sd (rep draws)")
    ax4.axvline(np.std(y_obs, ddof=1), color="k", linestyle="--", label="y sd (obs)")
    ax4.set_title("Posterior predictive of sd(y)")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    st.pyplot(fig4, clear_figure=True)

with download_tab:
    st.subheader("Export results")

    csv_buf = StringIO()
    summary_df.to_csv(csv_buf, index=False)
    st.download_button(
        label="Download summary CSV",
        data=csv_buf.getvalue(),
        file_name="bayes_sim_summary.csv",
        mime="text/csv",
    )

    # Pack arrays into an NPZ file
    npz_buf = BytesIO()
    np.savez_compressed(
        npz_buf,
        alpha_seq=res["alpha_seq"], beta_seq=res["beta_seq"], gamma_seq=res["gamma_seq"], sum_seq=res["sum_seq"],
        alpha_joint=res["alpha_joint"], beta_joint=res["beta_joint"], gamma_joint=res["gamma_joint"], sum_joint=res["sum_joint"],
        posterior_cloud=res["posterior_cloud"], posterior_mean=res["posterior_mean"],
    )
    st.download_button(
        label="Download arrays (NPZ)",
        data=npz_buf.getvalue(),
        file_name="bayes_sim_arrays.npz",
        mime="application/octet-stream",
    )

# ===== Explanatory notes =====
with st.expander("What’s happening under the hood? (math)"):
    st.markdown(r"""
    - **Sequential Bayes (fix)**: first compute \((\alpha,\gamma)\) from the submodel, then plug these into the second-stage Bayesian update for \(\beta\).
    - **Coverage** reported above is the empirical fraction of times the 95% *marginal* credible interval contains the true value (joint posterior only).
    """)
    st.latex(r"\text{Model: } y = \alpha x_1 + \beta x_2 + \gamma z + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)")
    st.latex(r"\text{Prior: }(\alpha, \beta, \gamma) \sim \mathcal{N}(\mu, \operatorname{diag}(\tau^2))")
    st.latex(r"\text{Joint posterior: }X^\top X/\sigma^2 + \operatorname{diag}(1/\tau^2)) \, \hat{\theta} = X^\top y/\sigma^2 + \operatorname{diag}(1/\tau^2)\,\mu")
with st.expander("Tips & caveats"):
    st.markdown("""
    - When \(|\rho| \to 1\), \(x_1\) and \(x_2\) are nearly collinear; OLS becomes unstable and the sequential estimator can be quite biased for \(\beta\).
    - Using `np.linalg.solve` avoids explicit matrix inversions for posterior means (better numerics). We only invert a 3×3 for the covariance used in plotting.
    - Toggle “Show progress bar” if you crank up replications; caching keeps results consistent for the same inputs.
    - Try different prior means \(\mu\) to see shrinkage toward nonzero targets.
    - Add your own priors by replacing the diagonal prior precision with a full matrix if you want correlated priors.
    """)
