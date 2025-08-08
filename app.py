import streamlit as st
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

# 1. Set Page Configuration
st.set_page_config(
    page_title="Power Calculator | Bayesian Toolkit",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Core Simulation Functions ---
@st.cache_data
def run_simulation(n, p_A, p_B, simulations, samples, alpha_prior, beta_prior, thresh, num_variants):
    """
    Runs a single set of simulations for a given sample size and conversion rates across multiple variants.
    Returns the calculated power.
    """
    num_treatments = num_variants - 1
    if num_treatments < 1:
        st.error("Number of variants must be at least 2.")
        return 0.0

    rng = np.random.default_rng(seed=42)

    # Simulate conversions for control
    conversions_A = rng.binomial(n, p_A, size=simulations)
    
    # Simulate conversions for all treatment variants
    conversions_treatments = rng.binomial(n, p_B, size=(num_treatments, simulations))

    # --- Posterior Calculations ---
    # Control
    alpha_post_A = alpha_prior + conversions_A
    beta_post_A = beta_prior + n - conversions_A

    # Treatments
    alpha_post_treatments = alpha_prior + conversions_treatments
    beta_post_treatments = beta_prior + n - conversions_treatments

    # --- Sample from Posteriors ---
    # Control samples
    post_samples_A = beta.rvs(alpha_post_A, beta_post_A, size=(samples, simulations), random_state=rng)

    # Treatment samples - requires careful shaping for rvs function
    post_samples_treatments = beta.rvs(
        alpha_post_treatments,
        beta_post_treatments,
        size=(samples, num_treatments, simulations),
        random_state=rng
    )
    
    # --- Power Calculation ---
    # For each treatment, calculate probability it's better than control via broadcasting
    # Shape A: (samples, simulations) -> (samples, 1, simulations)
    # Shape Treatments: (samples, num_treatments, simulations)
    prob_treatment_better = np.mean(post_samples_treatments > post_samples_A[:, np.newaxis, :], axis=0)
    # Resulting shape: (num_treatments, simulations)

    # For each simulation, find the maximum probability that any treatment beat the control
    prob_best_treatment_is_better = np.max(prob_treatment_better, axis=0)
    # Resulting shape: (simulations,)

    # Power is the proportion of simulations where we found a "winner"
    power = np.mean(prob_best_treatment_is_better > thresh)
    return power


@st.cache_data
def simulate_power(p_A, uplift, thresh, desired_power, simulations, samples, alpha_prior, beta_prior, num_variants):
    """
    Simulates power across a range of sample sizes to find the minimum
    sample size required to achieve the desired power.
    """
    p_B = p_A * (1 + uplift)
    if p_B > 1.0:
        st.error(f"Error: Uplift of {uplift:.2%} on baseline {p_A:.2%} results in a conversion rate > 100%. Please lower the uplift or baseline.")
        return []

    results = []
    n = 100
    power = 0
    MAX_SAMPLE_SIZE = 5_000_000

    with st.spinner("Searching for required sample size... This may take a moment."):
        while power < desired_power and n < MAX_SAMPLE_SIZE:
            power = run_simulation(n, p_A, p_B, simulations, samples, alpha_prior, beta_prior, thresh, num_variants)
            results.append((n, power))
            if power >= desired_power:
                break
            # Increase sample size search step
            if n < 1000:
                n += 100
            elif n < 20000:
                n = int(n * 1.5)
            else:
                n = int(n * 1.25)
    return results

@st.cache_data
def simulate_mde(p_A, thresh, desired_power, simulations, samples, alpha_prior, beta_prior, fixed_n, num_variants):
    """
    Simulates power across a range of uplifts (MDEs) for a fixed sample size.
    """
    results = []
    uplifts = np.linspace(0.01, 0.50, 20)

    with st.spinner("Running simulations for MDE..."):
        for uplift in uplifts:
            p_B = p_A * (1 + uplift)
            if p_B > 1.0:
                continue
            power = run_simulation(fixed_n, p_A, p_B, simulations, samples, alpha_prior, beta_prior, thresh, num_variants)
            results.append((uplift, power))
            if power >= desired_power:
                break
    return results

# 2. Page Title and Introduction
st.title("⚙️ Pre-Test Power Calculator")
st.markdown(
    "This tool helps you plan an A/B/n test by estimating the sample size required or the minimum effect you can detect."
)

# 3. Sidebar for All User Inputs
with st.sidebar:
    st.header("Test Parameters")

    mode = st.radio(
        "Planning Mode",
        ["Estimate Sample Size", "Estimate MDE (Minimum Detectable Effect)"],
        help="Choose whether to estimate required sample size for a given uplift, or the minimum uplift detectable for a fixed sample size."
    )

    # --- NEW: Number of variants ---
    num_variants = st.number_input(
        "Number of variants (including control)",
        min_value=2,
        value=2,
        step=1,
        help="Total number of variants in the test. E.g., a control and two challengers would be 3."
    )
    
    p_A = st.number_input(
        "Baseline conversion rate (p_A)", min_value=0.0001, max_value=0.999, value=0.05, step=0.001,
        format="%.4f",
        help="Conversion rate for your control variant (A), e.g., 5% = 0.050"
    )
    thresh = st.slider(
        "Posterior threshold (e.g., 0.95)", 0.5, 0.99, 0.95, step=0.01,
        help="Confidence level to declare a winner — usually 0.95 or 0.99"
    )
    desired_power = st.slider(
        "Desired power", 0.5, 0.99, 0.8, step=0.01,
        help="Minimum acceptable power of detecting a real uplift"
    )
    
    if mode == "Estimate Sample Size":
        uplift = st.number_input(
            "Expected uplift (e.g., 0.10 = +10%)", min_value=0.0001, max_value=0.999, value=0.10, step=0.01,
            format="%.4f",
            help="Relative improvement expected in all treatment variants over the control."
        )
    else: # MDE Mode
        fixed_n = st.number_input(
            "Fixed sample size per variant", min_value=100, value=10000, step=100,
            help="Fixed sample size used to determine the minimum detectable uplift."
        )
    
    with st.expander("Advanced Settings"):
        simulations = st.slider(
            "Simulations", 100, 2000, 300, step=100,
            help="How many test simulations to run. More is more accurate but slower."
        )
        samples = st.slider(
            "Posterior samples", 500, 3000, 1000, step=100,
            help="How many samples to draw from each posterior distribution. More is more accurate but slower."
        )

    # --- UPDATED PRIOR BELIEFS SECTION ---
    st.subheader("Optional: Prior Beliefs")
    use_auto_prior = st.checkbox(
        "Calculate priors from historical data",
        help="Check this to calculate priors based on past conversions and sample size."
    )
    if use_auto_prior:
        hist_conv = st.number_input(
            "Historical Conversions (Successes)", min_value=0, value=50, step=1,
            help="The raw number of conversions or successes from your historical data."
        )
        hist_n = st.number_input(
            "Historical Total Sample Size (Users)", min_value=1, value=1000, step=1,
            help="The total number of users or observations in your historical data."
        )
        if hist_conv > hist_n:
            st.error("Historical conversions cannot exceed the total sample size.")
            st.stop()
        
        alpha_prior = hist_conv
        beta_prior = hist_n - hist_conv
    else:
        alpha_prior = st.number_input(
            "Alpha (prior successes)", min_value=0.0, value=1.0, step=0.1,
            help="Manually set your prior belief in successes. Default is 1 (uninformative)."
        )
        beta_prior = st.number_input(
            "Beta (prior failures)", min_value=0.0, value=1.0, step=0.1,
            help="Manually set your prior belief in failures. Default is 1 (uninformative)."
        )
    
    # --- MOVED AND UPDATED: Time-Based Planning ---
    st.markdown("---")
    st.header("⏱️ Optional: Time-Based Planning")
    estimate_duration = st.checkbox(
        "Estimate test duration",
        help="Check this to calculate how long the test might take based on weekly traffic."
    )
    if estimate_duration:
        weekly_traffic = st.number_input(
            "Estimated total weekly traffic",
            min_value=1,
            value=20000,
            step=100,
            help="Enter the total number of users you expect to enter the experiment each week (before splitting into variants)."
        )

    st.markdown("---")
    run_button = st.button("Run Calculation", type="primary", use_container_width=True)


# 4. Main Page for Displaying Outputs
st.markdown("---")

# Initialize state
results_available = False
required_n_per_variant = 0

if run_button:
    if mode == "Estimate Sample Size":
        results = simulate_power(p_A, uplift, thresh, desired_power, simulations, samples, alpha_prior, beta_prior, num_variants)
        if results:
            x_vals, y_vals = zip(*results)
            results_available = True
            st.subheader("📈 Sample Size Estimation")
            if y_vals[-1] >= desired_power:
                required_n_per_variant = x_vals[-1]
                total_n = required_n_per_variant * num_variants
                st.success(f"✅ Estimated minimum sample size **per variant**: **{required_n_per_variant:,}**.")
                st.info(f"ℹ️ Total required sample size across all {num_variants} variants: **{total_n:,}**.")
            else:
                st.warning("Could not reach desired power. The uplift may be too small or the power target too high for a practical test.")
    
    else: # Estimate MDE Mode
        results = simulate_mde(p_A, thresh, desired_power, simulations, samples, alpha_prior, beta_prior, fixed_n, num_variants)
        if results:
            x_vals, y_vals = zip(*results)
            results_available = True
            st.subheader("📉 Minimum Detectable Effect (MDE)")
            total_n = fixed_n * num_variants
            st.info(f"ℹ️ With a fixed sample size of **{fixed_n:,} per variant** (total **{total_n:,}**), the simulation will determine the MDE.")
            if y_vals[-1] >= desired_power:
                st.success(f"✅ Minimum detectable relative uplift: **{x_vals[-1]:.2%}** (achieved {y_vals[-1]:.1%} power).")
            else:
                st.warning("Simulation could not reach target power with the given sample size and uplift range.")

    if results_available:
        st.subheader("Visualizations")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x_vals, y_vals, marker='o', label='Estimated Power')
        ax.axhline(desired_power, color='red', linestyle='--', label=f'Target Power ({desired_power:.0%})')
        
        if mode == "Estimate Sample Size":
            ax.set_xlabel("Sample Size per Variant")
            if len(x_vals) > 1:
                ax.set_xscale('log')
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
        else: # MDE Mode
            ax.set_xlabel("Relative Uplift (MDE)")
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1%}"))
            
        ax.set_ylabel("Estimated Power")
        ax.set_title("Power vs. " + ("Sample Size" if mode == "Estimate Sample Size" else "MDE"))
        ax.grid(True, which="both", ls="--", c='0.7')
        ax.legend()
        st.pyplot(fig)

else:
    st.info("Adjust the parameters in the sidebar and click 'Run Calculation' to see the results.")


# --- Duration calculation (now conditional) ---
if results_available and 'estimate_duration' in locals() and estimate_duration:
    st.markdown("---")
    st.header("🗓️ Duration Estimate")
    users_per_week_per_variant = weekly_traffic / num_variants
    if users_per_week_per_variant > 0:
        if mode == "Estimate Sample Size":
            if required_n_per_variant > 0:
                estimated_weeks = required_n_per_variant / users_per_week_per_variant
                st.info(f"To reach **{required_n_per_variant:,} users per variant**, you'll need to run this test for approximately **{estimated_weeks:.1f} weeks**.")
            else: # Power not reached
                pass # The warning is already shown above
        else: # MDE mode
            estimated_weeks = fixed_n / users_per_week_per_variant
            st.info(f"To reach **{fixed_n:,} users per variant**, it will take approximately **{estimated_weeks:.1f} weeks**.")

# 5. Explanations Section
st.markdown("---")
with st.expander("ℹ️ Learn about the concepts used in this calculator"):
    st.markdown("""
    #### What is Sample Size? 👥
    **Sample size** is the number of users in each **variant** of your test. Think of it like the lens on a camera you're using to see which variant is better. A bigger sample size gives you a bigger, more powerful lens.

    This bigger lens makes your test more sensitive in two key ways:

    1.  **You can spot smaller improvements (Lower MDE 🔎)**
        A powerful camera lens (**more users**) can spot a tiny, faint star that a weaker lens would miss. Similarly, a larger sample size allows you to reliably detect a very **small uplift (a lower MDE)**.

    2.  **You're more certain about what you see (Higher Power 💪)**
        When you're trying to photograph a specific star, a bigger lens (**more users**) gives you a much better chance (**higher power**) of capturing a sharp, undeniable photo instead of a blurry, inconclusive smudge.

    The goal is to find the right balance—a lens big enough to be confident in the result, but not so big that you waste time and resources.

    ---
    #### What is Minimum Detectable Effect (MDE)? 🔎
    The **Minimum Detectable Effect (MDE)** is the smallest improvement your test can reliably detect at a given power level.

    Think of it as the sensitivity of your experiment. If the true uplift from your change is smaller than the MDE, your test will likely miss it. This doesn't mean the uplift isn't real, just that your experiment isn't powerful enough to see it. Use the MDE to set realistic expectations for what your test can achieve with your available traffic.

    ---
    #### What is Bayesian Power? 💪
    **Power** answers one critical question: *"If one of your new variants is truly better by a specific amount, what's the probability my test will actually detect it?"*

    For example, 80% power means you have an 80% chance of getting a conclusive result (i.e., identifying a winning variant with high confidence) if the real improvement matches what you expected. Running a test with low power is like trying to read in a dim room—you're likely to miss things and end up with an inconclusive result, wasting valuable traffic.

    ---
    #### What are Priors? 🧠
    **Priors** represent what you believe about the conversion rate *before* the test begins. In this model, your belief is captured by two numbers:
    - **Alpha ($$\\alpha$$)**: The number of prior "successes".
    - **Beta ($$\\beta$$)**: The number of prior "failures".

    * **No strong belief?** Use an **uninformative prior** like `alpha = 1` and `beta = 1`. This treats all possible conversion rates as equally likely to start.
    * **Have historical data?** Create an **informative prior**. If past data showed 50 conversions from 1,000 users, you'd set `alpha = 50` and `beta = 950`.

    As your test collects new data, the evidence from the experiment will quickly outweigh the initial prior belief.
    """)
