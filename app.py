import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple

# 1. Set Page Configuration
st.set_page_config(
    page_title="A/B/n Test Power Calculator",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Core Calculation Functions ---
@st.cache_data
def calculate_power_frequentist(p_A: float, p_B: float, n_A: int, n_B: int,
                                alpha: float = 0.05, num_comparisons: int = 1) -> float:
    """Calculates power for two groups with potentially unequal sample sizes."""
    if p_B < 0 or p_B > 1.0 or n_A <= 0 or n_B <= 0:
        return 0.0
    adjusted_alpha = alpha / num_comparisons
    se = np.sqrt(p_A * (1 - p_A) / n_A + p_B * (1 - p_B) / n_B)
    if se == 0:
        return 1.0
    effect_size_norm = abs(p_B - p_A) / se
    z_alpha = norm.ppf(1 - adjusted_alpha / 2)
    return norm.cdf(effect_size_norm - z_alpha) + norm.cdf(-effect_size_norm - z_alpha)

@st.cache_data
def calculate_sample_size_frequentist(p_A: float, uplift: float,
                                      power_target: float = 0.8,
                                      alpha: float = 0.05,
                                      num_comparisons: int = 1,
                                      traffic_split: List[float] = None,
                                      max_sample_size: int = 5_000_000
                                      ) -> Tuple[int, int] | None:
    """Calculates required sample sizes per group, handling equal and unequal splits."""
    p_B = p_A * (1 + uplift)
    if p_B >= 1:
        return None
    
    if traffic_split is None: # Equal split
        n, power = 100, 0
        with st.spinner("Calculating sample size (equal split)..."):
            while power < power_target and n < max_sample_size:
                power = calculate_power_frequentist(p_A, p_B, n, n,
                                                    alpha, num_comparisons)
                if power >= power_target:
                    return n, n
                if n < 1000: n += 50
                elif n < 20000: n = int(n * 1.2)
                else: n = int(n * 1.1)
        return None
    else: # Unequal split
        pct_control, pct_variant = traffic_split
        total_n, power = 200, 0
        max_total_n = max_sample_size * (1 / min(traffic_split)) if min(traffic_split) > 0 else max_sample_size * 10
        with st.spinner("Calculating sample size (unequal split)..."):
            while power < power_target and total_n < max_total_n:
                n_A = int(total_n * pct_control)
                n_B = int(total_n * pct_variant)
                if n_A < 1 or n_B < 1:
                    total_n = int(total_n * 1.5) if total_n > 0 else 200
                    continue
                power = calculate_power_frequentist(p_A, p_B, n_A, n_B,
                                                    alpha, num_comparisons)
                if power >= power_target:
                    return n_A, n_B
                if total_n < 2000: total_n += 100
                elif total_n < 40000: total_n = int(total_n * 1.2)
                else: total_n = int(total_n * 1.1)
        return None

@st.cache_data
def calculate_mde_frequentist(p_A: float, n_A: int, n_B: int,
                              power_target: float = 0.8,
                              alpha: float = 0.05,
                              num_comparisons: int = 1
                              ) -> pd.DataFrame | None:
    """Calculates MDE vs. Power curve. Returns None if power target is not met."""
    results = []
    with st.spinner("Calculating Minimum Detectable Effect curve..."):
        for uplift in np.linspace(0.001, 0.50, 100):
            p_B = p_A * (1 + uplift)
            if p_B > 1.0: continue
            power = calculate_power_frequentist(p_A, p_B, n_A, n_B,
                                                alpha, num_comparisons)
            results.append({"uplift": uplift, "power": power})
    
    if not results: return None
    
    df = pd.DataFrame(results)
    if df['power'].max() < power_target:
        return df # Return the curve even if target not met
    return df

@st.cache_data
def generate_power_curve_data(p_A: float, uplift: float, n_A: int, n_B: int,
                              alpha: float, num_comparisons: int, traffic_split: List[float] = None) -> pd.DataFrame:
    """Generates data for plotting Power vs. Sample Size, handling unequal splits."""
    p_B = p_A * (1 + uplift)
    if p_B >= 1.0: return pd.DataFrame()

    # Generate a range of sample sizes for the variant group
    variant_sample_sizes = np.linspace(100, n_B * 2, 100, dtype=int)
    
    if traffic_split:
        # For unequal splits, calculate the corresponding control group size
        control_ratio = traffic_split[0] / traffic_split[1]
        control_sample_sizes = (variant_sample_sizes * control_ratio).astype(int)
        powers = [calculate_power_frequentist(p_A, p_B, n_a, n_b, alpha, num_comparisons) for n_a, n_b in zip(control_sample_sizes, variant_sample_sizes)]
    else:
        # For equal splits, control and variant sizes are the same
        powers = [calculate_power_frequentist(p_A, p_B, n, n, alpha, num_comparisons) for n in variant_sample_sizes]
    
    return pd.DataFrame({"variant_sample_size": variant_sample_sizes, "power": powers})


# --- Geo Testing Defaults & Helpers ---
GEO_DEFAULTS = pd.DataFrame({
    "Region": ["North East", "North West", "Yorkshire and the Humber",
               "East Midlands", "West Midlands", "East of England",
               "London", "South East", "South West",
               "Wales", "Scotland", "Northern Ireland"],
    "Weight": [0.03, 0.09, 0.07, 0.07, 0.09, 0.10, 0.18, 0.16, 0.07, 0.04, 0.07, 0.03],
    "CPM (£)": [7.50, 8.00, 8.25, 7.00, 7.80, 8.10, 12.00, 10.00, 7.60, 6.90, 9.00, 8.50]
})
ALL_REGIONS = GEO_DEFAULTS["Region"].tolist()

def reset_app_state():
    st.session_state.clear()
    st.rerun()

def prepare_geo_dataframe(
    base_df: pd.DataFrame,
    weighting_mode: str,
    active_regions: List[str]
) -> pd.DataFrame:
    """Filters and re-weights a geo dataframe based on user settings."""
    geo_df = base_df[base_df['Region'].isin(active_regions)].copy()
    if geo_df.empty: return geo_df
    
    if weighting_mode == 'Population-based':
        geo_df['Weight'] /= geo_df['Weight'].sum()
    elif weighting_mode == 'Equal':
        geo_df['Weight'] = 1 / len(geo_df)
    return geo_df

# --- UI ---
st.title("⚙️ A/B/n Pre-Test Power Calculator")
st.info("This tool helps you plan A/B tests by estimating the required users, duration, and cost to detect a target uplift with statistical confidence.")

st.sidebar.button("Reset All Settings", on_click=reset_app_state, use_container_width=True, help="Click to clear all inputs and reset the app to its default state.")
st.sidebar.markdown("---")

st.sidebar.header("1. Main Parameters")
use_unequal_split = st.sidebar.checkbox(
    "Use Unequal Traffic Split (A/B only)", key='unequal_split',
    help="Enable for a 1 vs 1 test with custom traffic allocation (e.g., 90/10). This is useful for minimizing risk by exposing fewer users to the variant."
)

if use_unequal_split:
    num_variants = 1
    st.sidebar.caption("Unequal split applies only to 1C vs 1V tests.")
else:
    num_variants = st.sidebar.number_input(
        "Number of Variants (excluding control)", 1, 10, 1, key='num_variants',
        help="The number of alternative versions to test against the control. 1 for a standard A/B test, 2 for A/B/C, etc."
    )

mode = st.sidebar.radio("Planning Goal",
    ["Find resources for a target uplift", "Find detectable uplift for fixed resources"],
    key='mode', help="Choose your objective: calculate the users/budget needed for a specific uplift, or find the smallest uplift you can detect with a fixed budget/sample size."
)
p_A = st.sidebar.number_input("Baseline Conversion Rate (p_A)", 0.0001, 0.999, 0.05, 0.001, format="%.4f", key='p_A', help="The current conversion rate of your control group. E.g., 0.05 for 5%.")

disable_run = False
if mode == "Find resources for a target uplift":
    uplift = st.sidebar.number_input("Target Relative Uplift", 0.0001, 2.0, 0.10, 0.01, format="%.4f", key='uplift', help="The percentage improvement you want to detect (e.g., 0.10 for a 10% lift over the baseline).")
    if p_A * (1 + uplift) >= 1.0:
        st.sidebar.error("Variant rate (Baseline * (1 + Uplift)) must be < 100%.")
        disable_run = True
else:
    mde_source = st.sidebar.radio("Fixed Resource Type:", ["Fixed Sample Size", "Fixed Budget"], key='mde_source', help="The constraint for your test.")
    if mde_source == "Fixed Sample Size":
        fixed_n_A = st.sidebar.number_input("Control Group Users", 100, value=10000, step=100, key='fixed_n_A')
        fixed_n_B = st.sidebar.number_input("Variant Group Users", 100, value=10000, step=100, key='fixed_n_B')
    else:
        total_budget = st.sidebar.number_input("Total Ad Spend (£)", 100, 50000, 100, key='total_budget')

st.sidebar.subheader("Test Settings")
alpha = st.sidebar.slider("Significance Level (α)", 0.01, 0.20, 0.05, key='alpha', help="The probability of a false positive (Type I error). A value of 0.05 means you accept a 5% chance of detecting an effect that doesn't exist.")
desired_power = st.sidebar.slider("Desired Power (1-β)", 0.5, 0.99, 0.8, key='desired_power_f', help="The probability of detecting a true effect (avoiding a false negative). A value of 0.80 means you have an 80% chance of finding a real uplift if it exists.")

if use_unequal_split:
    st.sidebar.header("Traffic Allocation")
    pct_control = st.sidebar.slider("Control Group % of Traffic", 1, 99, 90, key='pct_control')
    st.sidebar.metric(label="Variant Group %", value=f"{100-pct_control}%")

st.sidebar.header("Optional Calculations")
estimate_duration = st.sidebar.checkbox("Estimate Test Duration", value=True, key='estimate_duration')
if estimate_duration:
    weekly_traffic = st.sidebar.number_input("Total weekly traffic for test", 1, 20000, key='weekly_traffic', help="The total number of users available per week for this experiment.")

st.sidebar.header("Geo Spend Configuration")
force_geo = mode == "Find detectable uplift for fixed resources" and mde_source == "Fixed Budget"
calculate_geo_spend = st.sidebar.checkbox("Calculate Geo Spend & Cost", value=True, key='calculate_geo_spend', disabled=force_geo, help="Estimate the advertising spend based on regional CPMs.")

if calculate_geo_spend or force_geo:
    spend_mode = st.sidebar.radio("Weighting Mode", ["Population-based", "Equal", "Custom"], index=0, horizontal=True, key='spend_mode', help="How to distribute the budget across regions. Population-based is proportional to default weights, Equal splits it evenly, Custom lets you define weights.")
    with st.sidebar.expander("Configure Active Regions and Custom Data"):
        apply_to = st.radio("Apply region selection to:", ["Variant(s) only", "Both Control and Variant(s)"], horizontal=True, key='apply_to')
        with st.form("region_selection_form"):
            temp = [region for i, region in enumerate(ALL_REGIONS) if st.checkbox(region, value=region in st.session_state.get('selected_regions', ALL_REGIONS), key=f"chk_{region}")]
            if st.form_submit_button("Confirm Regions"):
                st.session_state.selected_regions = temp
                st.session_state.custom_geo_df = GEO_DEFAULTS[GEO_DEFAULTS['Region'].isin(temp)].copy()
                st.rerun()
        if spend_mode == 'Custom':
            if 'custom_geo_df' not in st.session_state:
                st.session_state.custom_geo_df = GEO_DEFAULTS[GEO_DEFAULTS['Region'].isin(st.session_state.get('selected_regions', ALL_REGIONS))].copy()
            edited = st.data_editor(st.session_state.custom_geo_df, use_container_width=True, key='custom_geo_editor')
            st.session_state.custom_geo_df = edited
            st.metric("Weight Sum", f"{edited['Weight'].sum():.1%}", delta=f"{edited['Weight'].sum()-1:.1%}")

with st.sidebar.expander("Advanced Settings"):
    max_sample_size = st.number_input("Max Sample Size Limit", 100_000, 10_000_000, 5_000_000, step=1_000_000, key='max_sample_size', help="The maximum sample size the calculator will search for to prevent very long run times.")

st.sidebar.markdown("---")
st.sidebar.markdown("Click below to run the calculation with the specified parameters.")
submit = st.sidebar.button("Run Calculation", type="primary", use_container_width=True, disabled=disable_run)
if 'submit' not in st.session_state: st.session_state.submit = False
if submit: st.session_state.submit = True

# --- RESULTS ---
if st.session_state.submit:
    st.header("📊 Executive Summary")
    p_A = st.session_state.p_A
    alpha = st.session_state.alpha
    desired_power = st.session_state.desired_power_f
    use_unequal = st.session_state.unequal_split
    num_vars = 1 if use_unequal else st.session_state.num_variants

    req_n_A = req_n_B = None
    if st.session_state.mode == "Find resources for a target uplift":
        uplift = st.session_state.uplift
        split = [st.session_state.pct_control / 100, (100 - st.session_state.pct_control) / 100] if use_unequal else None
        result = calculate_sample_size_frequentist(p_A, uplift, desired_power, alpha, num_vars, split, st.session_state.max_sample_size)
        if result: req_n_A, req_n_B = result
    else:
        if st.session_state.mde_source == "Fixed Sample Size":
            req_n_A, req_n_B = st.session_state.fixed_n_A, st.session_state.fixed_n_B
        else:
            sel_regions = st.session_state.get('selected_regions', ALL_REGIONS)
            apply_to = st.session_state.apply_to
            spend_mode = st.session_state.spend_mode
            geo_var_for_cpm = st.session_state.custom_geo_df.copy() if (spend_mode == 'Custom' and apply_to.startswith('Variant')) else GEO_DEFAULTS
            active_var_regions = sel_regions if apply_to.startswith('Variant') else ALL_REGIONS
            temp_geo_var = prepare_geo_dataframe(geo_var_for_cpm, spend_mode, active_var_regions)
            weighted_avg_cpm = (temp_geo_var['CPM (£)'] * temp_geo_var['Weight']).sum()
            total_users = int((st.session_state.total_budget / weighted_avg_cpm) * 1000) if weighted_avg_cpm > 0 else 0
            if use_unequal:
                req_n_A = int(total_users * st.session_state.pct_control / 100)
                req_n_B = total_users - req_n_A
            else:
                req_n_A = req_n_B = int(total_users / (1 + num_vars))

    if req_n_A is None or req_n_B is None:
        st.error("Could not determine sample sizes. The target uplift might be too small for the given constraints, or the budget may be too low.")
        st.stop()

    total_spend = None
    if st.session_state.calculate_geo_spend or force_geo:
        spend_mode = st.session_state.spend_mode
        apply_to = st.session_state.apply_to
        sel_regions = st.session_state.get('selected_regions', ALL_REGIONS)
        active_ctrl_regions = sel_regions if apply_to == 'Both Control and Variant(s)' else ALL_REGIONS
        geo_ctrl = prepare_geo_dataframe(GEO_DEFAULTS, spend_mode, active_ctrl_regions)
        ctrl_spend = ((req_n_A / 1000) * (geo_ctrl['Weight'] * geo_ctrl['CPM (£)'])).sum()
        active_var_regions = sel_regions if apply_to.startswith('Variant') else ALL_REGIONS
        base_df_var = st.session_state.custom_geo_df if (spend_mode == 'Custom' and apply_to.startswith('Variant')) else GEO_DEFAULTS
        geo_var = prepare_geo_dataframe(base_df_var, spend_mode, active_var_regions)
        var_spend = ((req_n_B / 1000) * (geo_var['Weight'] * geo_var['CPM (£)'])).sum()
        total_spend = ctrl_spend + (var_spend * num_vars)

    weeks = None
    if st.session_state.estimate_duration and st.session_state.weekly_traffic > 0:
        wt = st.session_state.weekly_traffic
        total_users_needed = req_n_A + (req_n_B * num_vars)
        weeks = total_users_needed / wt

    cols = st.columns(4)
    cols[0].metric("Users in Control", f"{req_n_A:,}")
    cols[1].metric("Users per Variant", f"{req_n_B:,}")
    cols[2].metric("Est. Duration (wks)", f"{weeks:.1f}" if weeks is not None else "N/A")
    cols[3].metric("Est. Total Spend", f"£{total_spend:,.0f}" if total_spend is not None else "N/A")

    st.markdown("---")
    st.header("🔬 Deeper Insights & Visuals")

    if st.session_state.mode == "Find resources for a target uplift":
        st.subheader("Power vs. Sample Size")
        split_ratio = [st.session_state.pct_control / 100, (100 - st.session_state.pct_control) / 100] if use_unequal else None
        power_curve_df = generate_power_curve_data(p_A, st.session_state.uplift, req_n_A, req_n_B, alpha, num_vars, split_ratio)
        if not power_curve_df.empty:
            fig, ax = plt.subplots()
            ax.plot(power_curve_df['variant_sample_size'], power_curve_df['power'], label='Power Curve')
            ax.axhline(y=desired_power, color='r', linestyle='--', label=f'Desired Power ({desired_power:.0%})')
            ax.axvline(x=req_n_B, color='g', linestyle='--', label=f'Required Users ({req_n_B:,})')
            ax.set_title('How Power Increases with More Users')
            ax.set_xlabel('Users per Variant Group')
            ax.set_ylabel('Statistical Power')
            ax.legend()
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            st.pyplot(fig)
            st.caption("This chart shows how the probability of detecting a true effect (power) increases as more users are added to the variant group. The calculation recommends the sample size where the blue power curve crosses your desired power level (red line).")

    if st.session_state.mode == "Find detectable uplift for fixed resources":
        st.subheader("Minimum Detectable Effect (MDE) vs. Power")
        mde_df = calculate_mde_frequentist(p_A, req_n_A, req_n_B, desired_power, alpha, num_vars)
        if mde_df is not None:
            mde_achieved = mde_df[mde_df['power'] >= desired_power].iloc[0] if not mde_df[mde_df['power'] >= desired_power].empty else None
            
            fig, ax = plt.subplots()
            ax.plot(mde_df['uplift'], mde_df['power'], label='Power Curve')
            ax.axhline(y=desired_power, color='r', linestyle='--', label=f'Desired Power ({desired_power:.0%})')
            if mde_achieved is not None:
                ax.axvline(x=mde_achieved['uplift'], color='g', linestyle='--', label=f'Detectable Uplift ({mde_achieved["uplift"]:.2%})')
                st.success(f"With the given resources, the minimum detectable uplift is **{mde_achieved['uplift']:.2%}** with **{mde_achieved['power']:.1%}** power.")
            else:
                st.warning("The desired power target was not reached for uplifts up to 50%. The chart below shows the power you can expect for smaller effects.")

            ax.set_title('Detectable Uplift at Different Power Levels')
            ax.set_xlabel('Minimum Detectable Uplift (MDE)')
            ax.set_ylabel('Statistical Power')
            ax.legend()
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            from matplotlib.ticker import PercentFormatter
            ax.xaxis.set_major_formatter(PercentFormatter(1.0))
            st.pyplot(fig)
            st.caption("This chart shows the trade-off between the size of the effect you want to detect and the power of your test. Smaller uplifts require more statistical power (and thus more users) to be detected reliably.")

    if st.session_state.calculate_geo_spend or force_geo:
        st.markdown("---")
        st.header("💰 Geo Spend & User Allocation")
        spend_mode = st.session_state.spend_mode
        apply_to = st.session_state.apply_to
        sel_regions = st.session_state.get('selected_regions', ALL_REGIONS)

        st.subheader("Control Group")
        active_ctrl_regions = sel_regions if apply_to == 'Both Control and Variant(s)' else ALL_REGIONS
        geo_ctrl = prepare_geo_dataframe(GEO_DEFAULTS, spend_mode, active_ctrl_regions)
        geo_ctrl['Users'] = (geo_ctrl['Weight'] * req_n_A).astype(int)
        geo_ctrl['Spend'] = (geo_ctrl['Weight'] * ctrl_spend).astype(float)
        st.dataframe(geo_ctrl[['Region', 'Weight', 'Users', 'Spend']].style.format({'Weight': '{:.1%}', 'Users': '{:,}', 'Spend': '£{:,.2f}'}), use_container_width=True)

        st.subheader("Variant Group")
        active_var_regions = sel_regions if apply_to.startswith('Variant') else ALL_REGIONS
        base_df_var = st.session_state.custom_geo_df if (spend_mode == 'Custom' and apply_to.startswith('Variant')) else GEO_DEFAULTS
        geo_var = prepare_geo_dataframe(base_df_var, spend_mode, active_var_regions)
        geo_var['Users'] = (geo_var['Weight'] * req_n_B).astype(int)
        geo_var['Spend'] = (geo_var['Weight'] * var_spend).astype(float)
        st.dataframe(geo_var[['Region', 'Weight', 'Users', 'Spend']].style.format({'Weight': '{:.1%}', 'Users': '{:,}', 'Spend': '£{:,.2f}'}), use_container_width=True)
        
        st.subheader("Spend Comparison by Region")
        combined_spend = pd.merge(geo_ctrl[['Region', 'Spend']], geo_var[['Region', 'Spend']], on='Region', suffixes=('_ctrl', '_var'), how='outer').fillna(0)
        fig, ax = plt.subplots()
        combined_spend.set_index('Region').plot(kind='barh', ax=ax, figsize=(10, 8))
        ax.set_title('Control vs. Variant Spend by Region')
        ax.set_xlabel('Spend (£)')
        ax.set_ylabel('Region')
        ax.legend(['Control', 'Variant'])
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        st.pyplot(fig)

else:
    st.info("👋 Welcome! Set your test parameters in the sidebar and click 'Run Calculation' to get started.")
