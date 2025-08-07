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
                # BUG FIX: Pass num_comparisons to the power function
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
                              ) -> List[Tuple[float, float]] | None:
    """Calculates MDE for two groups. Returns None if power target is not met."""
    results = []
    with st.spinner("Calculating Minimum Detectable Effect..."):
        for uplift in np.linspace(0.001, 0.50, 100):
            p_B = p_A * (1 + uplift)
            if p_B > 1.0: continue
            power = calculate_power_frequentist(p_A, p_B, n_A, n_B,
                                                alpha, num_comparisons)
            results.append((uplift, power))
            if power >= power_target:
                return results
    # BUG FIX: Return None instead of an empty list if power target is not reached
    return None

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

# REFACTOR: Helper function for preparing geo dataframes
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
    # 'Custom' mode uses pre-edited weights, so no changes needed here.
    return geo_df

# --- UI ---
st.title("⚙️ A/B/n Pre-Test Power Calculator")

# Sidebar controls
st.sidebar.button("Reset All Settings", on_click=reset_app_state, use_container_width=True)
st.sidebar.markdown("---")

st.sidebar.header("1. Main Parameters")
use_unequal_split = st.sidebar.checkbox(
    "Use Unequal Traffic Split (A/B only)", key='unequal_split',
    help="Enable for a 1 vs 1 test with custom traffic allocation (e.g., 90/10)."
)

if use_unequal_split:
    num_variants = 1
    st.sidebar.caption("Unequal split applies only to 1C vs 1V tests.")
else:
    num_variants = st.sidebar.number_input(
        "Number of Variants (excluding control)", 1, 10, 1, key='num_variants',
        help="1 for A/B, 2 for A/B/C, etc."
    )

mode = st.sidebar.radio("Planning Goal",
    ["Find resources for a target uplift", "Find detectable uplift for fixed resources"],
    key='mode'
)
p_A = st.sidebar.number_input("Baseline rate (p_A)", 0.0001, 0.999, 0.05, 0.001, format="%.4f", key='p_A')

disable_run = False
if mode == "Find resources for a target uplift":
    uplift = st.sidebar.number_input("Target Uplift", 0.0001, 0.999, 0.10, 0.01, format="%.4f", key='uplift')
    if p_A * (1 + uplift) >= 1.0:
        st.sidebar.error("Variant rate must be < 100%.")
        disable_run = True
else:
    mde_source = st.sidebar.radio("Fixed Resource Type:", ["Fixed Sample Size", "Fixed Budget"], key='mde_source')
    if mde_source == "Fixed Sample Size":
        fixed_n_A = st.sidebar.number_input("Control Sample Size", 100, value=10000, step=100, key='fixed_n_A')
        fixed_n_B = st.sidebar.number_input("Variant Sample Size", 100, value=10000, step=100, key='fixed_n_B')
    else:
        total_budget = st.sidebar.number_input("Total Ad Spend (£)", 100, 50000, 100, key='total_budget')

st.sidebar.subheader("Test Settings")
alpha = st.sidebar.slider("Significance α (Family-wise)", 0.01, 0.10, 0.05, key='alpha')
desired_power = st.sidebar.slider("Desired Power (1-β)", 0.5, 0.99, 0.8, key='desired_power_f')

if use_unequal_split:
    st.sidebar.header("Traffic Allocation")
    pct_control = st.sidebar.slider("Control Group % of Traffic", 1, 99, 90, key='pct_control')
    pct_variant = 100 - pct_control
    st.sidebar.metric(label="Variant Group %", value=f"{pct_variant}%")

st.sidebar.header("Optional Calculations")
estimate_duration = st.sidebar.checkbox("Estimate Test Duration", value=True, key='estimate_duration')
if estimate_duration:
    weekly_traffic = st.sidebar.number_input("Total weekly traffic for test", 1, 20000, key='weekly_traffic')

st.sidebar.header("Geo Spend Configuration")
force_geo = mode == "Find detectable uplift for fixed resources" and mde_source == "Fixed Budget"
calculate_geo_spend = st.sidebar.checkbox("Calculate Geo Spend", value=True, key='calculate_geo_spend', disabled=force_geo)

if calculate_geo_spend or force_geo:
    spend_mode = st.sidebar.radio("Weighting Mode", ["Population-based", "Equal", "Custom"], index=0, horizontal=True, key='spend_mode')
    with st.sidebar.expander("Configure Active Regions and Custom Data"):
        apply_to = st.radio("Apply region selection to:",
            ["Variant(s) only", "Both Control and Variant(s)"],
            horizontal=True, key='apply_to')
        
        with st.form("region_selection_form"):
            temp = []
            cols = st.columns(3)
            for i, region in enumerate(ALL_REGIONS):
                with cols[i % 3]:
                    selected = st.checkbox(region, value=region in st.session_state.get('selected_regions', ALL_REGIONS), key=f"chk_{region}")
                    if selected: temp.append(region)
            
            if st.form_submit_button("Confirm Regions"):
                st.session_state.selected_regions = temp
                st.session_state.custom_geo_df = GEO_DEFAULTS[GEO_DEFAULTS['Region'].isin(temp)].copy()
                # MODERNIZATION: Use st.rerun() instead of st.experimental_rerun()
                st.rerun()

        if spend_mode == 'Custom':
            if 'custom_geo_df' not in st.session_state:
                st.session_state.custom_geo_df = GEO_DEFAULTS[GEO_DEFAULTS['Region'].isin(st.session_state.get('selected_regions', ALL_REGIONS))].copy()
            
            edited = st.data_editor(st.session_state.custom_geo_df, use_container_width=True, key='custom_geo_editor')
            st.session_state.custom_geo_df = edited
            total_w = edited['Weight'].sum()
            st.metric("Weight Sum", f"{total_w:.1%}", delta=f"{total_w-1:.1%}")

with st.sidebar.expander("Advanced Settings"):
    max_sample_size = st.number_input("Max Sample Size Limit", 100_000, 5_000_000, 1_000_000, key='max_sample_size')

st.sidebar.markdown("---")
submit = st.sidebar.button("Run Calculation", type="primary", disabled=disable_run)
if 'submit' not in st.session_state: st.session_state.submit = False
if submit: st.session_state.submit = True

# --- RESULTS ---
if st.session_state.submit:
    st.header("Results")
    # Read inputs from session state for consistency
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
    else: # Find detectable uplift
        if st.session_state.mde_source == "Fixed Sample Size":
            req_n_A, req_n_B = st.session_state.fixed_n_A, st.session_state.fixed_n_B
        else: # Fixed Budget
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

    weeks = None
    if st.session_state.estimate_duration and st.session_state.weekly_traffic > 0:
        wt = st.session_state.weekly_traffic
        if use_unequal:
            wA = req_n_A / (wt * (st.session_state.pct_control / 100))
            wB = req_n_B / (wt * ((100 - st.session_state.pct_control) / 100))
            weeks = max(wA, wB)
        else:
            total = req_n_A + req_n_B * num_vars
            weeks = total / wt

    st.markdown('<div class="bordered-container">', unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].metric("Control n", f"{req_n_A:,}")
    cols[1].metric("Variant n", f"{req_n_B:,}")
    cols[2].metric("Duration (wks)", f"{weeks:.1f}" if weeks is not None else "N/A")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.mode == "Find detectable uplift for fixed resources":
        st.subheader("📉 Minimum Detectable Effect")
        mde_res = calculate_mde_frequentist(p_A, req_n_A, req_n_B, desired_power, alpha, num_vars)
        # BUG FIX: Check if mde_res is not None
        if mde_res:
            mde, ach_power = mde_res[-1]
            st.success(f"Detectable uplift: **{mde:.2%}** (attained power {ach_power:.1%})")
        else:
            st.warning("Power target could not be reached for uplifts up to 50%. Consider increasing sample size/budget or lowering the desired power.")

    if st.session_state.calculate_geo_spend or force_geo:
        st.header("💰 Geo Spend & User Allocation")
        spend_mode = st.session_state.spend_mode
        apply_to = st.session_state.apply_to
        sel_regions = st.session_state.get('selected_regions', ALL_REGIONS)

        # Control Spend
        st.subheader("Control Group")
        active_ctrl_regions = sel_regions if apply_to == 'Both Control and Variant(s)' else ALL_REGIONS
        geo_ctrl = prepare_geo_dataframe(GEO_DEFAULTS, spend_mode, active_ctrl_regions)
        geo_ctrl['Users'] = (geo_ctrl['Weight'] * req_n_A).astype(int)
        geo_ctrl['Imps_k'] = geo_ctrl['Users'] / 1000
        geo_ctrl['Spend'] = geo_ctrl['Imps_k'] * geo_ctrl['CPM (£)']
        st.dataframe(geo_ctrl.style.format({'Weight': '{:.1%}', 'Users': '{:,}', 'CPM (£)': '£{:.2f}', 'Spend': '£{:,.2f}'}), use_container_width=True)

        # Variant Spend
        st.subheader("Variant Group")
        # BUG FIX: Define geo_var correctly regardless of the mode
        active_var_regions = sel_regions if apply_to.startswith('Variant') else ALL_REGIONS
        base_df_var = st.session_state.custom_geo_df if (spend_mode == 'Custom' and apply_to.startswith('Variant')) else GEO_DEFAULTS
        geo_var = prepare_geo_dataframe(base_df_var, spend_mode, active_var_regions)
        geo_var['Users'] = (geo_var['Weight'] * req_n_B).astype(int)
        geo_var['Imps_k'] = geo_var['Users'] / 1000
        geo_var['Spend'] = geo_var['Imps_k'] * geo_var['CPM (£)']
        st.dataframe(geo_var.style.format({'Weight': '{:.1%}', 'Users': '{:,}', 'CPM (£)': '£{:.2f}', 'Spend': '£{:,.2f}'}), use_container_width=True)
        
        # Combined Chart
        st.subheader("Spend Comparison by Region")
        combined_spend = pd.merge(
            geo_ctrl[['Region', 'Spend']],
            geo_var[['Region', 'Spend']],
            on='Region',
            suffixes=('_ctrl', '_var'),
            how='outer'
        ).fillna(0)
        
        fig, ax = plt.subplots()
        combined_spend.set_index('Region').plot(kind='barh', ax=ax, figsize=(10, 8))
        ax.set_title('Control vs. Variant Spend by Region')
        ax.set_xlabel('Spend (£)')
        ax.set_ylabel('Region')
        ax.legend(['Control', 'Variant'])
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        st.pyplot(fig)

else:
    st.info("Set your parameters in the sidebar and click 'Run Calculation'.")
