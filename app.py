import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from io import StringIO
from scipy import stats
import base64
import io

st.set_page_config(page_title="Linear Regression Explorer", layout="wide")
st.title("📈 Linear Regression Explorer")

st.markdown("""
Upload your dataset or paste values to fit a linear regression model and visualize the results.
""")

# --- Data Upload or Entry ---
data_source = st.radio("How would you like to input data?", ["Upload CSV", "Paste CSV Text"], horizontal=True)

if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
else:
    csv_text = st.text_area("Paste your CSV data below:", height=200)
    if csv_text:
        df = pd.read_csv(StringIO(csv_text))

if 'df' in locals():
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # --- Correlation Matrix ---
    st.subheader("📊 Correlation Matrix")
    numeric_df = df.select_dtypes(include=[np.number])
    fig_corr, ax_corr = plt.subplots()
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

    # --- Variable Selection ---
    st.subheader("⚙️ Model Setup")
    columns = df.columns.tolist()
    y_col = st.selectbox("Select the dependent variable (Y)", columns)
    x_cols = st.multiselect("Select one or more independent variables (X)", [col for col in columns if col != y_col])

    if x_cols:
        formula = f"{y_col} ~ {' + '.join(x_cols)}"
        try:
            model = smf.ols(formula=formula, data=df).fit()

            st.subheader("📊 Regression Results")
            st.text(model.summary())

            # --- Confidence Intervals ---
            st.subheader("📏 Confidence Intervals for Predictions")
            pred_df = df[x_cols].copy()
            if 'const' not in pred_df.columns:
                pred_df = sm.add_constant(pred_df)
            predictions = model.get_prediction(pred_df)
            pred_summary = predictions.summary_frame(alpha=0.05)
            st.dataframe(pred_summary[["mean", "mean_ci_lower", "mean_ci_upper"]].head())

            # --- Assumption Diagnostics ---
            st.subheader("🧪 Assumptions Diagnostics")

            # Q-Q plot for normality
            fig_qq = sm.qqplot(model.resid, line='45', fit=True)
            st.pyplot(fig_qq)

            # Residual vs Fitted
            fig_resid, ax_resid = plt.subplots()
            sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, ax=ax_resid)
            ax_resid.set_xlabel("Fitted values")
            ax_resid.set_ylabel("Residuals")
            ax_resid.set_title("Residuals vs Fitted")
            st.pyplot(fig_resid)

            # --- Polynomial Regression Option ---
            st.subheader("🔁 Optional: Polynomial Regression")
            degree = st.slider("Select polynomial degree (1 = linear)", min_value=1, max_value=3, value=1)

            if degree > 1 and len(x_cols) == 1:
                x = df[x_cols[0]]
                for d in range(2, degree + 1):
                    df[f"{x_cols[0]}**{d}"] = x ** d
                poly_terms = [x_cols[0]] + [f"{x_cols[0]}**{d}" for d in range(2, degree + 1)]
                poly_formula = f"{y_col} ~ {' + '.join(poly_terms)}"
                poly_model = smf.ols(formula=poly_formula, data=df).fit()
                st.text(poly_model.summary())

            # --- Predict New Values ---
            st.subheader("🔮 Predict New Values")
            new_vals = {}
            for col in x_cols:
                new_vals[col] = st.number_input(f"Enter value for {col}", value=float(df[col].mean()))
            pred_input = pd.DataFrame([new_vals])
            if 'const' not in pred_input.columns:
                pred_input = sm.add_constant(pred_input)
            pred_result = model.get_prediction(pred_input).summary_frame(alpha=0.05)
            st.success(f"Predicted {y_col}: {pred_result['mean'].values[0]:.4f}")
            st.caption(f"95% CI: [{pred_result['mean_ci_lower'].values[0]:.4f}, {pred_result['mean_ci_upper'].values[0]:.4f}]")

            # --- Download Results ---
            st.subheader("📥 Download Model Summary")
            buf = io.StringIO()
            buf.write(model.summary().as_text())
            b64 = base64.b64encode(buf.getvalue().encode()).decode()
            href = f'<a href="data:text/plain;base64,{b64}" download="regression_summary.txt">Download regression summary</a>'
            st.markdown(f'<div style="font-size: 0.9em">{href}</div>', unsafe_allow_html=True)

            # --- Explanation ---
            with st.expander("ℹ️ How to interpret the regression output"):
                st.markdown("""
                **Key terms:**

                - **coef**: The estimated coefficient (slope) for each variable. For the constant, it’s the Y-intercept.
                - **std err**: The standard error of the coefficient estimate.
                - **t** and **P>|t|**: The t-statistic and associated p-value for testing whether the coefficient is significantly different from 0.
                - **R-squared**: Proportion of variance in the dependent variable explained by the independent variable(s). Higher is better.
                - **Adj. R-squared**: Adjusted version of R-squared accounting for number of predictors.
                - **F-statistic** and **Prob (F-statistic)**: Tests the overall significance of the model.

                **Model assumptions:**
                - **Linearity**: The relationship between X and Y should be linear.
                - **Independence**: Observations should be independent of each other.
                - **Homoscedasticity**: Constant variance of residuals.
                - **Normality**: Residuals should be normally distributed.

                **Interpreting residuals plot:**
                - A random scatter indicates a good fit.
                - Patterns (e.g., curves or fanning out) suggest violations in assumptions.

                **Example Use Case:**
                Suppose your X is "Marketing Spend" and Y is "Sales". A significant positive coefficient means as you increase marketing spend, sales are likely to increase as well.
                This model can help predict expected sales given a planned marketing spend, assess whether the relationship is statistically reliable, and how confident you can be in those estimates.
                """)

        except Exception as e:
            st.error(f"An error occurred while fitting the model: {e}")
