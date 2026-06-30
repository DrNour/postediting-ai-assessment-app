import itertools
import json
import math
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st
from supabase import create_client

try:
    from scipy import stats
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
        silhouette_score,
    )
    from sklearn.model_selection import (
        GroupKFold,
        KFold,
        StratifiedKFold,
        cross_validate,
        train_test_split,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    ANALYTICS_LIBS_AVAILABLE = True
    ANALYTICS_IMPORT_ERROR = None

except Exception as import_error:
    ANALYTICS_LIBS_AVAILABLE = False
    ANALYTICS_IMPORT_ERROR = import_error


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Research Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("Research Analytics")
st.write(
    "Run statistical, machine-learning, and reproducibility-oriented analyses "
    "on EduApp submissions."
)


# ============================================================
# Teacher-only access gate
# ============================================================

configured_password = st.secrets.get("TEACHER_PASSWORD", None)

if not configured_password:
    st.error(
        "TEACHER_PASSWORD is not configured. Add it to Streamlit Secrets before using this page."
    )
    st.stop()

teacher_password = st.sidebar.text_input(
    "Teacher password",
    type="password",
)

if teacher_password != configured_password:
    st.warning("This page is restricted to the instructor.")
    st.stop()


if not ANALYTICS_LIBS_AVAILABLE:
    st.error("Some required analytics packages are missing.")
    st.code(str(ANALYTICS_IMPORT_ERROR))
    st.write("Add these to requirements.txt:")
    st.code("scipy\nstatsmodels\nscikit-learn\nnumpy")
    st.stop()


# ============================================================
# Supabase connection
# ============================================================

@st.cache_resource
def get_supabase_client():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception:
        st.error(
            "Supabase is not configured. Add SUPABASE_URL and SUPABASE_KEY "
            "to Streamlit Secrets."
        )
        st.stop()


supabase = get_supabase_client()


# ============================================================
# Load data
# ============================================================

@st.cache_data(ttl=60)
def load_submissions():
    """
    Loads submissions from Supabase.

    The page is tolerant of different schema versions.
    """
    try:
        response = (
            supabase.table("submissions")
            .select("*")
            .range(0, 9999)
            .order("submitted_at", desc=True)
            .execute()
        )
        return pd.DataFrame(response.data or [])

    except Exception:
        try:
            response = (
                supabase.table("submissions")
                .select("*")
                .range(0, 9999)
                .execute()
            )
            return pd.DataFrame(response.data or [])
        except Exception as error:
            st.error("Could not load submissions from Supabase.")
            st.code(str(error))
            return pd.DataFrame()


raw_df = load_submissions()

if raw_df.empty:
    st.warning("No submissions found.")
    st.stop()


# ============================================================
# Helper functions
# ============================================================

def safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def to_numeric_series(series):
    return pd.to_numeric(series, errors="coerce")


def is_probably_long_text(series):
    sample = series.dropna().astype(str)

    if sample.empty:
        return False

    median_length = sample.str.len().median()
    return median_length > 80


def get_numeric_columns(df):
    numeric_cols = []

    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        non_null_count = converted.notna().sum()

        if non_null_count >= 3:
            numeric_cols.append(col)

    return numeric_cols


def get_categorical_columns(df):
    categorical_cols = []

    for col in df.columns:
        if col in get_numeric_columns(df):
            continue

        if is_probably_long_text(df[col]):
            continue

        unique_count = df[col].dropna().astype(str).nunique()

        if 2 <= unique_count <= 50:
            categorical_cols.append(col)

    return categorical_cols


def dataframe_to_csv_download(df):
    return df.to_csv(index=False).encode("utf-8")


def descriptive_statistics(df, columns):
    rows = []

    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce").dropna()

        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        rows.append(
            {
                "variable": col,
                "n": int(values.shape[0]),
                "mean": values.mean(),
                "sd": values.std(ddof=1),
                "median": values.median(),
                "iqr": q3 - q1,
                "min": values.min(),
                "max": values.max(),
                "skewness": stats.skew(values, nan_policy="omit"),
                "kurtosis": stats.kurtosis(values, nan_policy="omit"),
            }
        )

    return pd.DataFrame(rows)


def cohens_d(x, y):
    x = pd.Series(x).dropna().astype(float)
    y = pd.Series(y).dropna().astype(float)

    nx = len(x)
    ny = len(y)

    if nx < 2 or ny < 2:
        return np.nan

    pooled_sd = math.sqrt(
        ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    )

    if pooled_sd == 0:
        return np.nan

    return (x.mean() - y.mean()) / pooled_sd


def hedges_g(x, y):
    d = cohens_d(x, y)

    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()

    n = len(x) + len(y)

    if n <= 3 or pd.isna(d):
        return np.nan

    correction = 1 - (3 / (4 * n - 9))
    return d * correction


def cliffs_delta(x, y):
    x = pd.Series(x).dropna().astype(float).to_numpy()
    y = pd.Series(y).dropna().astype(float).to_numpy()

    if len(x) == 0 or len(y) == 0:
        return np.nan

    greater = 0
    lower = 0

    for xi in x:
        greater += np.sum(xi > y)
        lower += np.sum(xi < y)

    return (greater - lower) / (len(x) * len(y))


def eta_squared_oneway(groups):
    clean_groups = [pd.Series(g).dropna().astype(float) for g in groups]
    clean_groups = [g for g in clean_groups if len(g) > 0]

    if len(clean_groups) < 2:
        return np.nan

    all_values = pd.concat(clean_groups)
    grand_mean = all_values.mean()

    ss_between = sum(len(g) * ((g.mean() - grand_mean) ** 2) for g in clean_groups)
    ss_total = sum(((all_values - grand_mean) ** 2))

    if ss_total == 0:
        return np.nan

    return ss_between / ss_total


def epsilon_squared_kruskal(h_statistic, k_groups, n_total):
    if n_total <= k_groups:
        return np.nan

    return max(0, (h_statistic - k_groups + 1) / (n_total - k_groups))


def cramers_v(contingency_table):
    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
    n = contingency_table.to_numpy().sum()

    if n == 0:
        return np.nan

    r, k = contingency_table.shape

    denominator = n * (min(k - 1, r - 1))

    if denominator == 0:
        return np.nan

    return math.sqrt(chi2 / denominator)


def cronbach_alpha(df):
    items = df.apply(pd.to_numeric, errors="coerce").dropna()

    if items.shape[1] < 2 or items.shape[0] < 2:
        return np.nan

    item_variances = items.var(axis=0, ddof=1)
    total_score = items.sum(axis=1)
    total_variance = total_score.var(ddof=1)

    if total_variance == 0:
        return np.nan

    n_items = items.shape[1]
    return (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)


def build_model_matrix(df, feature_columns):
    X = df[feature_columns].copy()

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype(str)

    X = pd.get_dummies(X, drop_first=True, dummy_na=True)
    X = X.apply(pd.to_numeric, errors="coerce")

    for col in X.columns:
        median_value = X[col].median()
        if pd.isna(median_value):
            median_value = 0
        X[col] = X[col].fillna(median_value)

    return X


def make_cv(task_type, y, groups=None, n_splits=5):
    n_samples = len(y)
    n_splits = max(2, min(n_splits, n_samples))

    if groups is not None:
        groups = pd.Series(groups).astype(str)
        unique_groups = groups.nunique()

        if unique_groups >= n_splits:
            return GroupKFold(n_splits=n_splits), groups

    if task_type == "classification":
        y_series = pd.Series(y)

        min_class_size = y_series.value_counts().min()

        if min_class_size >= n_splits:
            return StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=42,
            ), None

    return KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    ), None


def clean_analysis_df(df):
    cleaned = df.copy()

    for col in cleaned.columns:
        if cleaned[col].dtype == "object":
            cleaned[col] = cleaned[col].replace({"": np.nan, "None": np.nan, "nan": np.nan})

    return cleaned


df = clean_analysis_df(raw_df)

numeric_columns = get_numeric_columns(df)
categorical_columns = get_categorical_columns(df)

if not numeric_columns:
    st.error("No numeric columns found. Check that metric fields are being saved.")
    st.stop()


# ============================================================
# Sidebar filters
# ============================================================

st.sidebar.header("Filters")

filtered_df = df.copy()

candidate_filter_columns = [
    "semester",
    "group_name",
    "assignment_title",
    "assignment_code",
    "task_id",
    "student_id",
    "domain",
    "source_language",
    "target_language",
]

available_filter_columns = [
    col for col in candidate_filter_columns
    if col in filtered_df.columns
]

for col in available_filter_columns:
    values = sorted(filtered_df[col].dropna().astype(str).unique().tolist())

    if not values:
        continue

    selected_values = st.sidebar.multiselect(
        f"Filter by {col}",
        values,
        default=[],
    )

    if selected_values:
        filtered_df = filtered_df[
            filtered_df[col].astype(str).isin(selected_values)
        ]


st.sidebar.write(f"Rows after filtering: **{len(filtered_df)}**")

if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()


if filtered_df.empty:
    st.warning("No rows remain after filtering.")
    st.stop()


# ============================================================
# Main tabs
# ============================================================

tabs = st.tabs(
    [
        "Dataset",
        "Descriptives",
        "Assumptions",
        "Group Tests",
        "Paired Tests",
        "Correlations",
        "Categorical Tests",
        "Regression / ML",
        "PCA / Clustering",
        "Reliability",
        "Export",
    ]
)


# ============================================================
# Dataset tab
# ============================================================

with tabs[0]:
    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", len(filtered_df))

    with col2:
        st.metric("Columns", filtered_df.shape[1])

    with col3:
        if "student_id" in filtered_df.columns:
            st.metric(
                "Unique students",
                filtered_df["student_id"].dropna().astype(str).nunique(),
            )
        else:
            st.metric("Unique students", "N/A")

    st.subheader("Available numeric variables")
    st.write(numeric_columns)

    st.subheader("Available categorical variables")
    st.write(categorical_columns)

    st.subheader("Preview")
    st.dataframe(
        filtered_df.head(100),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download filtered dataset as CSV",
        data=dataframe_to_csv_download(filtered_df),
        file_name="eduapp_filtered_research_data.csv",
        mime="text/csv",
    )


# ============================================================
# Descriptives tab
# ============================================================

with tabs[1]:
    st.header("Descriptive Statistics")

    selected_desc_columns = st.multiselect(
        "Select numeric variables",
        numeric_columns,
        default=numeric_columns[: min(8, len(numeric_columns))],
    )

    if selected_desc_columns:
        desc_df = descriptive_statistics(filtered_df, selected_desc_columns)

        st.dataframe(
            desc_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download descriptive statistics",
            data=dataframe_to_csv_download(desc_df),
            file_name="descriptive_statistics.csv",
            mime="text/csv",
        )

    else:
        st.info("Select at least one numeric variable.")


# ============================================================
# Assumptions tab
# ============================================================

with tabs[2]:
    st.header("Assumption Checks")

    assumption_variable = st.selectbox(
        "Numeric variable",
        numeric_columns,
        key="assumption_variable",
    )

    values = pd.to_numeric(filtered_df[assumption_variable], errors="coerce").dropna()

    st.subheader("Normality")

    if len(values) < 3:
        st.warning("At least 3 observations are needed for normality tests.")
    else:
        if len(values) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(values)
            st.write(
                {
                    "test": "Shapiro-Wilk",
                    "statistic": shapiro_stat,
                    "p_value": shapiro_p,
                    "n": len(values),
                }
            )
        else:
            normal_stat, normal_p = stats.normaltest(values)
            st.write(
                {
                    "test": "D'Agostino-Pearson normality test",
                    "statistic": normal_stat,
                    "p_value": normal_p,
                    "n": len(values),
                }
            )

    st.subheader("Homogeneity of variance")

    if categorical_columns:
        grouping_variable = st.selectbox(
            "Grouping variable",
            categorical_columns,
            key="levene_grouping_variable",
        )

        temp = filtered_df[[assumption_variable, grouping_variable]].dropna()
        temp[assumption_variable] = pd.to_numeric(
            temp[assumption_variable],
            errors="coerce",
        )
        temp = temp.dropna()

        groups = [
            group[assumption_variable].dropna()
            for _, group in temp.groupby(grouping_variable)
            if len(group[assumption_variable].dropna()) >= 2
        ]

        if len(groups) >= 2:
            levene_stat, levene_p = stats.levene(*groups, center="median")
            st.write(
                {
                    "test": "Levene test",
                    "statistic": levene_stat,
                    "p_value": levene_p,
                    "groups": len(groups),
                }
            )
        else:
            st.info("Need at least two groups with at least two observations each.")

    else:
        st.info("No categorical grouping variables available.")


# ============================================================
# Group tests tab
# ============================================================

with tabs[3]:
    st.header("Independent Group Comparisons")

    if not categorical_columns:
        st.warning("No categorical variables available for group tests.")
    else:
        metric_col = st.selectbox(
            "Outcome / metric",
            numeric_columns,
            key="group_metric",
        )

        group_col = st.selectbox(
            "Grouping variable",
            categorical_columns,
            key="group_col",
        )

        temp = filtered_df[[metric_col, group_col]].copy()
        temp[metric_col] = pd.to_numeric(temp[metric_col], errors="coerce")
        temp[group_col] = temp[group_col].astype(str)
        temp = temp.dropna()

        levels = sorted(temp[group_col].unique().tolist())

        st.write(f"Detected groups: {levels}")

        if len(levels) < 2:
            st.warning("At least two groups are needed.")
        else:
            groups = [
                temp[temp[group_col] == level][metric_col].dropna()
                for level in levels
            ]

            group_summary = temp.groupby(group_col)[metric_col].agg(
                ["count", "mean", "std", "median", "min", "max"]
            ).reset_index()

            st.subheader("Group summary")
            st.dataframe(
                group_summary,
                use_container_width=True,
                hide_index=True,
            )

            if len(levels) == 2:
                x = groups[0]
                y = groups[1]

                equal_var_t = stats.ttest_ind(x, y, equal_var=True, nan_policy="omit")
                welch_t = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
                mann_whitney = stats.mannwhitneyu(x, y, alternative="two-sided")

                results_df = pd.DataFrame(
                    [
                        {
                            "test": "Independent-samples t-test",
                            "statistic": equal_var_t.statistic,
                            "p_value": equal_var_t.pvalue,
                            "effect_size": "Cohen's d",
                            "effect_value": cohens_d(x, y),
                        },
                        {
                            "test": "Welch t-test",
                            "statistic": welch_t.statistic,
                            "p_value": welch_t.pvalue,
                            "effect_size": "Hedges' g",
                            "effect_value": hedges_g(x, y),
                        },
                        {
                            "test": "Mann-Whitney U",
                            "statistic": mann_whitney.statistic,
                            "p_value": mann_whitney.pvalue,
                            "effect_size": "Cliff's delta",
                            "effect_value": cliffs_delta(x, y),
                        },
                    ]
                )

                st.subheader("Two-group tests")
                st.dataframe(
                    results_df,
                    use_container_width=True,
                    hide_index=True,
                )

            else:
                anova_result = stats.f_oneway(*groups)
                kruskal_result = stats.kruskal(*groups)

                eta2 = eta_squared_oneway(groups)
                eps2 = epsilon_squared_kruskal(
                    kruskal_result.statistic,
                    len(groups),
                    len(temp),
                )

                omnibus_df = pd.DataFrame(
                    [
                        {
                            "test": "One-way ANOVA",
                            "statistic": anova_result.statistic,
                            "p_value": anova_result.pvalue,
                            "effect_size": "eta_squared",
                            "effect_value": eta2,
                        },
                        {
                            "test": "Kruskal-Wallis",
                            "statistic": kruskal_result.statistic,
                            "p_value": kruskal_result.pvalue,
                            "effect_size": "epsilon_squared",
                            "effect_value": eps2,
                        },
                    ]
                )

                st.subheader("Omnibus tests")
                st.dataframe(
                    omnibus_df,
                    use_container_width=True,
                    hide_index=True,
                )

                st.subheader("Pairwise post-hoc comparisons")

                pairwise_method = st.radio(
                    "Pairwise test",
                    ["Welch t-test", "Mann-Whitney U"],
                    horizontal=True,
                )

                correction_method = st.selectbox(
                    "Multiple-comparison correction",
                    ["holm", "fdr_bh", "bonferroni"],
                    index=0,
                )

                pairwise_rows = []

                for a, b in itertools.combinations(levels, 2):
                    group_a = temp[temp[group_col] == a][metric_col].dropna()
                    group_b = temp[temp[group_col] == b][metric_col].dropna()

                    if pairwise_method == "Welch t-test":
                        test_result = stats.ttest_ind(
                            group_a,
                            group_b,
                            equal_var=False,
                            nan_policy="omit",
                        )
                        effect_name = "Hedges' g"
                        effect_value = hedges_g(group_a, group_b)

                    else:
                        test_result = stats.mannwhitneyu(
                            group_a,
                            group_b,
                            alternative="two-sided",
                        )
                        effect_name = "Cliff's delta"
                        effect_value = cliffs_delta(group_a, group_b)

                    pairwise_rows.append(
                        {
                            "group_a": a,
                            "group_b": b,
                            "test": pairwise_method,
                            "statistic": test_result.statistic,
                            "raw_p_value": test_result.pvalue,
                            "effect_size": effect_name,
                            "effect_value": effect_value,
                        }
                    )

                pairwise_df = pd.DataFrame(pairwise_rows)

                if not pairwise_df.empty:
                    reject, adjusted_p, _, _ = multipletests(
                        pairwise_df["raw_p_value"],
                        method=correction_method,
                    )

                    pairwise_df["adjusted_p_value"] = adjusted_p
                    pairwise_df["reject_after_correction"] = reject

                    st.dataframe(
                        pairwise_df,
                        use_container_width=True,
                        hide_index=True,
                    )


# ============================================================
# Paired tests tab
# ============================================================

with tabs[4]:
    st.header("Paired / Within-Record Tests")

    col1, col2 = st.columns(2)

    with col1:
        paired_a = st.selectbox(
            "Variable A",
            numeric_columns,
            key="paired_a",
        )

    with col2:
        paired_b = st.selectbox(
            "Variable B",
            numeric_columns,
            key="paired_b",
            index=1 if len(numeric_columns) > 1 else 0,
        )

    paired_df = filtered_df[[paired_a, paired_b]].copy()
    paired_df[paired_a] = pd.to_numeric(paired_df[paired_a], errors="coerce")
    paired_df[paired_b] = pd.to_numeric(paired_df[paired_b], errors="coerce")
    paired_df = paired_df.dropna()

    if paired_df.empty or len(paired_df) < 2:
        st.warning("At least two complete paired observations are required.")
    else:
        x = paired_df[paired_a]
        y = paired_df[paired_b]
        diff = x - y

        paired_t = stats.ttest_rel(x, y, nan_policy="omit")

        try:
            wilcoxon_result = stats.wilcoxon(x, y)
            wilcoxon_stat = wilcoxon_result.statistic
            wilcoxon_p = wilcoxon_result.pvalue
        except Exception:
            wilcoxon_stat = np.nan
            wilcoxon_p = np.nan

        if diff.std(ddof=1) == 0:
            cohens_dz = np.nan
        else:
            cohens_dz = diff.mean() / diff.std(ddof=1)

        paired_results = pd.DataFrame(
            [
                {
                    "test": "Paired t-test",
                    "statistic": paired_t.statistic,
                    "p_value": paired_t.pvalue,
                    "effect_size": "Cohen's dz",
                    "effect_value": cohens_dz,
                    "n": len(paired_df),
                },
                {
                    "test": "Wilcoxon signed-rank",
                    "statistic": wilcoxon_stat,
                    "p_value": wilcoxon_p,
                    "effect_size": "median_difference",
                    "effect_value": diff.median(),
                    "n": len(paired_df),
                },
            ]
        )

        st.dataframe(
            paired_results,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# Correlations tab
# ============================================================

with tabs[5]:
    st.header("Correlation Analysis")

    col1, col2 = st.columns(2)

    with col1:
        x_col = st.selectbox(
            "X variable",
            numeric_columns,
            key="corr_x",
        )

    with col2:
        y_col = st.selectbox(
            "Y variable",
            numeric_columns,
            key="corr_y",
            index=1 if len(numeric_columns) > 1 else 0,
        )

    corr_df = filtered_df[[x_col, y_col]].copy()
    corr_df[x_col] = pd.to_numeric(corr_df[x_col], errors="coerce")
    corr_df[y_col] = pd.to_numeric(corr_df[y_col], errors="coerce")
    corr_df = corr_df.dropna()

    if len(corr_df) < 3:
        st.warning("At least three complete observations are required.")
    else:
        x = corr_df[x_col]
        y = corr_df[y_col]

        pearson = stats.pearsonr(x, y)
        spearman = stats.spearmanr(x, y)
        kendall = stats.kendalltau(x, y)

        corr_results = pd.DataFrame(
            [
                {
                    "method": "Pearson",
                    "correlation": pearson.statistic,
                    "p_value": pearson.pvalue,
                    "n": len(corr_df),
                },
                {
                    "method": "Spearman",
                    "correlation": spearman.statistic,
                    "p_value": spearman.pvalue,
                    "n": len(corr_df),
                },
                {
                    "method": "Kendall tau",
                    "correlation": kendall.statistic,
                    "p_value": kendall.pvalue,
                    "n": len(corr_df),
                },
            ]
        )

        st.dataframe(
            corr_results,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Scatter data preview")
        st.scatter_chart(
            corr_df,
            x=x_col,
            y=y_col,
        )

        st.subheader("Optional permutation test")

        run_permutation = st.checkbox(
            "Run permutation test for Spearman correlation",
            value=False,
        )

        if run_permutation:
            n_resamples = st.slider(
                "Number of permutation resamples",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
            )

            def spearman_statistic(a, b):
                return stats.spearmanr(a, b).statistic

            permutation_result = stats.permutation_test(
                (x.to_numpy(), y.to_numpy()),
                spearman_statistic,
                permutation_type="pairings",
                n_resamples=n_resamples,
                random_state=42,
            )

            st.write(
                {
                    "method": "Permutation test for Spearman correlation",
                    "statistic": permutation_result.statistic,
                    "p_value": permutation_result.pvalue,
                    "n_resamples": n_resamples,
                }
            )


# ============================================================
# Categorical tests tab
# ============================================================

with tabs[6]:
    st.header("Categorical Association Tests")

    if len(categorical_columns) < 2:
        st.warning("At least two categorical variables are required.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            cat_a = st.selectbox(
                "Categorical variable A",
                categorical_columns,
                key="cat_a",
            )

        with col2:
            cat_b = st.selectbox(
                "Categorical variable B",
                categorical_columns,
                key="cat_b",
                index=1 if len(categorical_columns) > 1 else 0,
            )

        cat_df = filtered_df[[cat_a, cat_b]].dropna().copy()
        cat_df[cat_a] = cat_df[cat_a].astype(str)
        cat_df[cat_b] = cat_df[cat_b].astype(str)

        contingency_table = pd.crosstab(cat_df[cat_a], cat_df[cat_b])

        st.subheader("Contingency table")
        st.dataframe(contingency_table, use_container_width=True)

        if contingency_table.shape[0] >= 2 and contingency_table.shape[1] >= 2:
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

            cat_results = [
                {
                    "test": "Chi-square test of independence",
                    "statistic": chi2,
                    "p_value": p,
                    "degrees_of_freedom": dof,
                    "effect_size": "Cramer's V",
                    "effect_value": cramers_v(contingency_table),
                }
            ]

            if contingency_table.shape == (2, 2):
                fisher_odds, fisher_p = stats.fisher_exact(contingency_table)
                cat_results.append(
                    {
                        "test": "Fisher exact test",
                        "statistic": fisher_odds,
                        "p_value": fisher_p,
                        "degrees_of_freedom": None,
                        "effect_size": "odds_ratio",
                        "effect_value": fisher_odds,
                    }
                )

            st.subheader("Results")
            st.dataframe(
                pd.DataFrame(cat_results),
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.warning("The contingency table is too small for association tests.")


# ============================================================
# Regression / ML tab
# ============================================================

with tabs[7]:
    st.header("Regression and Machine-Learning Evaluation")

    ml_task = st.radio(
        "Task type",
        ["Regression", "Binary classification"],
        horizontal=True,
    )

    if ml_task == "Regression":
        target_col = st.selectbox(
            "Numeric target variable",
            numeric_columns,
            key="reg_target",
        )

    else:
        target_options = numeric_columns + categorical_columns
        target_col = st.selectbox(
            "Target variable",
            target_options,
            key="clf_target",
        )

    possible_features = [
        col for col in filtered_df.columns
        if col != target_col
        and not is_probably_long_text(filtered_df[col])
    ]

    default_features = [
        col for col in possible_features
        if col in numeric_columns
        and col != target_col
    ][:8]

    feature_cols = st.multiselect(
        "Feature variables",
        possible_features,
        default=default_features,
    )

    group_options = ["None"] + [
        col for col in ["student_id", "participant_id", "group_name", "semester"]
        if col in filtered_df.columns
    ]

    group_col = st.selectbox(
        "Grouped cross-validation column",
        group_options,
        index=0,
        help="Use student_id if students may submit more than once.",
    )

    cv_folds = st.slider(
        "Cross-validation folds",
        min_value=2,
        max_value=10,
        value=5,
    )

    if not feature_cols:
        st.warning("Select at least one feature variable.")
    else:
        model_df = filtered_df[[target_col] + feature_cols].copy()

        if group_col != "None":
            model_df[group_col] = filtered_df[group_col]

        X = build_model_matrix(model_df, feature_cols)

        if ml_task == "Regression":
            y = pd.to_numeric(model_df[target_col], errors="coerce")
            valid_mask = y.notna()

            X_valid = X.loc[valid_mask]
            y_valid = y.loc[valid_mask]

            if len(y_valid) < 5:
                st.warning("At least five complete rows are recommended for regression.")
            else:
                groups = None

                if group_col != "None":
                    groups = model_df.loc[valid_mask, group_col]

                cv, cv_groups = make_cv(
                    task_type="regression",
                    y=y_valid,
                    groups=groups,
                    n_splits=cv_folds,
                )

                estimators = {
                    "Ridge regression": make_pipeline(
                        StandardScaler(),
                        Ridge(alpha=1.0),
                    ),
                    "Random forest regression": RandomForestRegressor(
                        n_estimators=200,
                        min_samples_leaf=5,
                        random_state=42,
                    ),
                }

                scoring = {
                    "mae": "neg_mean_absolute_error",
                    "rmse": "neg_root_mean_squared_error",
                    "r2": "r2",
                }

                cv_rows = []

                for model_name, estimator in estimators.items():
                    cv_result = cross_validate(
                        estimator,
                        X_valid,
                        y_valid,
                        cv=cv,
                        groups=cv_groups,
                        scoring=scoring,
                        n_jobs=None,
                        error_score=np.nan,
                    )

                    cv_rows.append(
                        {
                            "model": model_name,
                            "mae_mean": -np.nanmean(cv_result["test_mae"]),
                            "mae_sd": np.nanstd(-cv_result["test_mae"]),
                            "rmse_mean": -np.nanmean(cv_result["test_rmse"]),
                            "rmse_sd": np.nanstd(-cv_result["test_rmse"]),
                            "r2_mean": np.nanmean(cv_result["test_r2"]),
                            "r2_sd": np.nanstd(cv_result["test_r2"]),
                        }
                    )

                st.subheader("Cross-validated regression results")
                st.dataframe(
                    pd.DataFrame(cv_rows),
                    use_container_width=True,
                    hide_index=True,
                )

                st.subheader("OLS regression coefficients")

                try:
                    X_ols = sm.add_constant(X_valid)
                    ols_model = sm.OLS(y_valid, X_ols).fit()

                    coefficients_df = pd.DataFrame(
                        {
                            "term": ols_model.params.index,
                            "coefficient": ols_model.params.values,
                            "std_error": ols_model.bse.values,
                            "t": ols_model.tvalues.values,
                            "p_value": ols_model.pvalues.values,
                        }
                    )

                    st.write(
                        {
                            "r_squared": ols_model.rsquared,
                            "adjusted_r_squared": ols_model.rsquared_adj,
                            "aic": ols_model.aic,
                            "bic": ols_model.bic,
                        }
                    )

                    st.dataframe(
                        coefficients_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                except Exception as error:
                    st.warning("OLS regression failed. This can happen with collinear features.")
                    st.code(str(error))

                st.subheader("Random forest feature importance")

                try:
                    rf = RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=5,
                        random_state=42,
                    )
                    rf.fit(X_valid, y_valid)

                    importance_df = pd.DataFrame(
                        {
                            "feature": X_valid.columns,
                            "importance": rf.feature_importances_,
                        }
                    ).sort_values("importance", ascending=False)

                    st.dataframe(
                        importance_df.head(30),
                        use_container_width=True,
                        hide_index=True,
                    )

                    run_perm_importance = st.checkbox(
                        "Run permutation importance",
                        value=False,
                        key="reg_perm_importance",
                    )

                    if run_perm_importance:
                        perm = permutation_importance(
                            rf,
                            X_valid,
                            y_valid,
                            n_repeats=10,
                            random_state=42,
                            scoring="neg_mean_absolute_error",
                        )

                        perm_df = pd.DataFrame(
                            {
                                "feature": X_valid.columns,
                                "importance_mean": perm.importances_mean,
                                "importance_sd": perm.importances_std,
                            }
                        ).sort_values("importance_mean", ascending=False)

                        st.dataframe(
                            perm_df.head(30),
                            use_container_width=True,
                            hide_index=True,
                        )

                except Exception as error:
                    st.warning("Feature importance failed.")
                    st.code(str(error))

        else:
            target_series = model_df[target_col].copy()

            create_median_split = False

            if target_col in numeric_columns:
                create_median_split = st.checkbox(
                    "Create high/low target using the median",
                    value=True,
                )

            if create_median_split:
                numeric_target = pd.to_numeric(target_series, errors="coerce")
                median_value = numeric_target.median()
                y_raw = np.where(numeric_target > median_value, "high", "low")
                valid_mask = numeric_target.notna()

            else:
                target_series = target_series.astype(str)
                valid_mask = target_series.notna()
                y_raw = target_series

            X_valid = X.loc[valid_mask]
            y_raw_valid = pd.Series(y_raw, index=X.index).loc[valid_mask]

            unique_classes = sorted(pd.Series(y_raw_valid).dropna().unique().tolist())

            if len(unique_classes) != 2:
                st.warning(
                    "Binary classification requires exactly two classes. "
                    f"Detected classes: {unique_classes}"
                )
            else:
                label_encoder = LabelEncoder()
                y_valid = label_encoder.fit_transform(y_raw_valid.astype(str))

                groups = None

                if group_col != "None":
                    groups = model_df.loc[valid_mask, group_col]

                cv, cv_groups = make_cv(
                    task_type="classification",
                    y=y_valid,
                    groups=groups,
                    n_splits=cv_folds,
                )

                estimators = {
                    "Logistic regression": make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                        ),
                    ),
                    "Random forest classification": RandomForestClassifier(
                        n_estimators=200,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=42,
                    ),
                }

                scoring = {
                    "accuracy": "accuracy",
                    "balanced_accuracy": "balanced_accuracy",
                    "f1_weighted": "f1_weighted",
                    "roc_auc": "roc_auc",
                }

                cv_rows = []

                for model_name, estimator in estimators.items():
                    try:
                        cv_result = cross_validate(
                            estimator,
                            X_valid,
                            y_valid,
                            cv=cv,
                            groups=cv_groups,
                            scoring=scoring,
                            error_score=np.nan,
                        )

                        cv_rows.append(
                            {
                                "model": model_name,
                                "accuracy_mean": np.nanmean(cv_result["test_accuracy"]),
                                "accuracy_sd": np.nanstd(cv_result["test_accuracy"]),
                                "balanced_accuracy_mean": np.nanmean(
                                    cv_result["test_balanced_accuracy"]
                                ),
                                "balanced_accuracy_sd": np.nanstd(
                                    cv_result["test_balanced_accuracy"]
                                ),
                                "f1_weighted_mean": np.nanmean(
                                    cv_result["test_f1_weighted"]
                                ),
                                "f1_weighted_sd": np.nanstd(
                                    cv_result["test_f1_weighted"]
                                ),
                                "auc_mean": np.nanmean(cv_result["test_roc_auc"]),
                                "auc_sd": np.nanstd(cv_result["test_roc_auc"]),
                            }
                        )

                    except Exception as error:
                        cv_rows.append(
                            {
                                "model": model_name,
                                "accuracy_mean": np.nan,
                                "accuracy_sd": np.nan,
                                "balanced_accuracy_mean": np.nan,
                                "balanced_accuracy_sd": np.nan,
                                "f1_weighted_mean": np.nan,
                                "f1_weighted_sd": np.nan,
                                "auc_mean": np.nan,
                                "auc_sd": np.nan,
                                "error": str(error),
                            }
                        )

                st.subheader("Cross-validated classification results")
                st.dataframe(
                    pd.DataFrame(cv_rows),
                    use_container_width=True,
                    hide_index=True,
                )

                st.subheader("Random forest feature importance")

                try:
                    rf = RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=42,
                    )
                    rf.fit(X_valid, y_valid)

                    importance_df = pd.DataFrame(
                        {
                            "feature": X_valid.columns,
                            "importance": rf.feature_importances_,
                        }
                    ).sort_values("importance", ascending=False)

                    st.dataframe(
                        importance_df.head(30),
                        use_container_width=True,
                        hide_index=True,
                    )

                    run_perm_importance = st.checkbox(
                        "Run permutation importance",
                        value=False,
                        key="clf_perm_importance",
                    )

                    if run_perm_importance:
                        perm = permutation_importance(
                            rf,
                            X_valid,
                            y_valid,
                            n_repeats=10,
                            random_state=42,
                            scoring="balanced_accuracy",
                        )

                        perm_df = pd.DataFrame(
                            {
                                "feature": X_valid.columns,
                                "importance_mean": perm.importances_mean,
                                "importance_sd": perm.importances_std,
                            }
                        ).sort_values("importance_mean", ascending=False)

                        st.dataframe(
                            perm_df.head(30),
                            use_container_width=True,
                            hide_index=True,
                        )

                except Exception as error:
                    st.warning("Feature importance failed.")
                    st.code(str(error))


# ============================================================
# PCA / Clustering tab
# ============================================================

with tabs[8]:
    st.header("PCA and Clustering")

    selected_cluster_features = st.multiselect(
        "Select numeric features for PCA / clustering",
        numeric_columns,
        default=numeric_columns[: min(5, len(numeric_columns))],
    )

    if len(selected_cluster_features) < 2:
        st.info("Select at least two numeric features.")
    else:
        cluster_df = filtered_df[selected_cluster_features].copy()
        cluster_df = cluster_df.apply(pd.to_numeric, errors="coerce")

        for col in cluster_df.columns:
            cluster_df[col] = cluster_df[col].fillna(cluster_df[col].median())

        cluster_df = cluster_df.dropna()

        if len(cluster_df) < 5:
            st.warning("At least five complete rows are recommended.")
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(cluster_df)

            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_scaled)

            n_clusters = st.slider(
                "Number of KMeans clusters",
                min_value=2,
                max_value=min(10, len(cluster_df) - 1),
                value=2,
            )

            kmeans = KMeans(
                n_clusters=n_clusters,
                n_init=10,
                random_state=42,
            )

            labels = kmeans.fit_predict(X_scaled)

            if len(set(labels)) > 1:
                sil = silhouette_score(X_scaled, labels)
            else:
                sil = np.nan

            st.write(
                {
                    "pca_explained_variance_pc1": pca.explained_variance_ratio_[0],
                    "pca_explained_variance_pc2": pca.explained_variance_ratio_[1],
                    "silhouette_score": sil,
                }
            )

            plot_df = pd.DataFrame(
                {
                    "PC1": X_pca[:, 0],
                    "PC2": X_pca[:, 1],
                    "cluster": labels.astype(str),
                }
            )

            st.scatter_chart(
                plot_df,
                x="PC1",
                y="PC2",
                color="cluster",
            )

            st.subheader("Cluster assignments preview")
            st.dataframe(
                plot_df.head(100),
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# Reliability tab
# ============================================================

with tabs[9]:
    st.header("Reliability Analysis")

    st.write(
        "Use this for rubric items or multiple scoring dimensions, for example "
        "accuracy, fluency, terminology, style, and coherence."
    )

    selected_reliability_items = st.multiselect(
        "Select rubric / scale items",
        numeric_columns,
    )

    if len(selected_reliability_items) < 2:
        st.info("Select at least two numeric items.")
    else:
        reliability_df = filtered_df[selected_reliability_items].copy()
        alpha = cronbach_alpha(reliability_df)

        st.metric("Cronbach's alpha", format(alpha, ".3f") if isinstance(alpha, float) else alpha)

        item_desc = descriptive_statistics(filtered_df, selected_reliability_items)

        st.subheader("Item descriptives")
        st.dataframe(
            item_desc,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# Export tab
# ============================================================

with tabs[10]:
    st.header("Export Research Data")

    st.download_button(
        "Download filtered dataset",
        data=dataframe_to_csv_download(filtered_df),
        file_name="eduapp_research_filtered_dataset.csv",
        mime="text/csv",
    )

    selected_export_columns = st.multiselect(
        "Select columns to export",
        filtered_df.columns.tolist(),
        default=[
            col for col in [
                "student_id",
                "student_name",
                "assignment_code",
                "task_id",
                "assignment_title",
                "group_name",
                "semester",
                "domain",
                "editing_time_seconds",
                "raw_mt_word_count",
                "pe_word_count",
                "mt_pe_cosine_similarity",
                "mt_pe_edit_distance_ratio",
                "mt_pe_length_ratio",
                "mt_pe_overlap_bleu",
                "mt_pe_overlap_chrf",
                "mt_pe_overlap_ter",
                "pe_quality_bleu",
                "pe_quality_chrf",
                "pe_quality_ter",
                "teacher_score",
            ]
            if col in filtered_df.columns
        ],
    )

    if selected_export_columns:
        export_df = filtered_df[selected_export_columns]

        st.dataframe(
            export_df.head(100),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download selected columns",
            data=dataframe_to_csv_download(export_df),
            file_name="eduapp_selected_research_columns.csv",
            mime="text/csv",
        )

    st.subheader("Recommended reporting note")

    st.info(
        "For publication, report descriptive statistics, assumption checks, "
        "test statistics, p-values, effect sizes, correction method for multiple "
        "comparisons, and whether cross-validation was grouped by student."
    )
