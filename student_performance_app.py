import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Dark background */
.stApp {
    background: #0a0e1a;
    color: #e8eaf2;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f1628;
    border-right: 1px solid #1e2a45;
}
section[data-testid="stSidebar"] * {
    color: #c8d0e8 !important;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a2240 0%, #0f1628 100%);
    border: 1px solid #2a3a60;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    margin-bottom: 12px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: #4a6fa5;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.metric-label {
    font-size: 0.85rem;
    color: #7a8ab0;
    margin-top: 8px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Section headers */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #e8eaf2;
    border-left: 4px solid #60a5fa;
    padding-left: 14px;
    margin: 32px 0 20px 0;
}

/* Top banner */
.hero-banner {
    background: linear-gradient(135deg, #1a2855 0%, #0f1e3d 50%, #1a1230 100%);
    border: 1px solid #2a3a70;
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(96,165,250,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #fff;
    margin: 0 0 8px 0;
}
.hero-subtitle {
    color: #7a9acc;
    font-size: 1.05rem;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #0f1628;
    border-radius: 12px;
    padding: 6px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #7a8ab0;
    border-radius: 8px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #1e2e50 !important;
    color: #60a5fa !important;
}

/* Plotly chart backgrounds */
.js-plotly-plot {
    border-radius: 12px;
}

/* DataFrame */
.dataframe {
    background: #0f1628 !important;
    color: #c8d0e8 !important;
}

/* Selectbox and slider */
.stSelectbox > div > div, .stMultiSelect > div > div {
    background: #1a2240 !important;
    border: 1px solid #2a3a60 !important;
    color: #c8d0e8 !important;
}

/* Insight box */
.insight-box {
    background: #111c35;
    border: 1px solid #2a3a60;
    border-left: 4px solid #60a5fa;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 0.95rem;
    color: #c8d0e8;
}
</style>
""", unsafe_allow_html=True)

# ─── PLOTLY THEME ────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='#0f1628',
    plot_bgcolor='#111c35',
    font=dict(color='#c8d0e8', family='Space Grotesk'),
    xaxis=dict(gridcolor='#1e2a45', zerolinecolor='#1e2a45'),
    yaxis=dict(gridcolor='#1e2a45', zerolinecolor='#1e2a45'),
    margin=dict(l=20, r=20, t=40, b=20),
)
COLOR_SEQ = ['#60a5fa','#a78bfa','#34d399','#f59e0b','#f87171','#38bdf8','#c084fc']
COLOR_SCALE = 'Blues'

# ─── LOAD DATA ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    if hasattr(file, 'read'):
        df = pd.read_csv(file)
    else:
        df = pd.read_csv(file)
    return df

@st.cache_data
def encode_data(df):
    df_enc = df.copy()
    le = LabelEncoder()
    cat_cols = df_enc.select_dtypes(include='object').columns
    for col in cat_cols:
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    return df_enc

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Student Performance")
    st.markdown("---")
    uploaded = st.file_uploader("Upload your CSV", type=["csv"])
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("", [
        "📊 Overview",
        "🔍 Exploratory Analysis",
        "📈 Feature Insights",
        "🤖 Prediction Model",
        "📋 Raw Data"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<small style='color:#4a5a80'>Built with Streamlit + Plotly</small>", unsafe_allow_html=True)

# ─── LOAD ────────────────────────────────────────────────────────────────────────
default_path = "/mnt/user-data/uploads/StudentPerformanceFactors.csv"
try:
    if uploaded:
        df = load_data(uploaded)
    else:
        df = load_data(default_path)
except:
    st.error("Please upload a dataset to begin.")
    st.stop()

df_enc = encode_data(df)
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()
target = 'Exam_Score' if 'Exam_Score' in df.columns else num_cols[-1]

# ─── HERO BANNER ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-banner">
  <div class="hero-title">🎓 Student Performance Analysis</div>
  <div class="hero-subtitle">Explore, analyze, and predict academic outcomes from {len(df):,} student records across {len(df.columns)} features</div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        (f"{len(df):,}", "Total Students"),
        (f"{df[target].mean():.1f}", "Avg Exam Score"),
        (f"{df[target].max():.0f}", "Highest Score"),
        (f"{df[target].min():.0f}", "Lowest Score"),
    ]
    for col, (val, label) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Score Distribution</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(df, x=target, nbins=40, color_discrete_sequence=['#60a5fa'],
                           title="Exam Score Distribution")
        fig.update_layout(**PLOTLY_LAYOUT)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.box(df, y=target, color_discrete_sequence=['#a78bfa'],
                     title="Exam Score Box Plot")
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # Score by categorical
    st.markdown('<div class="section-title">Score by Key Factors</div>', unsafe_allow_html=True)
    if cat_cols:
        sel_cat = st.selectbox("Choose a category", cat_cols[:6])
        fig = px.box(df, x=sel_cat, y=target, color=sel_cat,
                     color_discrete_sequence=COLOR_SEQ,
                     title=f"Exam Score by {sel_cat}")
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
    corr = df_enc[num_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale='RdBu_r',
                    title="Feature Correlations", aspect="auto")
    fig.update_layout(**PLOTLY_LAYOUT, height=500)
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORATORY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Exploratory Analysis":
    st.markdown('<div class="section-title">Distribution Explorer</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        sel_num = st.selectbox("Numeric Feature", num_cols)
    with c2:
        group_by = st.selectbox("Group By (optional)", ["None"] + cat_cols)

    if group_by == "None":
        fig = px.histogram(df, x=sel_num, nbins=40, color_discrete_sequence=['#60a5fa'],
                           marginal="box", title=f"Distribution of {sel_num}")
    else:
        fig = px.histogram(df, x=sel_num, color=group_by, nbins=40,
                           color_discrete_sequence=COLOR_SEQ,
                           marginal="box", barmode="overlay",
                           title=f"{sel_num} grouped by {group_by}")
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # Scatter
    st.markdown('<div class="section-title">Scatter Plot</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    x_ax = c1.selectbox("X Axis", num_cols, index=0)
    y_ax = c2.selectbox("Y Axis", num_cols, index=len(num_cols)-1)
    color_by = c3.selectbox("Color By", ["None"] + cat_cols)

    fig = px.scatter(df, x=x_ax, y=y_ax,
                     color=None if color_by == "None" else color_by,
                     color_discrete_sequence=COLOR_SEQ,
                     trendline="ols",
                     opacity=0.6,
                     title=f"{x_ax} vs {y_ax}")
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # Categorical counts
    st.markdown('<div class="section-title">Categorical Breakdown</div>', unsafe_allow_html=True)
    sel_cat2 = st.selectbox("Select categorical column", cat_cols)
    vc = df[sel_cat2].value_counts().reset_index()
    vc.columns = [sel_cat2, 'Count']
    fig = px.bar(vc, x=sel_cat2, y='Count', color=sel_cat2,
                 color_discrete_sequence=COLOR_SEQ, title=f"Distribution of {sel_cat2}")
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — FEATURE INSIGHTS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📈 Feature Insights":
    st.markdown('<div class="section-title">Top Features Correlated with Exam Score</div>', unsafe_allow_html=True)

    corr_target = df_enc.corr()[target].drop(target).sort_values(key=abs, ascending=False)
    fig = px.bar(
        x=corr_target.values, y=corr_target.index,
        orientation='h',
        color=corr_target.values,
        color_continuous_scale='RdBu',
        labels={'x': 'Correlation', 'y': 'Feature'},
        title=f"Pearson Correlation with {target}"
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=500, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # Auto key insight
    top_pos = corr_target[corr_target > 0].index[0]
    top_neg = corr_target[corr_target < 0].index[0] if (corr_target < 0).any() else None
    st.markdown(f"""
    <div class="insight-box">💡 <b>{top_pos}</b> has the strongest positive correlation with exam score ({corr_target[top_pos]:.2f}).
    {"<br>⚠️ <b>" + top_neg + "</b> has the strongest negative correlation (" + f"{corr_target[top_neg]:.2f}" + ")." if top_neg else ""}
    </div>""", unsafe_allow_html=True)

    # Multi-feature box plots against score categories
    st.markdown('<div class="section-title">Score Category Analysis</div>', unsafe_allow_html=True)

    df['Score_Category'] = pd.cut(df[target],
                                   bins=[0, 60, 70, 80, 100],
                                   labels=['Below 60', '60–70', '70–80', '80+'])

    sel_feat = st.selectbox("Choose numeric feature to compare across score groups", num_cols[:-1])
    fig = px.violin(df, x='Score_Category', y=sel_feat, color='Score_Category',
                    color_discrete_sequence=COLOR_SEQ, box=True,
                    title=f"{sel_feat} across Score Categories")
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # Pairplot subset
    st.markdown('<div class="section-title">Pairwise Relationships</div>', unsafe_allow_html=True)
    pair_cols = st.multiselect("Select features (max 4)", num_cols, default=num_cols[:4])
    if len(pair_cols) >= 2:
        fig = px.scatter_matrix(df, dimensions=pair_cols[:4], color=target,
                                color_continuous_scale='Viridis',
                                title="Scatter Matrix")
        fig.update_layout(**PLOTLY_LAYOUT, height=550)
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICTION MODEL
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Prediction Model":
    st.markdown('<div class="section-title">Train a Prediction Model</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    model_name = c1.selectbox("Algorithm", ["Random Forest", "Gradient Boosting", "Linear Regression"])
    test_size = c2.slider("Test Size %", 10, 40, 20) / 100
    n_estimators = c3.slider("n_estimators (tree models)", 50, 300, 100)

    feature_cols = [c for c in df_enc.columns if c != target and c != 'Score_Category']
    X = df_enc[feature_cols]
    y = df_enc[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if st.button("🚀 Train Model", use_container_width=True):
        with st.spinner("Training..."):
            if model_name == "Random Forest":
                model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            elif model_name == "Gradient Boosting":
                model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
            else:
                model = LinearRegression()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

        col1, col2, col3 = st.columns(3)
        for col, (val, label) in zip([col1, col2, col3], [
            (f"{r2:.3f}", "R² Score"), (f"{rmse:.2f}", "RMSE"), (f"{mae:.2f}", "MAE")
        ]):
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

        # Actual vs Predicted
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(x=y_test, y=y_pred, opacity=0.6,
                             color_discrete_sequence=['#60a5fa'],
                             labels={'x': 'Actual', 'y': 'Predicted'},
                             title="Actual vs Predicted")
            fig.add_shape(type='line', x0=y_test.min(), y0=y_test.min(),
                          x1=y_test.max(), y1=y_test.max(),
                          line=dict(color='#f87171', dash='dash'))
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            residuals = y_test - y_pred
            fig = px.histogram(x=residuals, nbins=40,
                               color_discrete_sequence=['#a78bfa'],
                               labels={'x': 'Residual'},
                               title="Residual Distribution")
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
            imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': model.feature_importances_})
            imp_df = imp_df.sort_values('Importance', ascending=False).head(15)
            fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                         color='Importance', color_continuous_scale='Blues',
                         title="Top 15 Feature Importances")
            fig.update_layout(**PLOTLY_LAYOUT, height=450, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── Live Predictor ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🔮 Predict a Student\'s Score</div>', unsafe_allow_html=True)
    st.markdown("Fill in student details to get a predicted exam score:")

    with st.expander("⚙️ Student Input Form", expanded=True):
        input_data = {}
        cols = st.columns(3)
        for i, feat in enumerate(feature_cols):
            with cols[i % 3]:
                if feat in df.select_dtypes(include='object').columns:
                    opts = df[feat].unique().tolist()
                    val = st.selectbox(feat, opts, key=f"inp_{feat}")
                    le = LabelEncoder()
                    le.fit(df[feat].astype(str))
                    input_data[feat] = le.transform([val])[0]
                else:
                    mn, mx = float(df[feat].min()), float(df[feat].max())
                    med = float(df[feat].median())
                    input_data[feat] = st.slider(feat, mn, mx, med, key=f"inp_{feat}")

        if st.button("🎯 Predict Score", use_container_width=True):
            # Train quick RF for prediction
            quick_model = RandomForestRegressor(n_estimators=100, random_state=42)
            quick_model.fit(X_train, y_train)
            inp_arr = np.array([[input_data[f] for f in feature_cols]])
            pred_score = quick_model.predict(inp_arr)[0]
            grade = "A+" if pred_score >= 85 else "A" if pred_score >= 75 else "B" if pred_score >= 65 else "C"
            st.markdown(f"""
            <div class="metric-card" style="margin-top:16px;">
                <div class="metric-value">{pred_score:.1f}</div>
                <div class="metric-label">Predicted Exam Score &nbsp;|&nbsp; Grade: {grade}</div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 5 — RAW DATA
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📋 Raw Data":
    st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])
    search = c1.text_input("🔍 Search (any column)")
    n_rows = c2.number_input("Rows to show", 10, len(df), 50)

    display_df = df.copy()
    if search:
        mask = display_df.apply(lambda col: col.astype(str).str.contains(search, case=False)).any(axis=1)
        display_df = display_df[mask]

    st.dataframe(display_df.head(n_rows), use_container_width=True)
    st.markdown(f"<small style='color:#4a5a80'>Showing {min(n_rows, len(display_df))} of {len(display_df)} rows</small>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Dataset Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown('<div class="section-title">Missing Values</div>', unsafe_allow_html=True)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        st.success("✅ No missing values found in the dataset!")
    else:
        fig = px.bar(x=missing.index, y=missing.values, color_discrete_sequence=['#f59e0b'],
                     labels={'x': 'Column', 'y': 'Missing Count'}, title="Missing Values per Column")
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
