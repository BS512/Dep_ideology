import streamlit as st
import pandas as pd
import altair as alt

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Academic & Mortality Risk Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enable Altair large dataset handling if required
alt.data_transformers.disable_max_rows()

# -----------------------------------------------------------------------------
# DATA LOADING & PREPROCESSING
# -----------------------------------------------------------------------------
@st.cache_data
def load_datasets():
    # Load primary hazard ratio dataset
    try:
        df_hr = pd.read_csv("ModelPredictor_2.csv")
    except FileNotFoundError:
        # Fallback to alternative filename if running in different environment
        df_hr = pd.read_csv("ModelPredictor.csv")
        
    # Load primary student performance dataset
    try:
        df_acad = pd.read_csv("cleaned_academic_performance_3.csv")
    except FileNotFoundError:
        df_acad = pd.read_csv("cleaned_academic_performance.csv")
        
    return df_hr, df_acad

df_hr, df_acad = load_datasets()

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & DATASET SELECTION
# -----------------------------------------------------------------------------
st.sidebar.title("🛠️ Navigation & Settings")

active_tab = st.sidebar.radio(
    "Select Dashboard Mode:",
    [
        "1. Hazard Ratio & Mortality Risk Analysis",
        "2. Student Academic Performance Analysis",
        "3. Special Multi-Data Pattern Overlay"
    ]
)

st.sidebar.markdown("---")

# =============================================================================
# MODE 1: HAZARD RATIO & MORTALITY RISK ANALYSIS
# =============================================================================
if active_tab == "1. Hazard Ratio & Mortality Risk Analysis":
    st.header("📈 Model Estimates & Hazard Ratio Analysis")
    st.caption("Data source: `ModelPredictor_2.csv`")
    
    # Filter Controls
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        categories = st.multiselect(
            "Filter Model Category:",
            options=df_hr["Model Category"].unique(),
            default=df_hr["Model Category"].unique()
        )
    with col_f2:
        genders = st.multiselect(
            "Filter Gender:",
            options=df_hr["Gender"].unique(),
            default=df_hr["Gender"].unique()
        )

    # Filtered DataFrame
    filtered_hr = df_hr[
        (df_hr["Model Category"].isin(categories)) & 
        (df_hr["Gender"].isin(genders))
    ]

    st.markdown("---")
    
    # Layer 1 & Layer 2 Menu Selection
    st.subheader("🔍 Layer 1 & 2: Component Comparison & View Toggles")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        x_col = st.selectbox("X-Axis Variable:", ["Quartile / Level", "Predictor", "Model Category"], index=0)
    with c2:
        y_col = st.selectbox("Y-Axis Variable (Metric):", ["HR", "CI_Lower", "CI_Upper"], index=0)
    with c3:
        chart_view = st.radio(
            "Select Display Form (3+ Options):",
            ["Bar Chart View", "Point & Error Bar View", "Heatmap Matrix View", "Raw Data Summary Table"],
            horizontal=False
        )

    st.markdown("### Visual Output")
    
    if filtered_hr.empty:
        st.warning("No data available for the selected filters.")
    else:
        # Form 1: Bar Chart
        if chart_view == "Bar Chart View":
            chart = alt.Chart(filtered_hr).mark_bar(opacity=0.85).encode(
                x=alt.X(f"{x_col}:N", title=x_col, sort=None),
                y=alt.Y(f"{y_col}:Q", title=f"Value ({y_col})"),
                color=alt.Color("Gender:N", scale=alt.Scale(scheme="category10")),
                column=alt.Column("Model Category:N", title="Model Category"),
                tooltip=["Model Category", "Predictor", "Quartile / Level", "Gender", "HR", "CI_Lower", "CI_Upper", "Notes"]
            ).properties(width=160, height=300).interactive()
            st.altair_chart(chart, use_container_width=True)

        # Form 2: Point Plot + Error Bars
        elif chart_view == "Point & Error Bar View":
            base = alt.Chart(filtered_hr).encode(
                x=alt.X(f"{x_col}:N", title=x_col),
                color=alt.Color("Gender:N", scale=alt.Scale(scheme="set1"))
            )
            points = base.mark_point(size=80, filled=True).encode(
                y=alt.Y("HR:Q", title="Hazard Ratio (95% CI)"),
                tooltip=["Model Category", "Predictor", "Quartile / Level", "Gender", "HR", "CI_Lower", "CI_Upper"]
            )
            error_bars = base.mark_errorbar().encode(
                y=alt.Y("CI_Lower:Q", title="CI Lower"),
                y2=alt.Y2("CI_Upper:Q")
            )
            ref_line = alt.Chart(pd.DataFrame({'y': [1.0]})).mark_rule(color='red', strokeDash=[4, 4]).encode(y='y:Q')
            
            combined_chart = (ref_line + error_bars + points).properties(
                height=400, 
                title="Hazard Ratios with 95% Confidence Bounds (Red dashed line = Baseline HR 1.0)"
            ).interactive()
            
            st.altair_chart(combined_chart, use_container_width=True)

        # Form 3: Heatmap Matrix View
        elif chart_view == "Heatmap Matrix View":
            heatmap = alt.Chart(filtered_hr).mark_rect().encode(
                x=alt.X("Quartile / Level:N", title="Level / Quartile"),
                y=alt.Y("Model Category:N", title="Model Category"),
                color=alt.Color(f"{y_col}:Q", scale=alt.Scale(scheme="viridis"), title=y_col),
                facet=alt.Facet("Gender:N", columns=2),
                tooltip=["Model Category", "Predictor", "Quartile / Level", "Gender", y_col]
            ).properties(width=300, height=250)
            st.altair_chart(heatmap, use_container_width=True)

        # Form 4: Raw Summary Table
        else:
            st.dataframe(filtered_hr, use_container_width=True)

# =============================================================================
# MODE 2: STUDENT ACADEMIC PERFORMANCE ANALYSIS
# =============================================================================
elif active_tab == "2. Student Academic Performance Analysis":
    st.header("🎓 Student Cohort Academic Performance")
    st.caption("Data source: `cleaned_academic_performance_3.csv`")
    
    # Filters
    st.markdown("---")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        if "Target" in df_acad.columns:
            targets = st.multiselect("Filter Outcome Status (Target):", df_acad["Target"].unique(), default=df_acad["Target"].unique())
            filtered_acad = df_acad[df_acad["Target"].isin(targets)]
        else:
            filtered_acad = df_acad.copy()
            
    with col_f2:
        if "Gender" in df_acad.columns:
            genders_acad = st.multiselect("Filter Gender (0 = Female, 1 = Male):", df_acad["Gender"].unique(), default=df_acad["Gender"].unique())
            filtered_acad = filtered_acad[filtered_acad["Gender"].isin(genders_acad)]

    st.markdown("---")
    st.subheader("🔍 Layer 1 & 2: Component Comparison & View Toggles")
    
    num_cols = filtered_acad.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = filtered_acad.select_dtypes(include=["object", "category"]).columns.tolist()
    
    c1, c2, c3 = st.columns(3)
    with c1:
        acad_x = st.selectbox("X-Axis Variable:", num_cols + cat_cols, index=num_cols.index("Admission grade") if "Admission grade" in num_cols else 0)
    with c2:
        default_y_idx = num_cols.index("Curricular units 2nd sem (grade)") if "Curricular units 2nd sem (grade)" in num_cols else 1
        acad_y = st.selectbox("Y-Axis Variable:", num_cols, index=default_y_idx)
    with c3:
        acad_view = st.radio(
            "Select Display Form (3 Options):",
            ["Scatter / Trend View", "Aggregated Bar Chart", "Binned Heatmap Density"],
            horizontal=False
        )

    st.markdown("### Visual Output")
    
    # Form 1: Scatter / Trend View
    if acad_view == "Scatter / Trend View":
        sample_df = filtered_acad.sample(min(1000, len(filtered_acad)), random_state=42)
        scatter = alt.Chart(sample_df).mark_circle(size=45, opacity=0.6).encode(
            x=alt.X(f"{acad_x}:Q" if acad_x in num_cols else f"{acad_x}:N", title=acad_x),
            y=alt.Y(f"{acad_y}:Q", title=acad_y),
            color=alt.Color("Target:N" if "Target" in cat_cols else alt.value("steelblue")),
            tooltip=[acad_x, acad_y] + [c for c in ["Age at enrollment", "Target"] if c in df_acad.columns]
        )
        trend = scatter.transform_regression(acad_x, acad_y).mark_line(color="black", strokeWidth=2)
        st.altair_chart((scatter + trend).properties(height=450).interactive(), use_container_width=True)

    # Form 2: Aggregated Bar Chart
    elif acad_view == "Aggregated Bar Chart":
        bar = alt.Chart(filtered_acad).mark_bar().encode(
            x=alt.X(f"{acad_x}:N" if acad_x in cat_cols else f"{acad_x}:Q", bin=alt.Bin(maxbins=15) if acad_x in num_cols else None, title=acad_x),
            y=alt.Y(f"mean({acad_y}):Q", title=f"Average {acad_y}"),
            color=alt.Color("Target:N" if "Target" in cat_cols else alt.value("teal")),
            tooltip=[f"mean({acad_y}):Q"]
        ).properties(height=450).interactive()
        st.altair_chart(bar, use_container_width=True)

    # Form 3: Heatmap Density
    else:
        heatmap = alt.Chart(filtered_acad).mark_rect().encode(
            x=alt.X(f"{acad_x}:Q", bin=alt.Bin(maxbins=20), title=acad_x) if acad_x in num_cols else alt.X(f"{acad_x}:N", title=acad_x),
            y=alt.Y(f"{acad_y}:Q", bin=alt.Bin(maxbins=20), title=acad_y),
            color=alt.Color("count():Q", scale=alt.Scale(scheme="magma"), title="Count of Students"),
            tooltip=["count():Q"]
        ).properties(height=450).interactive()
        st.altair_chart(heatmap, use_container_width=True)

# =============================================================================
# MODE 3: SPECIAL MULTI-DATA PATTERN OVERLAY
# =============================================================================
else:
    st.header("⚡ Layer 3: Special Pattern Overlay Analysis")
    st.caption("Layer any custom assortment of data subsets or model categories over each other to discover hidden patterns.")

    st.markdown("""
    **How to use this layer:**
    Select multiple **Model Categories** from the Hazard Ratio dataset or **Predictor Groups**. 
    Altair will dynamically layer the statistical trajectories on top of one another on a shared coordinate space.
    """)

    st.markdown("---")

    overlay_categories = st.multiselect(
        "Select Model Categories to Overlay:",
        options=df_hr["Model Category"].unique(),
        default=df_hr["Model Category"].unique()[:3]
    )

    overlay_gender = st.radio(
        "Compare Across Gender:",
        options=["All", "Boys", "Girls"],
        horizontal=True
    )

    # Filter overlay dataset
    df_overlay = df_hr[df_hr["Model Category"].isin(overlay_categories)]
    if overlay_gender != "All":
        df_overlay = df_overlay[df_overlay["Gender"] == overlay_gender]

    if df_overlay.empty:
        st.info("Select at least one Model Category above to visualize the layered pattern.")
    else:
        # Multi-layer Altair Construction
        line_base = alt.Chart(df_overlay).encode(
            x=alt.X("Quartile / Level:N", title="Performance Level / Quartile", sort=None),
            y=alt.Y("HR:Q", title="Hazard Ratio (HR)"),
            color=alt.Color("Model Category:N", scale=alt.Scale(scheme="tableau10"), title="Layered Model Category"),
            strokeDash=alt.StrokeDash("Gender:N", title="Gender")
        )

        # Component 1: Connected trend lines across levels
        trend_lines = line_base.mark_line(point=True, strokeWidth=2.5)

        # Component 2: Confidence Interval Shading/Bands
        ci_bands = line_base.mark_area(opacity=0.15).encode(
            y=alt.Y("CI_Lower:Q"),
            y2=alt.Y2("CI_Upper:Q")
        )

        # Component 3: Reference Baseline (HR = 1.0)
        rule = alt.Chart(pd.DataFrame({'y': [1.0]})).mark_rule(
            color="red", 
            strokeDash=[4, 4], 
            strokeWidth=1.5
        ).encode(y='y:Q')

        # Layering components together
        composite_overlay = alt.layer(
            rule, ci_bands, trend_lines
        ).properties(
            height=500,
            title="Composite Multi-Model Trajectory Overlay"
        ).interactive()

        st.altair_chart(composite_overlay, use_container_width=True)

        st.markdown("### Pattern Insights Table")
        st.dataframe(
            df_overlay[["Model Category", "Predictor", "Quartile / Level", "Gender", "HR", "CI_Lower", "CI_Upper", "Notes"]],
            use_container_width=True
        )
