import glob
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from backend_logic import ExecutiveSummaryEngine, AadhaarAnalyticsEngine
from data_preprocessing import preprocess_dataframe
import trends_analytics

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Aadhaar Analytics Command Center",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Mode Styling
st.markdown("""
<style>
    /* Dark Mode Styling */
    :root {
        --bg-primary: #0e1117;
        --bg-secondary: #1a1d29;
        --text-primary: #fafafa;
        --text-secondary: #a9b1c0;
        --accent-color: #4a9eff;
        --border-color: #2d3139;
        --card-bg: #1a1d29;
        --hover-bg: #252830;
    }
    
    .stApp {
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 600;
    }
    
    h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .stMetric {
        background-color: var(--card-bg);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        border-color: var(--accent-color);
        box-shadow: 0 4px 12px rgba(74, 158, 255, 0.2);
        transform: translateY(-2px);
    }
    
    [data-testid="stMarkdownContainer"] {
        color: var(--text-secondary);
    }
    
    .stDataFrame {
        border: 1px solid var(--border-color);
        border-radius: 8px;
        overflow: hidden;
    }
    
    .stButton > button {
        background-color: var(--accent-color);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #3d8ce8;
        box-shadow: 0 4px 12px rgba(74, 158, 255, 0.3);
    }
    
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(to right, transparent, var(--border-color), transparent);
        margin: 2rem 0;
    }
    
    .css-1d391kg, .css-1v0mbdj {
        background-color: var(--card-bg);
        border-radius: 8px;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
    }
    
    /* Radio Button Styling */
    .stRadio > label {
        color: var(--text-primary);
    }
    
    /* Popover Button Fix - Ensure icons display horizontally */
    [data-testid="stPopover"] button {
        min-width: 40px !important;
        padding: 0.5rem !important;
        text-align: center !important;
        white-space: nowrap !important;
        writing-mode: horizontal-tb !important;
    }
    
    [data-testid="stPopover"] button p {
        display: inline !important;
        writing-mode: horizontal-tb !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING (CRITICAL STEP)
# ==========================================
# ==========================================
# 2. DATA LOADING (FIXED)
# ==========================================
@st.cache_data
def load_data():
    
    def read_all_csvs_in_folder(folder_name):
        """
        Finds all .csv files in a folder and merges them into one DataFrame.
        """
        # Get list of all CSV files in the folder
        file_paths = glob.glob(os.path.join(folder_name, "*.csv"))
        
        if not file_paths:
            return pd.DataFrame() # Return empty if no files
        
        # Read and combine them
        df_list = []
        for file in file_paths:
            try:
                # Read CSV without parsing dates here (preprocessing will handle it)
                temp_df = pd.read_csv(file, low_memory=False)
                df_list.append(temp_df)
            except Exception as e:
                print(f"Skipping bad file: {file}") # Print to console, not Streamlit
        
        if df_list:
            full_df = pd.concat(df_list, ignore_index=True)
            return full_df
        else:
            return pd.DataFrame()

    # Load data without any UI elements inside
    df_bio = read_all_csvs_in_folder("api_data_aadhar_biometric")
    df_demo = read_all_csvs_in_folder("api_data_aadhar_demographic")
    df_enrol = read_all_csvs_in_folder("api_data_aadhar_enrolment")
    
    # Apply preprocessing to clean and normalize data
    if not df_bio.empty:
        df_bio = preprocess_dataframe(df_bio, "Biometric")
    if not df_demo.empty:
        df_demo = preprocess_dataframe(df_demo, "Demographic")
    if not df_enrol.empty:
        df_enrol = preprocess_dataframe(df_enrol, "Enrollment")
    
    return df_enrol, df_bio, df_demo

# ==========================================
# CALLING THE FUNCTION & PRE-GENERATE ALL CHARTS
# ==========================================
@st.cache_data
def preload_all_charts(_df_enrol, _df_bio, _df_demo):
    """Pre-generate all visualizations"""
    charts = {}
    
    print("Pre-generating charts...")
    
    # Data for Streamlit native charts
    charts['forecast_data'] = trends_analytics.get_forecast_data(_df_enrol, days_forward=30)
    print("  ✓ Forecast data")
    
    charts['dow_data'] = trends_analytics.get_dow_data(_df_enrol)
    print("  ✓ Day-of-week data")
    
    charts['growth_data'] = trends_analytics.get_growth_data(_df_enrol, top_n=10)
    print("  ✓ Growth trajectory data")
    
    # Plotly charts for complex visualizations
    charts['seasonal_radar'] = trends_analytics.create_seasonal_radar(_df_enrol)
    print("  ✓ Seasonal radar")
    
    charts['network_graph'] = trends_analytics.create_network_graph(_df_enrol, top_n=25)
    print("  ✓ Network graph")
    
    print("All charts ready!")
    return charts

try:
    # Create progress bar
    progress_text = "Loading data and pre-generating visualizations..."
    progress_bar = st.progress(0, text=progress_text)
    
    # Load data (20% progress)
    progress_bar.progress(20, text="Loading CSV files...")
    df_enrol, df_bio, df_demo = load_data()
    
    # Initialize engines (40% progress)
    progress_bar.progress(40, text="Initializing analytics engines...")
    exec_engine = ExecutiveSummaryEngine(df_enrol, df_bio, df_demo)
    adv_engine = AadhaarAnalyticsEngine(df_enrol, df_bio, df_demo)
    
    # Pre-generate all charts (60-100% progress)
    progress_bar.progress(60, text="Pre-generating all visualizations...")
    preloaded_charts = preload_all_charts(df_enrol, df_bio, df_demo)
    
    progress_bar.progress(100, text="Complete!")
    progress_bar.empty()  # Remove progress bar
    
    st.toast(f"✅ Ready! {len(df_enrol):,} enrollments loaded", icon="🎉")

except Exception as e:
    st.error(f"Critical Error loading data: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("UIDAI Analytics Dashboard")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/c/cf/Aadhaar_Logo.svg", width=150)

st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate to Module:", [
    "Executive Summary",
    "Operations & Logistics",
    "Trends & Forecasting",
    "Demographics & Policy",
    "Security & Integrity"
])

st.sidebar.markdown("---")
st.sidebar.info("System Status: **Online**")

# ==========================================
# 4. PAGE LOGIC
# ==========================================

# --- PAGE 1: EXECUTIVE SUMMARY ---
if page == "Executive Summary":
    st.title("Executive Command Center")
    st.markdown("**High-level overview of ecosystem health and critical alerts**")
    
    # ---------------------------------------------------------
    # SECTION 1: CRITICAL ALERTS (Metrics + Popups)
    # ---------------------------------------------------------
    st.subheader("Critical Alerts")
    
    # Calculate Data
    early_warning = exec_engine.get_early_warning_system()
    stagnation = exec_engine.get_stagnation_detection()
    age_cols = [col for col in df_enrol.columns if col.startswith('age_')]
    total_vol = df_enrol[age_cols].sum().sum()
    
    # Display Big Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Critical Decline Districts", early_warning['metric_value'], delta="-5%")
        
        # THE "VIEW ALL" POPUP BUTTON
        with st.popover("View All Declining Districts"):
            st.markdown("### Full List of Critical Districts")
            st.write("Districts with >20% drop in enrollment volume.")
            # displaying the dataframe makes it scrollable automatically
            st.dataframe(
                early_warning['details_df'][['change_pct']].style.format("{:.1f}%"),
                use_container_width=True,
                height=400  # Fixed height makes it scrollable
            )

    with col2:
        st.metric("Stagnant Pincodes (30 Days)", stagnation['total_stagnant'], delta_color="inverse")
        
        # THE "VIEW ALL" POPUP BUTTON
        with st.popover("View All Stagnant Pincodes"):
            st.markdown("### Full List of Inactive Pincodes")
            st.write("Pincodes with zero activity in the last 30 days.")
            if stagnation['pincode_list']:
                st.dataframe(
                    pd.DataFrame(stagnation['pincode_list'], columns=["Inactive Pincode"]),
                    use_container_width=True,
                    height=400 # Scrollable
                )
            else:
                st.success("No stagnant pincodes found.")

    with col3:
        st.metric("Total Enrollment Volume", f"{total_vol:,.0f}")

    st.markdown("---")
    
    # ---------------------------------------------------------
    # SECTION 2: LEADERBOARDS (Keeping the Top 5 Tables)
    # ---------------------------------------------------------
    st.subheader("Performance Leaderboard")
    benchmarks = exec_engine.get_peer_benchmarking()
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Top 5 Performing Districts")
        st.dataframe(benchmarks['top_performers'], hide_index=True, use_container_width=True)
    with c2:
        st.markdown("### Bottom 5 Districts (Needs Support)")
        st.dataframe(benchmarks['bottom_performers'], hide_index=True, use_container_width=True)

    # ---------------------------------------------------------
    # SECTION 3: MAP (Your India Map)
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("Location Archetypes (Cluster Map)")
    
    clusters = exec_engine.get_location_clusters()
    
    if not clusters.empty:
        # Load District GeoJSON
        district_geojson_url = "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson"
        
        fig_cluster_map = px.choropleth(
            clusters,
            geojson=district_geojson_url,
            locations='district',          
            featureidkey='properties.NAME_2',
            color='cluster_label',
            color_discrete_map={
                'Cluster 0': '#1f77b4',
                'Cluster 1': '#ff7f0e',
                'Cluster 2': '#2ca02c',
                'Cluster 3': '#d62728'
            },
            hover_name='district',
            hover_data=['state', 'total_volume'],
            title="District-Level Market Archetypes",
            projection="mercator"
        )
        
        # Zoom to India
        fig_cluster_map.update_geos(fitbounds="locations", visible=False)
        fig_cluster_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
        
        st.plotly_chart(fig_cluster_map, use_container_width=True)
    else:
        st.warning("Not enough data to generate clusters.")

# --- PAGE 2: OPERATIONS ---
elif page == "Operations & Logistics":
    st.title("Operations & Resource Planning")
    st.markdown("**Comprehensive operational intelligence for infrastructure allocation and capacity optimization**")
    
    # VISUALIZATION 1: BCG Service Strain Matrix
    st.markdown("---")
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        st.subheader("Service Strain Matrix: Kit Allocation vs Staff Planning")
    with col2:
        with st.popover("ℹ️"):
            st.markdown("""
            **What This Shows:**
            - **X-axis:** New enrollment demand (kit requirements)
            - **Y-axis:** Update request load (staff requirements)
            - **Quadrants guide resource allocation decisions**
            
            **Action Items:**
            - **Top-Right (Warzone):** Need both kits + staff
            - **Top-Left (Hub):** Maintenance-heavy, allocate experienced staff
            - **Bottom-Right (Nursery):** Growth areas, deploy kits & trainers
            - **Bottom-Left (Dormant):** Low priority, monitor only
            """)
    
    bcg_data = adv_engine.get_bcg_matrix_data()
    if not bcg_data.empty:
        fig_bcg = px.scatter(
            bcg_data, 
            x='total_enrollments', 
            y='total_updates',
            text='district',
            labels={'total_enrollments': 'New Enrollment Demand', 'total_updates': 'Update Request Load'},
            template="plotly_white",
            height=600
        )
        fig_bcg.update_traces(textposition='top center', textfont_size=8)
        fig_bcg.add_hline(y=bcg_data['total_updates'].median(), line_dash="dot", 
                         annotation_text="High Staff Needed", line_color="red")
        fig_bcg.add_vline(x=bcg_data['total_enrollments'].median(), line_dash="dot", 
                         annotation_text="High Kits Needed", line_color="blue")
        st.plotly_chart(fig_bcg, use_container_width=True)
    else:
        st.warning("⚠️ Insufficient data for BCG matrix")
    
    # VISUALIZATION 2: Mobile Van Priority Solver
    st.markdown("---")
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        st.subheader("Mobile Van Solver: Recommended Deployment Spots")
    with col2:
        with st.popover("ℹ️"):
            st.markdown("""
            **What This Shows:**
            - **Red points:** High-priority areas needing mobile van service
            - **Blue points:** Areas adequately served by fixed centers
            - **Size:** Total activity volume
            
            **Action Items:**
            - Deploy mobile vans to **red-flagged pincodes**
            - These areas have high update demand but limited fixed infrastructure
            - Prioritize by bubble size (larger = more urgent)
            """)
    
    van_data = adv_engine.get_mobile_van_priority_data()
    if not van_data.empty:
        fig_van = px.scatter(
            van_data, 
            x='district', 
            y='total_updates',
            color='van_priority',
            size='total_activity',
            hover_data=['pincode'],
            labels={'total_updates': 'Update Requests', 'district': 'District'},
            color_discrete_map={True: 'red', False: 'royalblue'},
            template="plotly_white",
            height=600
        )
        fig_van.update_xaxes(tickangle=45)
        st.plotly_chart(fig_van, use_container_width=True)
    else:
        st.warning("No mobile van priority data available")
    
    # VISUALIZATION 3: Center Productivity Rankings
    st.markdown("---")
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        st.subheader("Top 20 Center Productivity (Activity Volume)")
    with col2:
        with st.popover("ℹ️"):
            st.markdown("""
            **What This Shows:**
            - Highest-performing centers by total requests handled
            - Colored by district for regional patterns
            
            **Action Items:**
            - Study **best practices** from top centers
            - Allocate additional resources to maintain performance
            - Identify if multiple top centers cluster in same district
            """)
    
    prod_data = adv_engine.get_center_productivity_data(top_n=20)
    if not prod_data.empty:
        prod_data['pincode_label'] = prod_data['pincode'].astype(str)
        fig_prod = px.bar(
            prod_data, 
            x='pincode_label', 
            y='total_activity',
            color='district',
            labels={'total_activity': 'Total Requests Handled', 'pincode_label': 'Pincode/Center ID'},
            template="plotly_white",
            height=500
        )
        fig_prod.update_xaxes(tickangle=45)
        st.plotly_chart(fig_prod, use_container_width=True)
    else:
        st.warning("No center productivity data available")
    
    # VISUALIZATION 4: Weekly Capacity Utilization Heatmap
    st.markdown("---")
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        st.subheader("Weekly Capacity Utilization: Load Distribution")
    with col2:
        with st.popover("ℹ️"):
            st.markdown("""
            **What This Shows:**
            - Average activity levels across days of the week
            - Darker colors = higher load
            
            **Action Items:**
            - **Peak days:** Increase staffing
            - **Low days:** Schedule maintenance, training
            - Optimize shift schedules to match demand curve
            """)
    
    weekly_data = adv_engine.get_weekly_capacity_data()
    if not weekly_data.empty:
        fig_heat = px.imshow(
            [weekly_data['total_activity'].values],
            x=weekly_data['day_of_week'].values,
            y=['Avg Activity'],
            color_continuous_scale='Viridis',
            labels={'color': 'Activity Level'},
            template="plotly_white",
            height=300
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.warning("No weekly capacity data available")
    
    # VISUALIZATION 5: Growth Velocity Tracking
    st.markdown("---")
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        st.subheader("Enrollment Growth Velocity: Top 5 Districts")
    with col2:
        with st.popover("ℹ️"):
            st.markdown("""
            **What This Shows:**
            - Daily enrollment trends for highest-volume districts
            - Shows momentum and growth acceleration
            
            **Action Items:**
            - **Steep slopes:** Fast-growing districts need urgent resource scaling
            - **Plateaus:** Investigate saturation or operational bottlenecks
            - **Dips:** Identify and address sudden drops immediately
            """)
    
    velocity_data = adv_engine.get_growth_velocity_data(top_n=5)
    if not velocity_data.empty:
        fig_vel = px.line(
            velocity_data, 
            x='date', 
            y='total_enrollments',
            color='district',
            labels={'total_enrollments': 'Enrollment Volume', 'date': 'Date'},
            template="plotly_white",
            height=500
        )
        fig_vel.update_traces(line_shape='spline')
        st.plotly_chart(fig_vel, use_container_width=True)
    else:
        st.warning("⚠️ No growth velocity data available")


# --- PAGE 3: TRENDS ---
elif page == "Trends & Forecasting":
    st.title("Trends & Predictive Analytics")
    st.markdown(f"**Analyzing {len(df_enrol):,} enrollment records across {df_enrol['district'].nunique()} districts**")
    
    # 30-Day Enrollment Forecast
    st.markdown("---")
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        st.subheader("30-Day Enrollment Forecast")
    with col2:
        with st.popover("ℹ️"):
            st.markdown("""
            **What This Shows:**
            - **Blue Line (Historical):** Actual enrollment data from past records, showing real trends and patterns
            - **Orange Line (Forecast):** Prophet model predictions for the next 30 days based on historical patterns
            
            **Key Insights:**
            - The model detects **weekly seasonality** (enrollment patterns repeat each week)
            - **Yearly trends** capture seasonal variations (academic cycles, festivals, policy changes)
            - **Confidence intervals** (shaded area) show prediction uncertainty - wider bands = less certainty
            
            **Action Items:**
            - **Sharp drops** in forecast → Plan staff reduction or investigate causes
            - **Steep increases** → Allocate additional resources, prepare for capacity expansion
            - **Consistent patterns** → Optimize staffing schedules to match predicted demand
            - **Irregular spikes** → Investigate for policy impacts or external events
            """)
    
    forecast_data = preloaded_charts.get('forecast_data', pd.DataFrame())
    
    if not forecast_data.empty:
        # Create Plotly chart with interactive legend
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=forecast_data.index,
            y=forecast_data['Historical'],
            mode='lines',
            name='Historical',
            line=dict(color='#1f77b4', width=2)
        ))
        fig_forecast.add_trace(go.Scatter(
            x=forecast_data.index,
            y=forecast_data['Forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='#ff7f0e', width=2)
        ))
        fig_forecast.update_layout(
            height=500,
            xaxis_title='Date',
            yaxis_title='Enrollment Count',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.warning("⚠️ Insufficient historical data for forecasting. Need at least 2 days of enrollment records.")
    
    # Seasonal Trends
    st.markdown("---")
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        st.subheader("Seasonal Enrollment Patterns: North vs South India")
    with col2:
        with st.popover("ℹ️"):
            st.markdown("""
            **What This Shows:**
            - Monthly enrollment volumes plotted as a **radar chart** to reveal seasonal cycles
            - **North vs South comparison** highlights regional behavioral differences
            
            **Key Insights:**
            - **Peak months** indicate optimal times for enrollment drives or awareness campaigns
            - **Low months** may correlate with agricultural seasons, festivals, or extreme weather
            - **Regional gaps** show where targeted interventions are needed
            - **Symmetry** suggests predictable patterns; **asymmetry** indicates external factors
            
            **Action Items:**
            - Schedule mobile van drives during **high-enrollment months** for maximum impact
            - Plan maintenance and training during **low-activity periods**
            - Investigate **regional disparities** - are Southern states facing unique barriers?
            - Align policy announcements with **peak engagement months**
            """)
    
    st.plotly_chart(preloaded_charts['seasonal_radar'], use_container_width=True)
    
    # Day-of-Week Patterns
    st.markdown("---")
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        st.subheader("Weekly Enrollment Activity: Rural vs Urban Areas")
    with col2:
        with st.popover("ℹ️"):
            st.markdown("""
            **What This Shows:**
            - Daily enrollment patterns across the week for **Rural (Green)** vs **Urban (Red)** areas
            - Reveals behavioral differences between demographics
            
            **Key Insights:**
            - **Weekday peaks** in urban areas → Office workers enrolling during lunch breaks
            - **Weekend spikes** in rural areas → Farmers/laborers free on weekends
            - **Monday dips** (if present) → Post-weekend administrative delays
            - **Friday surges** → "Get it done before weekend" mentality
            
            **Action Items:**
            - **Urban centers:** Extend operating hours on weekdays, add lunch-hour slots
            - **Rural areas:** Boost Saturday/Sunday staffing, deploy mobile vans on weekends
            - **Minimize wait times** on high-volume days to prevent drop-offs
            - **Optimize staff rosters** to match daily demand curves
            """)
    
    dow_data = preloaded_charts.get('dow_data', pd.DataFrame())
    if not dow_data.empty:
        st.bar_chart(dow_data, height=500, color=["#2ca02c", "#d62728"])
    else:
        st.warning("⚠️ Day-of-week data not available")
    
    # Growth Trajectories
    st.markdown("---")
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        st.subheader("Cumulative Growth Trajectories: Top Performing Districts")
    with col2:
        with st.popover("ℹ️"):
            st.markdown("""
            **What This Shows:**
            - **Cumulative enrollment growth** over time for highest-volume districts
            - Steeper slopes = faster growth rates; flat lines = stagnation
            
            **Key Insights:**
            - **Leading districts** set benchmarks - study their success factors
            - **Plateaus** indicate market saturation or policy barriers
            - **Diverging lines** show widening inequality - some districts left behind
            - **Parallel lines** suggest similar external factors (state-level policies)
            
            **Action Items:**
            - **Replicate best practices** from top-performing districts
            - **Investigate lagging districts** - infrastructure gaps? Awareness issues?
            - **Benchmark targets** against top-quartile performers
            - **Monitor inflection points** - sudden slope changes indicate events worth investigating
            """)
    
    growth_data = preloaded_charts.get('growth_data', pd.DataFrame())
    if not growth_data.empty:
        # Create Plotly chart with interactive legend
        fig_growth = go.Figure()
        for district in growth_data.columns:
            fig_growth.add_trace(go.Scatter(
                x=growth_data.index,
                y=growth_data[district],
                mode='lines',
                name=district,
                line=dict(width=2)
            ))
        fig_growth.update_layout(
            height=500,
            xaxis_title='Date',
            yaxis_title='Cumulative Enrollment',
            hovermode='x unified',
            legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.02)
        )
        st.plotly_chart(fig_growth, use_container_width=True)
    else:
        st.warning("Growth trajectory data not available")
    
    # Network Graph
    st.markdown("---")
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        st.subheader("District Correlation Network: Enrollment Pattern Relationships")
    with col2:
        with st.popover("ℹ️"):
            st.markdown("""
            **What This Shows:**
            - **Network nodes** = Districts with similar enrollment timing patterns
            - **Edges (connections)** = Strong correlation (>0.7) in enrollment trends
            - **Node size** = Number of connections (centrality in the network)
            
            **Key Insights:**
            - **Clusters** reveal districts influenced by shared factors (migration corridors, economic zones)
            - **Hub nodes** (large, many connections) are influence centers - policy changes here ripple outward
            - **Isolated nodes** operate independently - unique local factors dominate
            - **Connected pairs** may share borders, economic ties, or administrative practices
            
            **Action Items:**
            - **Pilot programs** in hub districts for maximum spillover effect
            - **Investigate clusters** - Are they sharing staff? Experiencing common events?
            - **Monitor synchronized dips** - Could indicate coordinated fraud or systemic issues
            - **Leverage connections** - Success in one district can be promoted to its network neighbors
            """)
    
    st.plotly_chart(preloaded_charts['network_graph'], use_container_width=True)



# --- PAGE 4: DEMOGRAPHICS ---
elif page == "Demographics & Policy":
    st.title("Demographics & Inclusion Policy")
    
    districts = df_enrol['district'].unique()
    selected_dist = st.selectbox("Select District", districts)
    
    col1, col2 = st.columns(2)
    
    # 4A: Age Distribution
    with col1:
        st.subheader("Age Cohort Distribution")
        age_stats = adv_engine.get_age_distribution(selected_dist)
        fig_donut = px.pie(names=age_stats.keys(), values=age_stats.values(), hole=0.4)
        st.plotly_chart(fig_donut, use_container_width=True)
        
    # 5C: Update Lag
    with col2:
        st.subheader("Engagement & Retention")
        lag_stats = adv_engine.get_update_lag_analysis(selected_dist)
        
        st.metric("Engagement Ratio", lag_stats['engagement_ratio'])
        st.progress(min(lag_stats['engagement_ratio'], 1.0))
        st.caption("Ratio of Updates to New Adult Enrollments. Higher is better.")


# --- PAGE 5: SECURITY ---
elif page == "Security & Integrity":
    st.title("Security & Forensic Audit")
    st.markdown("**Comprehensive fraud detection and data integrity analysis**")
    
    # VISUALIZATION 1: Global Benford's Law Analysis
    st.markdown("---")
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        st.subheader("Fraud Audit: Benford's Law Distribution")
    with col2:
        with st.popover("ℹ️"):
            st.markdown("""
            **What This Shows:**
            - **Benford's Law** states that naturally occurring numbers follow a predictable pattern for first digits
            - **Blue bars:** Your actual data distribution
            - **Red line:** Expected natural distribution
            
            **Key Insights:**
            - **Close match** = Organic, genuine data
            - **Significant deviation** = Potential fabrication, data entry errors, or fraud
            - Numbers starting with 1 should appear ~30% of the time naturally
            
            **Action Items:**
            - **High deviation score (>0.15):** Investigate data sources for manipulation
            - Audit districts showing unusual patterns
            - Review data entry processes if systematic deviations found
            """)
    
    benford_global = adv_engine.get_benfords_law_global()
    
    if "error" not in benford_global:
        deviation = benford_global['deviation_score']
        is_suspicious = benford_global['is_suspicious']
        
        if is_suspicious:
            st.error(f"⚠️ HIGH RISK DETECTED | Deviation Score: {deviation:.3f}")
            st.warning("The data shows significant deviation from Benford's Law. Recommend immediate audit.")
        else:
            st.success(f"✓ Normal Organic Behavior | Deviation Score: {deviation:.3f}")
        
        # Plot
        df_ben = benford_global['distribution']
        fig_ben = go.Figure()
        fig_ben.add_trace(go.Bar(
            x=df_ben['digit'], 
            y=df_ben['actual_freq'], 
            name='Your Data (Actual)', 
            marker_color='royalblue'
        ))
        fig_ben.add_trace(go.Scatter(
            x=df_ben['digit'], 
            y=df_ben['benford_freq'], 
            mode='lines+markers',
            name="Benford's Law (Theoretical)", 
            line=dict(color='red', width=3)
        ))
        fig_ben.update_layout(
            xaxis_title="Leading Digit",
            yaxis_title="Relative Frequency",
            template="plotly_white",
            height=500,
            xaxis=dict(tickmode='linear', tick0=1, dtick=1)
        )
        st.plotly_chart(fig_ben, use_container_width=True)
    else:
        st.warning("Insufficient data for Benford's Law analysis")
    
    # VISUALIZATION 2: Statistical Outliers (Time Series)
    st.markdown("---")
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        st.subheader("Statistical Outliers: Detecting Suspicious Activity Spikes")
    with col2:
        with st.popover("ℹ️"):
            st.markdown("""
            **What This Shows:**
            - Daily activity volumes over time
            - **Gray points:** Normal activity days
            - **Red points:** Anomalous spikes (>2.5 standard deviations above mean)
            
            **Key Insights:**
            - Sudden spikes may indicate:
              - Policy announcements causing rush
              - Data backlog processing
              - Potential fraud or data manipulation
            
            **Action Items:**
            - Investigate **red spikes** for root cause
            - Verify if spikes correspond to legitimate events
            - Review centers active during anomalous periods
            """)
    
    outliers_data = adv_engine.get_statistical_outliers()
    
    if not outliers_data.empty:
        anomaly_count = outliers_data['is_anomaly'].sum()
        st.metric("Suspicious Activity Days Detected", anomaly_count)
        
        fig_outliers = px.scatter(
            outliers_data, 
            x='date', 
            y='total_activity',
            color='is_anomaly',
            color_discrete_map={True: 'red', False: 'gray'},
            labels={'total_activity': 'Daily Volume', 'is_anomaly': 'Suspicious Spike'},
            template="plotly_white",
            height=500
        )
        
        # Add trend line
        fig_outliers.add_trace(go.Scatter(
            x=outliers_data['date'], 
            y=outliers_data['total_activity'],
            mode='lines',
            line=dict(color='lightgray', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add threshold line
        if 'threshold' in outliers_data.columns:
            fig_outliers.add_hline(
                y=outliers_data['threshold'].iloc[0],
                line_dash="dash",
                line_color="orange",
                annotation_text="Anomaly Threshold"
            )
        
        st.plotly_chart(fig_outliers, use_container_width=True)
    else:
        st.info("Insufficient time-series data for outlier detection")
    
    # VISUALIZATION 3: Volatility Analysis
    st.markdown("---")
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        st.subheader("Volatility Analysis: Top 20 Most Erratic Centers")
    with col2:
        with st.popover("ℹ️"):
            st.markdown("""
            **What This Shows:**
            - Centers ranked by inconsistency in activity levels
            - **Variance Score:** Coefficient of variation (higher = more erratic)
            
            **Key Insights:**
            - High volatility may indicate:
              - Seasonal/irregular operations
              - Inconsistent staffing
              - Potential gaming of quotas
              - Data quality issues
            
            **Action Items:**
            - Audit top-scoring centers for operational irregularities
            - Verify if volatility matches expected patterns (e.g., rural vs urban)
            - Implement monitoring for consistently erratic centers
            """)
    
    volatility_data = adv_engine.get_volatility_analysis(top_n=20)
    
    if not volatility_data.empty:
        fig_volatility = px.bar(
            volatility_data,
            x='pincode',
            y='variance_score',
            color='variance_score',
            color_continuous_scale='Reds',
            labels={'pincode': 'Pincode', 'variance_score': 'Erratic Behavior Score'},
            template="plotly_white",
            height=500
        )
        fig_volatility.update_xaxes(tickangle=45)
        st.plotly_chart(fig_volatility, use_container_width=True)
        
        # Display detailed table
        with st.expander("View Detailed Volatility Metrics"):
            st.dataframe(
                volatility_data[['pincode', 'mean', 'std', 'variance_score']].style.format({
                    'mean': '{:.1f}',
                    'std': '{:.1f}',
                    'variance_score': '{:.3f}'
                }),
                use_container_width=True
            )
    else:
        st.info("Insufficient data for volatility analysis")
    
    # VISUALIZATION 4: State Variance (Box Plot)
    st.markdown("---")
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        st.subheader("Within-State Variance: Analyzing Consistency Across State Borders")
    with col2:
        with st.popover("ℹ️"):
            st.markdown("""
            **What This Shows:**
            - Distribution of district-level activity within each state
            - **Box:** Interquartile range (25th-75th percentile)
            - **Outlier points:** Districts significantly different from state norm
            
            **Key Insights:**
            - **Wide boxes:** High internal variance within state
            - **Outliers:** Districts needing investigation
            - **Narrow boxes:** Consistent state-level operations
            
            **Action Items:**
            - Investigate outlier districts for unique factors
            - States with high variance may need standardized processes
            - Compare resource allocation across high/low variance states
            """)
    
    state_variance = adv_engine.get_state_variance_data()
    
    if not state_variance.empty and len(state_variance['state'].unique()) > 1:
        fig_variance = px.box(
            state_variance,
            x='state',
            y='total_activity',
            color='state',
            points="outliers",
            labels={'total_activity': 'Activity Level', 'state': 'State Name'},
            template="plotly_white",
            height=600
        )
        fig_variance.update_xaxes(tickangle=45)
        st.plotly_chart(fig_variance, use_container_width=True)
    else:
        st.info("Insufficient state-level data for variance analysis")