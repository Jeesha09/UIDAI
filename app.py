import glob
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from backend_logic import ExecutiveSummaryEngine, AadhaarAnalyticsEngine

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Aadhaar Analytics Command Center",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border-left: 5px solid #ff4b4b;
    }
    .stMetric {
        text-align: center;
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
                # Read CSV
                temp_df = pd.read_csv(file)
                df_list.append(temp_df)
            except Exception as e:
                print(f"Skipping bad file: {file}") # Print to console, not Streamlit
        
        if df_list:
            full_df = pd.concat(df_list, ignore_index=True)
            # Ensure date column is correct
            if 'date' in full_df.columns:
                full_df['date'] = pd.to_datetime(full_df['date'], format='%Y-%m-%d', errors='coerce')
            return full_df
        else:
            return pd.DataFrame()

    # Load data without any UI elements inside
    df_bio = read_all_csvs_in_folder("api_data_aadhar_biometric")
    df_demo = read_all_csvs_in_folder("api_data_aadhar_demographic")
    df_enrol = read_all_csvs_in_folder("api_data_aadhar_enrolment")
    
    return df_enrol, df_bio, df_demo

# ==========================================
# CALLING THE FUNCTION (The UI part goes here)
# ==========================================
try:
    with st.spinner('Loading millions of rows... This might take a minute...'):
        df_enrol, df_bio, df_demo = load_data()
        
    # Show success message only after loading is done
    st.toast(f"Data Loaded: {len(df_enrol)} Enrolments", icon="✅")

except Exception as e:
    st.error(f"Critical Error loading data: {e}")
    st.stop()

# Load the data into variables
df_enrol, df_bio, df_demo = load_data()

# Initialize the Backend Engines
exec_engine = ExecutiveSummaryEngine(df_enrol, df_bio, df_demo)
adv_engine = AadhaarAnalyticsEngine(df_enrol, df_bio, df_demo)

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("UIDAI Analytics")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/c/cf/Aadhaar_Logo.svg", width=150)

page = st.sidebar.radio("Navigate Module:", [
    "🏠 Executive Summary",
    "🚚 Operations & Logistics",
    "📈 Trends & Forecasting",
    "👥 Demographics & Policy",
    "🛡️ Security & Integrity"
])

st.sidebar.markdown("---")
st.sidebar.info("System Status: **Online** 🟢")

# ==========================================
# 4. PAGE LOGIC
# ==========================================

# --- PAGE 1: EXECUTIVE SUMMARY ---
if page == "🏠 Executive Summary":
    st.title("🏠 Executive Command Center")
    st.markdown("High-level overview of ecosystem health and critical alerts.")
    
    # ---------------------------------------------------------
    # SECTION 1: CRITICAL ALERTS (Metrics + Popups)
    # ---------------------------------------------------------
    st.subheader("🚨 Critical Alerts")
    
    # Calculate Data
    early_warning = exec_engine.get_early_warning_system()
    stagnation = exec_engine.get_stagnation_detection()
    total_vol = df_enrol['age_0_5'].sum() + df_enrol['age_18_greater'].sum()
    
    # Display Big Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Critical Decline Districts", early_warning['metric_value'], delta="-5%")
        
        # THE "VIEW ALL" POPUP BUTTON
        with st.popover("📉 View All Declining Districts"):
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
        with st.popover("🛑 View All Stagnant Pincodes"):
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
    st.subheader("🏆 Performance Leaderboard")
    benchmarks = exec_engine.get_peer_benchmarking()
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ✅ Top 5 Performing Districts")
        st.dataframe(benchmarks['top_performers'], hide_index=True, use_container_width=True)
    with c2:
        st.markdown("### ⚠️ Bottom 5 Districts (Needs Support)")
        st.dataframe(benchmarks['bottom_performers'], hide_index=True, use_container_width=True)

    # ---------------------------------------------------------
    # SECTION 3: MAP (Your India Map)
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📍 Location Archetypes (Cluster Map)")
    
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
elif page == "🚚 Operations & Logistics":
    st.title("🚚 Operations & Resource Planning")
    
    # Select District
    districts = df_enrol['district'].unique()
    selected_dist = st.selectbox("Select District for Analysis", districts)
    
    # Mine #2: Mobile Van Solver
    st.subheader(f"📍 Recommended Mobile Van Location for {selected_dist}")
    cog_data = adv_engine.get_center_of_gravity(selected_dist)
    
    if cog_data:
        optimal = cog_data['optimal_location']
        col1, col2 = st.columns([1, 3])
        with col1:
            st.info(f"**Optimal Coordinates:**\n\nLat: {optimal[0]:.4f}\nLong: {optimal[1]:.4f}")
            st.markdown("*Deploy van here to minimize citizen travel time.*")
        with col2:
            # Simple map visualization
            map_data = pd.DataFrame({
                'lat': [optimal[0]], 
                'lon': [optimal[1]], 
                'type': ['Recommended New Center']
            })
            st.map(map_data)
    
    st.markdown("---")
    
    # Mine #3: Service Strain Matrix
    st.subheader("🏗️ Service Strain Matrix (Growth vs Load)")
    matrix_df = adv_engine.get_service_strain_matrix()
    
    fig_matrix = px.scatter(
        matrix_df, 
        x="new_demand", 
        y="maintenance_load", 
        color="category",
        hover_name="district",
        title="Infrastructure Planning Matrix",
        labels={"new_demand": "New Enrollments (Growth)", "maintenance_load": "Update Requests (Churn)"}
    )
    # Add quadrants lines
    fig_matrix.add_vline(x=matrix_df['new_demand'].median(), line_dash="dash", line_color="gray")
    fig_matrix.add_hline(y=matrix_df['maintenance_load'].median(), line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig_matrix, use_container_width=True)


# --- PAGE 3: TRENDS ---
elif page == "📈 Trends & Forecasting":
    st.title("📈 Trends & Predictive Analytics")
    
    districts = df_enrol['district'].unique()
    selected_dist = st.selectbox("Select District to Forecast", districts)
    
    # 7A: Prophet Forecast
    st.subheader("🔮 30-Day Volume Forecast")
    
    forecast_df = adv_engine.get_forecast_prophet(selected_dist)
    
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Prediction'))
    fig_forecast.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
    fig_forecast.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', name='Confidence Interval'))
    
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Mine #4: Ripple Effect
    st.markdown("---")
    st.subheader("🌊 Ripple Effect Analysis")
    st.write("Does activity in Mumbai trigger activity in Bengaluru?")
    
    ripple = adv_engine.get_ripple_effect('Mumbai', 'Bengaluru')
    
    if ripple.get('is_significant'):
        st.success(f"✅ Significant Causal Link Detected! (P-Value: {ripple['p_value']:.4f})")
        st.markdown("**Insight:** Spikes in Mumbai reliably predict spikes in Bengaluru 3 days later.")
    else:
        st.warning("No significant causal link detected between these districts.")


# --- PAGE 4: DEMOGRAPHICS ---
elif page == "👥 Demographics & Policy":
    st.title("👥 Demographics & Inclusion Policy")
    
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
elif page == "🛡️ Security & Integrity":
    st.title("🛡️ Security & Forensic Audit")
    
    # Mine #1: Benford's Law
    st.subheader("🕵️ Benford's Law Fraud Detection")
    st.markdown("Analyzing 'Adult Enrollment' digits to detect synthetic/fake data entry.")
    
    districts = df_enrol['district'].unique()
    target_dist = st.selectbox("Select District for Audit", districts)
    
    benford_res = adv_engine.check_benfords_law(target_dist)
    
    if "error" in benford_res:
        st.error(benford_res['error'])
    else:
        score = benford_res['fraud_score']
        
        if score > 0.1:
            st.error(f"⚠️ HIGH RISK DETECTED (Score: {score:.2f})")
        else:
            st.success(f"✅ Normal Organic Behavior (Score: {score:.2f})")
            
        # Plot
        df_ben = benford_res['distribution']
        fig_ben = go.Figure()
        fig_ben.add_trace(go.Bar(x=df_ben['digit'], y=df_ben['actual_freq'], name='Actual Data'))
        fig_ben.add_trace(go.Scatter(x=df_ben['digit'], y=df_ben['benford_freq'], name='Benford Expected', line=dict(color='red')))
        
        st.plotly_chart(fig_ben, use_container_width=True)

    # 6A: Anomalies
    st.markdown("---")
    st.subheader("📉 Statistical Anomaly Detector")
    st.markdown("Using Isolation Forest to find unusual volume spikes.")
    
    anomalies = adv_engine.detect_anomalies_isolation_forest()
    st.write(f"Detected {len(anomalies)} anomalous events.")
    st.dataframe(anomalies.head(10))