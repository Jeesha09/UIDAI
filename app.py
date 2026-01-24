import glob
import os
import streamlit as st
import pandas as pd
import numpy as np 
import plotly.express as px
import plotly.graph_objects as go
from backend_logic import ExecutiveSummaryEngine, AadhaarAnalyticsEngine
from data_preprocessing import preprocess_dataframe
from chatbot_module import AadhaarChatbot

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
        color: #ffffff !important;
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
    
    /* Sidebar button specific styling */
    [data-testid="stSidebar"] .stButton > button {
        background-color: #2874d4 !important;
        color: #ffffff !important;
        font-weight: 600;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #1e5fb8 !important;
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
def load_trends_from_cache():
    """Load pre-generated trends analytics from cache file"""
    import json
    import os
    
    cache_file = 'trends_cache.json'
    
    if not os.path.exists(cache_file):
        st.warning(f"⚠️ Trends cache file not found. Please run `generate_trends_cache.py` first to generate trends data.")
        return {}
    
    try:
        with open(cache_file, 'r') as f:
            trends_cache = json.load(f)
        
        # Convert back to appropriate formats
        trends_data = {}
        
        # 1. Forecast Data
        if 'forecast_data' in trends_cache and trends_cache['forecast_data']:
            forecast_dict = trends_cache['forecast_data']
            if forecast_dict and 'dates' in forecast_dict:
                forecast_df = pd.DataFrame({
                    'Historical': forecast_dict['Historical'],
                    'Forecast': forecast_dict['Forecast']
                })
                forecast_df.index = pd.to_datetime(forecast_dict['dates'])
                if forecast_dict.get('yhat_lower'):
                    forecast_df['yhat_lower'] = forecast_dict['yhat_lower']
                if forecast_dict.get('yhat_upper'):
                    forecast_df['yhat_upper'] = forecast_dict['yhat_upper']
                trends_data['forecast_data'] = forecast_df
            else:
                trends_data['forecast_data'] = pd.DataFrame()
        else:
            trends_data['forecast_data'] = pd.DataFrame()
        
        # 2. Day-of-Week Data
        if 'dow_data' in trends_cache and trends_cache['dow_data']:
            dow_dict = trends_cache['dow_data']
            if dow_dict and 'days' in dow_dict:
                dow_df = pd.DataFrame(
                    dow_dict['data'],
                    index=dow_dict['days'],
                    columns=dow_dict['columns']
                )
                # Fix day ordering
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                dow_df = dow_df.reindex([d for d in day_order if d in dow_df.index])
                trends_data['dow_data'] = dow_df
            else:
                trends_data['dow_data'] = pd.DataFrame()
        else:
            trends_data['dow_data'] = pd.DataFrame()
        
        # 3. Growth Data
        if 'growth_data' in trends_cache and trends_cache['growth_data']:
            growth_dict = trends_cache['growth_data']
            if growth_dict and 'dates' in growth_dict:
                growth_df = pd.DataFrame(
                    growth_dict['data'],
                    index=pd.to_datetime(growth_dict['dates']),
                    columns=growth_dict['districts']
                )
                trends_data['growth_data'] = growth_df
            else:
                trends_data['growth_data'] = pd.DataFrame()
        else:
            trends_data['growth_data'] = pd.DataFrame()
        
        # 4. Seasonal Radar (Plotly figure)
        if 'seasonal_radar' in trends_cache and trends_cache['seasonal_radar']:
            import plotly.io as pio
            trends_data['seasonal_radar'] = pio.from_json(trends_cache['seasonal_radar'])
        else:
            trends_data['seasonal_radar'] = None
        
        # 5. Network Graph (Plotly figure)
        if 'network_graph' in trends_cache and trends_cache['network_graph']:
            import plotly.io as pio
            trends_data['network_graph'] = pio.from_json(trends_cache['network_graph'])
        else:
            trends_data['network_graph'] = None
        
        # Display cache info in sidebar
        if 'metadata' in trends_cache:
            metadata = trends_cache['metadata']
            generated_at = pd.to_datetime(metadata['generated_at']).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  ✓ Loaded trends data from cache (generated: {generated_at})")
        
        return trends_data
        
    except Exception as e:
        st.error(f"❌ Error loading trends cache: {str(e)}")
        return {}


@st.cache_data
def load_operations_from_cache():
    """Load pre-generated operations analytics from cache file"""
    import json
    import os
    
    cache_file = 'operations_cache.json'
    
    if not os.path.exists(cache_file):
        st.warning(f"⚠️ Operations cache file not found. Please run `generate_operations_cache.py` first to generate operations data.")
        return {}
    
    try:
        with open(cache_file, 'r') as f:
            operations_cache = json.load(f)
        
        # Convert back to appropriate formats
        operations_data = {}
        
        # 1. BCG Matrix Data
        if 'bcg_data' in operations_cache and operations_cache['bcg_data']:
            bcg_dict = operations_cache['bcg_data']
            if bcg_dict and 'districts' in bcg_dict:
                bcg_df = pd.DataFrame({
                    'district': bcg_dict['districts'],
                    'total_enrollments': bcg_dict['total_enrollments'],
                    'total_updates': bcg_dict['total_updates']
                })
                operations_data['bcg_data'] = bcg_df
            else:
                operations_data['bcg_data'] = pd.DataFrame()
        else:
            operations_data['bcg_data'] = pd.DataFrame()
        
        # 2. BCG Chart (Plotly figure)
        if 'bcg_chart' in operations_cache and operations_cache['bcg_chart']:
            import plotly.io as pio
            operations_data['bcg_chart'] = pio.from_json(operations_cache['bcg_chart'])
        else:
            operations_data['bcg_chart'] = None
        
        # 3. Mobile Van Priority Data
        if 'van_data' in operations_cache and operations_cache['van_data']:
            van_dict = operations_cache['van_data']
            if van_dict and 'districts' in van_dict:
                van_df = pd.DataFrame({
                    'district': van_dict['districts'],
                    'pincode': [int(p) if p.isdigit() else p for p in van_dict['pincodes']],
                    'total_enrollments': van_dict['total_enrollments'],
                    'total_updates': van_dict['total_updates'],
                    'total_activity': van_dict['total_activity'],
                    'van_priority': van_dict['van_priority']
                })
                operations_data['van_data'] = van_df
            else:
                operations_data['van_data'] = pd.DataFrame()
        else:
            operations_data['van_data'] = pd.DataFrame()
        
        # 4. Mobile Van Chart (Plotly figure)
        if 'van_chart' in operations_cache and operations_cache['van_chart']:
            import plotly.io as pio
            operations_data['van_chart'] = pio.from_json(operations_cache['van_chart'])
        else:
            operations_data['van_chart'] = None
        
        # 5. Center Productivity Data
        if 'productivity_data' in operations_cache and operations_cache['productivity_data']:
            prod_dict = operations_cache['productivity_data']
            if prod_dict and 'districts' in prod_dict:
                prod_df = pd.DataFrame({
                    'district': prod_dict['districts'],
                    'pincode': [int(p) if p.isdigit() else p for p in prod_dict['pincodes']],
                    'total_activity': prod_dict['total_activity']
                })
                operations_data['productivity_data'] = prod_df
            else:
                operations_data['productivity_data'] = pd.DataFrame()
        else:
            operations_data['productivity_data'] = pd.DataFrame()
        
        # 6. Productivity Chart (Plotly figure)
        if 'productivity_chart' in operations_cache and operations_cache['productivity_chart']:
            import plotly.io as pio
            operations_data['productivity_chart'] = pio.from_json(operations_cache['productivity_chart'])
        else:
            operations_data['productivity_chart'] = None
        
        # 7. Weekly Capacity Data
        if 'weekly_data' in operations_cache and operations_cache['weekly_data']:
            weekly_dict = operations_cache['weekly_data']
            if weekly_dict and 'days' in weekly_dict:
                weekly_df = pd.DataFrame({
                    'day_of_week': weekly_dict['days'],
                    'total_activity': weekly_dict['total_activity']
                })
                operations_data['weekly_data'] = weekly_df
            else:
                operations_data['weekly_data'] = pd.DataFrame()
        else:
            operations_data['weekly_data'] = pd.DataFrame()
        
        # 8. Weekly Chart (Plotly figure)
        if 'weekly_chart' in operations_cache and operations_cache['weekly_chart']:
            import plotly.io as pio
            operations_data['weekly_chart'] = pio.from_json(operations_cache['weekly_chart'])
        else:
            operations_data['weekly_chart'] = None
        
        # 9. Growth Velocity Data
        if 'velocity_data' in operations_cache and operations_cache['velocity_data']:
            vel_dict = operations_cache['velocity_data']
            if vel_dict and 'dates' in vel_dict:
                vel_df = pd.DataFrame({
                    'date': pd.to_datetime(vel_dict['dates']),
                    'district': vel_dict['districts'],
                    'total_enrollments': vel_dict['total_enrollments']
                })
                operations_data['velocity_data'] = vel_df
            else:
                operations_data['velocity_data'] = pd.DataFrame()
        else:
            operations_data['velocity_data'] = pd.DataFrame()
        
        # 10. Velocity Chart (Plotly figure)
        if 'velocity_chart' in operations_cache and operations_cache['velocity_chart']:
            import plotly.io as pio
            operations_data['velocity_chart'] = pio.from_json(operations_cache['velocity_chart'])
        else:
            operations_data['velocity_chart'] = None
        
        # Display cache info in sidebar
        if 'metadata' in operations_cache:
            metadata = operations_cache['metadata']
            generated_at = pd.to_datetime(metadata['generated_at']).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  ✓ Loaded operations data from cache (generated: {generated_at})")
        
        return operations_data
        
    except Exception as e:
        st.error(f"❌ Error loading operations cache: {str(e)}")
        return {}


@st.cache_data
def load_security_from_cache():
    """Load pre-generated security analytics from cache file"""
    import json
    import os
    
    cache_file = 'security_cache.json'
    
    if not os.path.exists(cache_file):
        st.warning(f"⚠️ Security cache file not found. Please run `generate_security_cache.py` first to generate security data.")
        return {}
    
    try:
        with open(cache_file, 'r') as f:
            security_cache = json.load(f)
        
        # Convert back to appropriate formats
        security_data = {}
        
        # 1. Benford's Law Data
        if 'benford_data' in security_cache and security_cache['benford_data']:
            benford_dict = security_cache['benford_data']
            if benford_dict and 'digits' in benford_dict:
                benford_df = pd.DataFrame({
                    'digit': benford_dict['digits'],
                    'actual_freq': benford_dict['actual_freq'],
                    'benford_freq': benford_dict['benford_freq']
                })
                security_data['benford_distribution'] = benford_df
                security_data['benford_deviation_score'] = benford_dict['deviation_score']
                security_data['benford_is_suspicious'] = benford_dict['is_suspicious']
            else:
                security_data['benford_distribution'] = pd.DataFrame()
                security_data['benford_deviation_score'] = 0.0
                security_data['benford_is_suspicious'] = False
        else:
            security_data['benford_distribution'] = pd.DataFrame()
            security_data['benford_deviation_score'] = 0.0
            security_data['benford_is_suspicious'] = False
        
        # 2. Benford Chart (Plotly figure)
        if 'benford_chart' in security_cache and security_cache['benford_chart']:
            import plotly.io as pio
            security_data['benford_chart'] = pio.from_json(security_cache['benford_chart'])
        else:
            security_data['benford_chart'] = None
        
        # 3. Statistical Outliers Data
        if 'outliers_data' in security_cache and security_cache['outliers_data']:
            outliers_dict = security_cache['outliers_data']
            if outliers_dict and 'dates' in outliers_dict:
                outliers_df = pd.DataFrame({
                    'date': pd.to_datetime(outliers_dict['dates']),
                    'total_activity': outliers_dict['total_activity'],
                    'is_anomaly': outliers_dict['is_anomaly'],
                    'threshold': outliers_dict['threshold']
                })
                security_data['outliers_data'] = outliers_df
            else:
                security_data['outliers_data'] = pd.DataFrame()
        else:
            security_data['outliers_data'] = pd.DataFrame()
        
        # 4. Outliers Chart (Plotly figure)
        if 'outliers_chart' in security_cache and security_cache['outliers_chart']:
            import plotly.io as pio
            security_data['outliers_chart'] = pio.from_json(security_cache['outliers_chart'])
        else:
            security_data['outliers_chart'] = None
        
        # 5. Anomaly Count
        security_data['anomaly_count'] = security_cache.get('anomaly_count', 0)
        
        # 6. Volatility Analysis Data
        if 'volatility_data' in security_cache and security_cache['volatility_data']:
            vol_dict = security_cache['volatility_data']
            if vol_dict and 'pincodes' in vol_dict:
                vol_df = pd.DataFrame({
                    'pincode': vol_dict['pincodes'],
                    'mean': vol_dict['mean'],
                    'std': vol_dict['std'],
                    'count': vol_dict['count'],
                    'variance_score': vol_dict['variance_score']
                })
                security_data['volatility_data'] = vol_df
            else:
                security_data['volatility_data'] = pd.DataFrame()
        else:
            security_data['volatility_data'] = pd.DataFrame()
        
        # 7. Volatility Chart (Plotly figure)
        if 'volatility_chart' in security_cache and security_cache['volatility_chart']:
            import plotly.io as pio
            security_data['volatility_chart'] = pio.from_json(security_cache['volatility_chart'])
        else:
            security_data['volatility_chart'] = None
        
        # 8. State Variance Data
        if 'state_variance_data' in security_cache and security_cache['state_variance_data']:
            state_dict = security_cache['state_variance_data']
            if state_dict and 'states' in state_dict:
                state_df = pd.DataFrame({
                    'state': state_dict['states'],
                    'district': state_dict['districts'],
                    'total_activity': state_dict['total_activity']
                })
                security_data['state_variance_data'] = state_df
            else:
                security_data['state_variance_data'] = pd.DataFrame()
        else:
            security_data['state_variance_data'] = pd.DataFrame()
        
        # 9. State Variance Chart (Plotly figure)
        if 'state_variance_chart' in security_cache and security_cache['state_variance_chart']:
            import plotly.io as pio
            security_data['state_variance_chart'] = pio.from_json(security_cache['state_variance_chart'])
        else:
            security_data['state_variance_chart'] = None
        
        # Display cache info in sidebar
        if 'metadata' in security_cache:
            metadata = security_cache['metadata']
            generated_at = pd.to_datetime(metadata['generated_at']).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  ✓ Loaded security data from cache (generated: {generated_at})")
        
        return security_data
        
    except Exception as e:
        st.error(f"❌ Error loading security cache: {str(e)}")
        return {}


@st.cache_data
def load_ml_predictions_from_cache():
    """Load pre-generated ML predictions from cache file"""
    import json
    import os
    
    cache_file = 'ml_predictions_cache.json'
    
    if not os.path.exists(cache_file):
        st.warning(f"⚠️ ML cache file not found. Please run `generate_ml_predictions.py` first to generate predictions.")
        return {}
    
    try:
        with open(cache_file, 'r') as f:
            ml_cache = json.load(f)
        
        # Convert back to DataFrames
        ml_predictions = {}
        
        if 'fraud_scores' in ml_cache and ml_cache['fraud_scores']:
            ml_predictions['fraud_scores'] = pd.DataFrame(ml_cache['fraud_scores'])
        else:
            ml_predictions['fraud_scores'] = pd.DataFrame()
        
        if 'churn_predictions' in ml_cache and ml_cache['churn_predictions']:
            ml_predictions['churn_predictions'] = pd.DataFrame(ml_cache['churn_predictions'])
        else:
            ml_predictions['churn_predictions'] = pd.DataFrame()
        
        if 'compliance_predictions' in ml_cache and ml_cache['compliance_predictions']:
            ml_predictions['compliance_predictions'] = pd.DataFrame(ml_cache['compliance_predictions'])
        else:
            ml_predictions['compliance_predictions'] = pd.DataFrame()
        
        if 'cross_anomalies_all' in ml_cache and ml_cache['cross_anomalies_all']:
            ml_predictions['cross_anomalies_all'] = pd.DataFrame(ml_cache['cross_anomalies_all'])
        else:
            ml_predictions['cross_anomalies_all'] = pd.DataFrame()
        
        if 'cross_anomalies_only' in ml_cache and ml_cache['cross_anomalies_only']:
            ml_predictions['cross_anomalies_only'] = pd.DataFrame(ml_cache['cross_anomalies_only'])
        else:
            ml_predictions['cross_anomalies_only'] = pd.DataFrame()
        
        if 'peak_load_forecast' in ml_cache and ml_cache['peak_load_forecast']:
            df_peak = pd.DataFrame(ml_cache['peak_load_forecast'])
            # Convert date strings back to datetime
            if 'date' in df_peak.columns:
                df_peak['date'] = pd.to_datetime(df_peak['date'])
            ml_predictions['peak_load_forecast'] = df_peak
        else:
            ml_predictions['peak_load_forecast'] = pd.DataFrame()
        
        if 'transition_predictions' in ml_cache and ml_cache['transition_predictions']:
            df_trans = pd.DataFrame(ml_cache['transition_predictions'])
            # Convert date strings back to datetime
            date_cols = ['avg_enrollment_date', 'predicted_transition_date', 'campaign_start_date']
            for col in date_cols:
                if col in df_trans.columns:
                    df_trans[col] = pd.to_datetime(df_trans[col])
            ml_predictions['transition_predictions'] = df_trans
        else:
            ml_predictions['transition_predictions'] = pd.DataFrame()
        
        # Display cache info in sidebar
        if 'metadata' in ml_cache:
            metadata = ml_cache['metadata']
            generated_at = pd.to_datetime(metadata['generated_at']).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  ✓ Loaded ML predictions from cache (generated: {generated_at})")
        
        return ml_predictions
        
    except Exception as e:
        st.error(f"❌ Error loading ML cache: {str(e)}")
        return {}



@st.cache_data
def load_demographics_from_cache():
    """Load pre-generated demographics analytics from cache file"""
    import json
    import os
    
    cache_file = 'demographics_cache.json'
    
    if not os.path.exists(cache_file):
        st.warning(f"⚠️ Demographics cache file not found. Please run `generate_demographics_cache.py` first to generate demographics data.")
        return {}
    
    try:
        with open(cache_file, 'r') as f:
            demographics_cache = json.load(f)
        
        # Display cache info in sidebar
        if 'metadata' in demographics_cache:
            metadata = demographics_cache['metadata']
            generated_at = pd.to_datetime(metadata['generated_at']).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  ✓ Loaded demographics data from cache (generated: {generated_at})")
        
        return demographics_cache
        
    except Exception as e:
        st.error(f"❌ Error loading demographics cache: {str(e)}")
        return {}


@st.cache_data
def load_executive_from_cache():
    """Load pre-generated executive summary from cache file"""
    import json
    import os
    
    cache_file = 'executive_cache.json'
    
    if not os.path.exists(cache_file):
        st.warning(f"⚠️ Executive cache file not found. Please run `generate_executive_cache.py` first to generate executive summary data.")
        return {}
    
    try:
        with open(cache_file, 'r') as f:
            executive_cache = json.load(f)
        
        # Convert back to appropriate formats
        executive_data = {}
        
        # 1. Early Warning System
        if 'early_warning' in executive_cache:
            ew = executive_cache['early_warning']
            executive_data['early_warning'] = {
                'metric_value': ew['metric_value'],
                'details_df': pd.DataFrame(ew['details_df'])
            }
        else:
            executive_data['early_warning'] = {'metric_value': 0, 'details_df': pd.DataFrame()}
        
        # 2. Stagnation Detection
        if 'stagnation' in executive_cache:
            st_data = executive_cache['stagnation']
            executive_data['stagnation'] = {
                'total_stagnant': st_data['total_stagnant'],
                'pincode_list': st_data['pincode_list']
            }
        else:
            executive_data['stagnation'] = {'total_stagnant': 0, 'pincode_list': []}
        
        # 3. Benchmarks
        if 'benchmarks' in executive_cache:
            bm = executive_cache['benchmarks']
            executive_data['benchmarks'] = {
                'top_performers': pd.DataFrame(bm['top_performers']),
                'bottom_performers': pd.DataFrame(bm['bottom_performers'])
            }
        else:
            executive_data['benchmarks'] = {
                'top_performers': pd.DataFrame(),
                'bottom_performers': pd.DataFrame()
            }
        
        # 4. Clusters
        if 'clusters' in executive_cache and executive_cache['clusters']:
            executive_data['clusters'] = pd.DataFrame(executive_cache['clusters'])
        else:
            executive_data['clusters'] = pd.DataFrame()
        
        # 5. Total Volume
        executive_data['total_volume'] = executive_cache.get('total_volume', 0)
        
        # 6. Stats
        executive_data['stats'] = executive_cache.get('stats', {
            'total_enrollments': 0,
            'total_districts': 0,
            'total_states': 0
        })
        
        print("  ✓ Executive summary loaded from cache")
        
        return executive_data
        
    except Exception as e:
        st.error(f"❌ Error loading executive cache: {str(e)}")
        return {}


def preload_all_charts(_df_enrol, _df_bio, _df_demo):
    """Load pre-generated visualizations and predictions from cache files"""
    charts = {}
    
    print("Loading cached data...")
    
    # Load Executive Summary from Cache (FAST!)
    executive_data = load_executive_from_cache()
    charts.update(executive_data)
    print("  ✓ Executive summary loaded from cache")
    
    # Load Trends Analytics from Cache (FAST!)
    trends_data = load_trends_from_cache()
    charts.update(trends_data)
    print("  ✓ Trends analytics loaded from cache")
    
    # Load Operations Analytics from Cache (FAST!)
    operations_data = load_operations_from_cache()
    charts.update(operations_data)
    print("  ✓ Operations analytics loaded from cache")
    
    # Load Security Analytics from Cache (FAST!)
    security_data = load_security_from_cache()
    charts.update(security_data)
    print("  ✓ Security analytics loaded from cache")
    
    # Load ML Predictions from Cache (FAST!)
    ml_predictions = load_ml_predictions_from_cache()
    charts.update(ml_predictions)
    print("  ✓ ML predictions loaded from cache")

    # Load Demographics Analytics from Cache (FAST!)
    demographics_data = load_demographics_from_cache()
    charts.update(demographics_data)
    print("  ✓ Demographics analytics loaded from cache")
    
    print("All cached data ready!")
    return charts

try:
    # Streamlit Cloud health check optimization
    # Load ALL data from cache files (no CSV loading!)
    if "data_loaded" not in st.session_state:
        # Create progress bar
        progress_text = "Loading all cached analytics (Executive Summary, Trends, Operations, ML, Demographics)..."
        progress_bar = st.progress(0, text=progress_text)
        
        # Skip CSV loading - Load everything from cache
        progress_bar.progress(20, text="Loading cached data...")
        
        # Load all cached data (20-100% progress)
        progress_bar.progress(40, text="Loading Executive Summary, Trends & Operations from cache...")
        preloaded_charts = preload_all_charts(None, None, None)
        
        progress_bar.progress(80, text="Finalizing...")
        
        # Initialize empty dataframes (CSV loading disabled for performance)
        progress_bar.progress(90, text="Initializing chatbot components...")
        df_enrol, df_bio, df_demo = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        progress_bar.progress(100, text="Complete!")
        progress_bar.empty()  # Remove progress bar
        
        # Store in session state
        st.session_state.data_loaded = True
        st.session_state.df_enrol = df_enrol
        st.session_state.df_bio = df_bio
        st.session_state.df_demo = df_demo
        st.session_state.preloaded_charts = preloaded_charts
        
        st.toast(f"✅ Ready! All analytics loaded from cache", icon="🎉")
    else:
        # Use cached data
        df_enrol = st.session_state.df_enrol
        df_bio = st.session_state.df_bio
        df_demo = st.session_state.df_demo
        preloaded_charts = st.session_state.preloaded_charts

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
    "Security & Integrity",
    "ML Predictions & Intelligence"
])

st.sidebar.markdown("---")
st.sidebar.info("System Status: **Online**")

# ==========================================
# 4. PAGE LOGIC
# ==========================================

# --- PAGE 1: EXECUTIVE SUMMARY ---
if page == "Executive Summary":
    st.title("Executive Command Center")
    st.markdown("**High-level overview of ecosystem health and critical alerts** • _Data loaded from cache_")
    
    # ---------------------------------------------------------
    # SECTION 1: CRITICAL ALERTS (Metrics + Popups)
    # ---------------------------------------------------------
    st.subheader("Critical Alerts")
    
    # Get Data from Cache
    early_warning = preloaded_charts.get('early_warning', {'metric_value': 0, 'details_df': pd.DataFrame()})
    stagnation = preloaded_charts.get('stagnation', {'total_stagnant': 0, 'pincode_list': []})
    total_vol = preloaded_charts.get('total_volume', 0)
    
    # Display Big Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style='padding: 1rem; background-color: #1a1d29; border-radius: 0.5rem; border: 1px solid #2d3139; min-height: 120px;'>
            <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
                <span style='color: #a9b1c0; font-size: 0.875rem;'>Critical Decline Districts</span>
                <span style='background-color: rgba(255, 75, 75, 0.1); color: #ff4b4b; padding: 0.125rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem; font-weight: 500;'>↓ -5%</span>
            </div>
            <p style='color: #fafafa; font-size: 2.5rem; font-weight: 600; margin: 0; line-height: 1.2;'>{early_warning['metric_value']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
        
        # THE "VIEW ALL" POPUP BUTTON
        with st.popover("View All Declining Districts"):
            st.markdown("### Full List of Critical Districts")
            st.write("Districts with >20% drop in enrollment volume.")
            # Display as static table (top 10 rows)
            if not early_warning['details_df'].empty:
                display_df = early_warning['details_df'][['change_pct']].head(10).copy()
                display_df['change_pct'] = display_df['change_pct'].apply(lambda x: f"{x:.1f}%")
                st.table(display_df)
            else:
                st.info("No critical decline districts found.")

    with col2:
        st.metric("Stagnant Pincodes (30 Days)", stagnation['total_stagnant'], delta_color="inverse")
        
        # THE "VIEW ALL" POPUP BUTTON
        with st.popover("View All Stagnant Pincodes"):
            st.markdown("### Full List of Inactive Pincodes")
            st.write("Pincodes with zero activity in the last 30 days.")
            if stagnation['pincode_list']:
                # Display as static table (top 10 rows)
                display_df = pd.DataFrame(stagnation['pincode_list'][:10], columns=["Inactive Pincode"])
                display_df["Inactive Pincode"] = display_df["Inactive Pincode"].astype(str)
                st.table(display_df)
            else:
                st.success("No stagnant pincodes found.")

    with col3:
        st.metric("Total Enrollment Volume", f"{total_vol:,.0f}")

    st.markdown("---")
    
    # ---------------------------------------------------------
    # SECTION 2: LEADERBOARDS (Keeping the Top 5 Tables)
    # ---------------------------------------------------------
    st.subheader("Performance Leaderboard")
    benchmarks = preloaded_charts.get('benchmarks', {'top_performers': pd.DataFrame(), 'bottom_performers': pd.DataFrame()})
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Top 5 Performing Districts")
        if not benchmarks['top_performers'].empty:
            st.dataframe(benchmarks['top_performers'], hide_index=True, use_container_width=True)
        else:
            st.info("No benchmark data available.")
    with c2:
        st.markdown("### Bottom 5 Districts (Needs Support)")
        if not benchmarks['bottom_performers'].empty:
            st.dataframe(benchmarks['bottom_performers'], hide_index=True, use_container_width=True)
        else:
            st.info("No benchmark data available.")

    # ---------------------------------------------------------
    # SECTION 3: MAP (Your India Map)
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("Location Archetypes (Cluster Map)")
    
    clusters = preloaded_charts.get('clusters', pd.DataFrame())
    
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
                'Balanced Growth': '#1f77b4',
                'Rapid Growth': '#ff7f0e',
                'Stable': '#2ca02c',
                'Dormant': '#d62728'
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
        st.warning("Not enough data to generate clusters. Run `python generate_executive_cache.py` to generate cache.")

# --- PAGE 2: OPERATIONS ---
elif page == "Operations & Logistics":
    st.title("Operations & Resource Planning")
    st.markdown("**Comprehensive operational intelligence for infrastructure allocation and capacity optimization** • _Data loaded from cache_")
    
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
    
    bcg_chart = preloaded_charts.get('bcg_chart')
    if bcg_chart:
        st.plotly_chart(bcg_chart, use_container_width=True)
    else:
        st.warning("⚠️ BCG matrix not available. Run `python generate_operations_cache.py` to generate.")
    
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
    
    van_chart = preloaded_charts.get('van_chart')
    if van_chart:
        st.plotly_chart(van_chart, use_container_width=True)
    else:
        st.warning("⚠️ Mobile van priority data not available. Run `python generate_operations_cache.py` to generate.")
    
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
    
    productivity_chart = preloaded_charts.get('productivity_chart')
    if productivity_chart:
        st.plotly_chart(productivity_chart, use_container_width=True)
    else:
        st.warning("⚠️ Center productivity data not available. Run `python generate_operations_cache.py` to generate.")
    
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
    
    weekly_chart = preloaded_charts.get('weekly_chart')
    if weekly_chart:
        st.plotly_chart(weekly_chart, use_container_width=True)
    else:
        st.warning("⚠️ Weekly capacity data not available. Run `python generate_operations_cache.py` to generate.")
    
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
    
    velocity_chart = preloaded_charts.get('velocity_chart')
    if velocity_chart:
        st.plotly_chart(velocity_chart, use_container_width=True)
    else:
        st.warning("⚠️ Growth velocity data not available. Run `python generate_operations_cache.py` to generate.")


# --- PAGE 3: TRENDS ---
elif page == "Trends & Forecasting":
    st.title("Trends & Predictive Analytics")
    st.markdown("**Advanced predictive analytics and trend forecasting** • _Data loaded from cache_")
    
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
    
    seasonal_radar = preloaded_charts.get('seasonal_radar')
    if seasonal_radar:
        st.plotly_chart(seasonal_radar, use_container_width=True)
    else:
        st.warning("⚠️ Seasonal radar chart not available. Run `python generate_trends_cache.py` to generate.")
    
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
        # Force correct day order for display
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_data_ordered = dow_data.reindex([d for d in day_order if d in dow_data.index])
        
        # Check if Urban has any data
        has_urban = 'Urban' in dow_data_ordered.columns and dow_data_ordered['Urban'].sum() > 0
        
        # Use Plotly for proper ordering control
        fig_dow = go.Figure()
        if 'Rural' in dow_data_ordered.columns:
            fig_dow.add_trace(go.Bar(
                name='Rural',
                x=dow_data_ordered.index,
                y=dow_data_ordered['Rural'],
                marker_color='#2ca02c'
            ))
        if has_urban:
            fig_dow.add_trace(go.Bar(
                name='Urban',
                x=dow_data_ordered.index,
                y=dow_data_ordered['Urban'],
                marker_color='#d62728'
            ))
        
        fig_dow.update_layout(
            barmode='group',
            height=500,
            xaxis_title='Day of Week',
            yaxis_title='Enrollment Count',
            template='plotly_white',
            xaxis={'categoryorder': 'array', 'categoryarray': day_order}
        )
        st.plotly_chart(fig_dow, use_container_width=True)
        
        if not has_urban:
            st.info("ℹ️ No Urban enrollment data available in the dataset - all enrollments are from Rural areas.")
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
    
    network_graph = preloaded_charts.get('network_graph')
    if network_graph:
        st.plotly_chart(network_graph, use_container_width=True)
    else:
        st.warning("⚠️ Network graph not available. Run `python generate_trends_cache.py` to generate.")



# --- PAGE 4: DEMOGRAPHICS ---
elif page == "Demographics & Policy":
    st.title("📊 Demographics & Behavior Analysis")
    st.markdown("**Comprehensive age distribution and behavioral insights** • _Data loaded from cache_")
    
    # Get cached data
    age_distribution = preloaded_charts.get('age_distribution', {})
    health_scores = preloaded_charts.get('health_scores', {})
    lag_distributions = preloaded_charts.get('lag_distributions', {})
    age_growth = preloaded_charts.get('age_growth', {})
    state_comparisons = preloaded_charts.get('state_comparisons', {})
    
    if not age_distribution:
        st.error("⚠️ Demographics data not available. Run `python generate_demographics_cache.py` to generate.")
        st.stop()
    
    # District filter
    districts = sorted(list(age_distribution.keys()))
    selected_dist = st.selectbox("Select District", districts, key="dist_filter")
    
    # ===== CHART 1: Age Distribution =====
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("**Age Group Distribution**")
        
        if selected_dist in age_distribution:
            age_data = age_distribution[selected_dist]
            
            fig_4a = px.pie(
                names=list(age_data.keys()), 
                values=list(age_data.values()), 
                hole=0.5,
                color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
            )
            fig_4a.update_traces(textinfo='label+percent', textfont_size=14)
            fig_4a.update_layout(showlegend=True, height=400)
            st.plotly_chart(fig_4a, use_container_width=True)
        else:
            st.warning("No data for selected district")
    
    # ===== CHART 2: Health Score =====
    with col2:
        st.subheader("**Health Score** (Update-to-Enrollment)")
        
        if selected_dist in health_scores:
            health_data = health_scores[selected_dist]
            health_score = health_data['health_score']
            
            fig_5a = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=health_score,
                number={'font': {'size': 50}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#3498db"},
                    'steps': [
                        {'range': [0, 30], 'color': "#ffcdd2"},
                        {'range': [30, 70], 'color': "#fff9c4"}, 
                        {'range': [70, 100], 'color': "#c8e6c9"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
                }
            ))
            fig_5a.update_layout(height=400, margin=dict(l=20, r=20))
            st.plotly_chart(fig_5a, use_container_width=True)
            
            st.metric(
                "Total Enrollment", 
                f"{health_data['total_enrollment']:,.0f}", 
                delta=f"{health_data['total_updates']:,.0f} Updates"
            )
        else:
            st.warning("No health score data for selected district")
    
    # ===== CHART 3: Update Lag Distribution =====
    st.markdown("---")
    col1, col2 = st.columns([0.6, 0.4])
    
    with col1:
        st.subheader("**Update Lag Distribution**")
        age_options = list(lag_distributions.keys())
        selected_age = st.selectbox("Filter by Age Group", age_options)
        
        if selected_age in lag_distributions:
            lag_data = pd.Series(lag_distributions[selected_age]['values'])
            mean_lag = lag_distributions[selected_age]['mean']
            
            fig_5c = px.histogram(lag_data, nbins=50, color_discrete_sequence=['#9b59b6'])
            fig_5c.add_vline(mean_lag, line_dash="dash", line_color="red", 
                            annotation_text=f"Mean: {mean_lag:.1f} days")
            fig_5c.update_layout(height=400, xaxis_title="Days to Update", yaxis_title="Count")
            st.plotly_chart(fig_5c, use_container_width=True)
        else:
            st.warning("No lag data available")
    
    # ===== CHART 4: Age Growth Over Time =====
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("**Age Group Growth Over Time**")
        
        if selected_dist in age_growth:
            growth_data_dict = age_growth[selected_dist]
            growth_df = pd.DataFrame({
                'date': pd.to_datetime(growth_data_dict['dates']),
                '0-5 years': growth_data_dict['age_0_5'],
                '5-18 years': growth_data_dict['age_5_17'],
                '18+ years': growth_data_dict['age_18_greater']
            })
            
            fig_4b = px.area(
                growth_df, 
                x='date', 
                y=['0-5 years', '5-18 years', '18+ years'],
                color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
            )
            fig_4b.update_layout(height=400, legend_orientation="h", xaxis_title="Date", yaxis_title="Enrollment Count")
            st.plotly_chart(fig_4b, use_container_width=True)
        else:
            st.warning("No time-series data available for selected district")
    
    # ===== CHART 5: Behavioral Segmentation =====
    with col2:
        st.subheader("**Behavioral Segmentation**")
        states = sorted(list(state_comparisons.keys()))
        
        state_a = st.selectbox("State A", states, key="state_a")
        state_b = st.selectbox("State B", states[1:] if len(states) > 1 else states, key="state_b")
        
        demo_a = state_comparisons.get(state_a, {}).get('demo_updates', 0)
        bio_a = state_comparisons.get(state_a, {}).get('bio_updates', 0)
        demo_b = state_comparisons.get(state_b, {}).get('demo_updates', 0)
        bio_b = state_comparisons.get(state_b, {}).get('bio_updates', 0)
        
        fig_8b = go.Figure()
        fig_8b.add_trace(go.Bar(
            name='Demo Updates', 
            x=[state_a, state_b], 
            y=[demo_a, demo_b], 
            marker_color='#3498db', 
            text=[f'{demo_a:,.0f}', f'{demo_b:,.0f}'], 
            textposition='outside'
        ))
        fig_8b.add_trace(go.Bar(
            name='Bio Updates', 
            x=[state_a, state_b], 
            y=[bio_a, bio_b], 
            marker_color='#e74c3c', 
            text=[f'{bio_a:,.0f}', f'{bio_b:,.0f}'], 
            textposition='outside'
        ))
        fig_8b.update_layout(barmode='group', height=400, xaxis_title="State", yaxis_title="Update Count")
        st.plotly_chart(fig_8b, use_container_width=True)


# --- PAGE 5: SECURITY ---
elif page == "Security & Integrity":
    st.title("Security & Forensic Audit")
    st.markdown("**Comprehensive fraud detection and data integrity analysis** • _Data loaded from cache_")
    
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
    
    benford_chart = preloaded_charts.get('benford_chart')
    deviation = preloaded_charts.get('benford_deviation_score', 0.0)
    is_suspicious = preloaded_charts.get('benford_is_suspicious', False)
    
    if benford_chart:
        if is_suspicious:
            st.error(f"⚠️ HIGH RISK DETECTED | Deviation Score: {deviation:.3f}")
            st.warning("The data shows significant deviation from Benford's Law. Recommend immediate audit.")
        else:
            st.success(f"✓ Normal Organic Behavior | Deviation Score: {deviation:.3f}")
        
        st.plotly_chart(benford_chart, use_container_width=True)
    else:
        st.warning("⚠️ Benford's Law analysis not available. Run `python generate_security_cache.py` to generate.")
    
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
    
    outliers_chart = preloaded_charts.get('outliers_chart')
    anomaly_count = preloaded_charts.get('anomaly_count', 0)
    
    if outliers_chart:
        st.metric("Suspicious Activity Days Detected", anomaly_count)
        st.plotly_chart(outliers_chart, use_container_width=True)
    else:
        st.warning("⚠️ Statistical outliers analysis not available. Run `python generate_security_cache.py` to generate.")
    
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
    
    volatility_chart = preloaded_charts.get('volatility_chart')
    volatility_data = preloaded_charts.get('volatility_data', pd.DataFrame())
    
    if volatility_chart:
        st.plotly_chart(volatility_chart, use_container_width=True)
        
        # Display detailed table
        if not volatility_data.empty:
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
        st.warning("⚠️ Volatility analysis not available. Run `python generate_security_cache.py` to generate.")
    
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
    
    state_variance_chart = preloaded_charts.get('state_variance_chart')
    
    if state_variance_chart:
        st.plotly_chart(state_variance_chart, use_container_width=True)
    else:
        st.warning("⚠️ State variance analysis not available. Run `python generate_security_cache.py` to generate.")


# --- PAGE 6: ML PREDICTIONS & INTELLIGENCE ---
elif page == "ML Predictions & Intelligence":
    st.title("🤖 Machine Learning Predictions & Advanced Intelligence")
    st.markdown("**Predictive analytics, risk forecasting, and AI-powered operational intelligence**")
    
    # Display cache information
    import os
    import json
    cache_file = 'ml_predictions_cache.json'
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                ml_cache = json.load(f)
            if 'metadata' in ml_cache:
                metadata = ml_cache['metadata']
                generated_at = pd.to_datetime(metadata['generated_at']).strftime('%B %d, %Y at %I:%M %p')
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.info(f"📊 **Predictions generated:** {generated_at} | **Total predictions:** {sum(metadata['predictions'].values()):,}")
                with col2:
                    st.markdown("To regenerate predictions, run: `python generate_ml_predictions.py`")
        except:
            pass
    else:
        st.warning("⚠️ ML predictions cache not found. Run `python generate_ml_predictions.py` to generate predictions.")
    
    # ---------------------------------------------------------
    # SECTION 1: MULTI-MODAL FRAUD SCORE
    # ---------------------------------------------------------
    st.markdown("---")
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        st.subheader("🎯 Multi-Modal Fraud Risk Score")
    with col2:
        with st.popover("ℹ️"):
            st.markdown("""
            **What This Chart Shows:**
            - **Composite fraud score (0-100)** ranking pincodes by suspicious activity patterns
            - **Top 20 highest-risk pincodes** requiring immediate audit attention
            - **Risk-level color coding:** Red (CRITICAL), Orange (HIGH), Yellow (MEDIUM), Green (LOW)
            - **Top signal identification:** Which detection method flagged each pincode
            
            **Key Insights from This Analysis:**
            - CRITICAL pincodes (score ≥70) show multiple fraud indicators simultaneously
            - "Top signal" reveals the primary fraud pattern (e.g., Benford violation = fabricated data)
            - High scores don't prove fraud, but indicate areas needing investigation
            - Most pincodes cluster in LOW range - high scores are true outliers
            - Expandable table shows all 5 signal scores for forensic deep-dive
            
            ---
            
            **Calculation Methodology:**
            
            **Composite Score Formula:**
            ```
            Fraud Score = (Benford × 0.30) + (Isolation × 0.25) + 
                         (Outlier × 0.20) + (Volatility × 0.15) + (Ratio × 0.10)
            ```
            
            **Signal 1: Benford's Law (30% weight)**
            - **Calculation:** Compares first-digit distribution of activity volumes to Benford's theoretical distribution
            - **Formula:** `Σ|actual_freq(d) - benford_freq(d)|` for digits 1-9, where `benford_freq(d) = log₁₀(1 + 1/d)`
            - **Rationale:** Natural data follows Benford's Law; fabricated numbers don't. High deviation indicates potential data manipulation
            - **Threshold:** Only applied to pincodes with >100 total activity and ≥10 data points
            
            **Signal 2: Isolation Forest (25% weight)**
            - **Calculation:** Sklearn's IsolationForest algorithm (contamination=0.1) on total activity volumes
            - **Formula:** Decision function normalized to 0-100, inverted so higher = more anomalous
            - **Rationale:** Unsupervised ML detects statistical outliers without predefined rules, catches unusual patterns
            - **Why chosen:** Handles high-dimensional data and doesn't assume normal distribution
            
            **Signal 3: Statistical Outliers (20% weight)**
            - **Calculation:** Count of days where activity > mean + 2.5σ, converted to percentage
            - **Formula:** `(outlier_days / total_days) × 100 × 2`, capped at 100
            - **Rationale:** Frequent spikes suggest irregular operations or data dumping
            - **Threshold:** 2.5 standard deviations chosen to minimize false positives
            
            **Signal 4: Volatility (15% weight)**
            - **Calculation:** Coefficient of variation across time series
            - **Formula:** `(std_dev / mean) × 50`, capped at 100
            - **Rationale:** Erratic patterns indicate inconsistent operations or gaming of quotas
            - **Why CV:** Normalizes volatility across different volume scales
            
            **Signal 5: Cross-Dataset Ratio (10% weight)**
            - **Calculation:** Deviation from expected biometric-to-enrollment ratio (0.7-1.2)
            - **Formula:** If ratio < 0.7: `(0.7 - ratio) × 100`; If ratio > 1.2: `(ratio - 1.2) × 50`
            - **Rationale:** Impossible ratios indicate data integrity issues (ghost enrollments or phantom updates)
            
            ---
            
            **Action Items:**
            - **CRITICAL (≥70):** Deploy field audit team within 24 hours, freeze pincode operations pending review
            - **HIGH (50-69):** Schedule verification visit within 7 days, cross-reference with other databases
            - **MEDIUM (30-49):** Add to monitoring watchlist, flag for next quarterly audit
            - **Investigate "top signal"** to understand specific fraud pattern (Benford = data fabrication, Isolation = unusual volume, etc.)
            - **Prioritize by score:** Work through list top-to-bottom for maximum impact
            - **Document findings:** Feed audit results back to improve detection thresholds
            """)
    
    # Use preloaded data
    fraud_scores = preloaded_charts.get('fraud_scores', pd.DataFrame())
    
    if not fraud_scores.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            critical_count = len(fraud_scores[fraud_scores['risk_level'] == 'CRITICAL'])
            st.metric("🚨 Critical Risk Pincodes", critical_count)
        with col2:
            high_count = len(fraud_scores[fraud_scores['risk_level'] == 'HIGH'])
            st.metric("⚠️ High Risk", high_count)
        with col3:
            avg_score = fraud_scores['fraud_score'].mean()
            st.metric("Average Fraud Score", f"{avg_score:.1f}")
        with col4:
            max_score = fraud_scores['fraud_score'].max()
            st.metric("Highest Score", f"{max_score:.1f}")
        
        # Distribution Overview
        st.markdown("#### 📊 Fraud Score Distribution Across All Pincodes")
        st.markdown("*Most pincodes have low risk - high scores are true outliers needing investigation*")
        
        fig_dist = px.histogram(
            fraud_scores,
            x='fraud_score',
            color='risk_level',
            color_discrete_map={'CRITICAL': '#e74c3c', 'HIGH': '#e67e22', 'MEDIUM': '#f39c12', 'LOW': '#27ae60'},
            nbins=20,
            labels={'fraud_score': 'Fraud Risk Score', 'count': 'Number of Pincodes'},
            title="Distribution of Fraud Scores (All Pincodes)",
            template="plotly",
            height=400
        )
        fig_dist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig_dist.update_traces(marker=dict(line=dict(width=0)))
        fig_dist.add_vline(x=70, line_dash="dash", line_color="red", annotation_text="CRITICAL")
        fig_dist.add_vline(x=50, line_dash="dash", line_color="orange", annotation_text="HIGH")
        fig_dist.add_vline(x=30, line_dash="dash", line_color="#f39c12", annotation_text="MEDIUM")
        st.plotly_chart(fig_dist, use_container_width=True)
        
        st.markdown("---")
        
        # Visualization: Top 20 High-Risk Pincodes
        st.markdown("#### 🎯 Top 20 Highest Risk Pincodes (Immediate Audit Required)")
        st.markdown("*Hover over bars to see which fraud signal flagged each pincode*")
        
        top_fraud = fraud_scores.head(20)
        
        fig_fraud = px.bar(
            top_fraud,
            x='pincode',
            y='fraud_score',
            color='risk_level',
            color_discrete_map={'CRITICAL': '#e74c3c', 'HIGH': '#e67e22', 'MEDIUM': '#f39c12', 'LOW': '#27ae60'},
            hover_data=['top_signal', 'benford_score', 'isolation_score', 'volatility_score'],
            labels={'fraud_score': 'Composite Fraud Score', 'pincode': 'Pincode'},
            title="",
            template="plotly",
            height=500,
            text='fraud_score'
        )
        fig_fraud.update_traces(width=0.8)
        fig_fraud.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_fraud.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Critical Threshold", annotation_position="right")
        fig_fraud.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="High Threshold", annotation_position="right")
        fig_fraud.update_xaxes(tickangle=45, title="Pincode")
        fig_fraud.update_yaxes(title="Fraud Risk Score (0-100)", range=[0, 105])
        fig_fraud.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_fraud, use_container_width=True)
        
        # Detailed table with expandable view
        with st.expander("📋 View Detailed Fraud Score Breakdown (Top 50)"):
            display_cols = ['pincode', 'fraud_score', 'risk_level', 'top_signal', 
                          'benford_score', 'isolation_score', 'volatility_score', 'ratio_score', 'outlier_score']
            st.dataframe(
                fraud_scores[display_cols].head(50).style.background_gradient(
                    subset=['fraud_score'], cmap='Reds'
                ),
                use_container_width=True
            )
    else:
        st.warning("⚠️ Insufficient data for multi-modal fraud analysis")
    
    # ---------------------------------------------------------
    # SECTION 2: CHURN PREDICTION
    # ---------------------------------------------------------
    st.markdown("---")
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        st.subheader("📉 Pincode Churn Risk Prediction")
    with col2:
        with st.popover("ℹ️"):
            st.markdown("""
            **What This Chart Shows:**
            - **Predictive churn probability (0-100%)** for pincodes likely to become inactive in next 30 days
            - **Top 20 at-risk pincodes** requiring immediate intervention
            - **Two visualizations:** Bar chart (top risks) + Scatter plot (root cause analysis)
            - **Scatter plot insights:** X-axis = days inactive, Y-axis = activity trend, Size = churn risk
            
            **Key Insights from This Analysis:**
            - Pincodes in **top-right quadrant** (long inactivity + declining trend) are highest risk
            - **CRITICAL risk (≥70%)** pincodes show multiple failure signals simultaneously
            - Early intervention prevents permanent operational loss
            - Scatter plot reveals if churn is due to **sudden shutdown** (X-axis) or **gradual decline** (Y-axis)
            - Most effective intervention window: **7-14 days of inactivity**
            - Expandable table shows all risk factors for targeted intervention planning
            
            ---
            
            **Calculation Methodology:**
            
            **Churn Probability Formula (Additive Risk Scoring):**
            ```
            Churn Risk = InactivityScore + TrendScore + FrequencyScore + VolumeScore
            Capped at 100%
            ```
            
            **Feature 1: Days Since Last Activity**
            - **Calculation:** `(today - last_active_date).days`
            - **Scoring:**
              - >14 days: +40 points (long absence, critical risk)
              - 7-14 days: +20 points (concerning gap)
              - 3-7 days: +10 points (slight concern)
              - <3 days: 0 points (active)
            - **Rationale:** Inactivity duration is strongest predictor of permanent cessation
            - **Why thresholds:** Based on operational patterns - 2 weeks indicates systematic issue
            
            **Feature 2: 7-Day Activity Trend**
            - **Calculation:** `((last_7d_activity - prev_7d_activity) / prev_7d_activity) × 100`
            - **Scoring:**
              - Decline >50%: +30 points (severe drop)
              - Decline 20-50%: +15 points (moderate drop)
              - Otherwise: 0 points
            - **Rationale:** Rapid decline signals imminent shutdown
            - **Why 7 days:** Short enough to catch recent changes, long enough to avoid noise
            
            **Feature 3: Historical Activity Frequency**
            - **Calculation:** `(active_days / total_days_since_first_activity)`
            - **Scoring:**
              - Frequency <0.2 (active <20% of time): +20 points
              - Frequency 0.2-0.4: +10 points
              - Otherwise: 0 points
            - **Rationale:** Sporadic operation history indicates unreliable pincode
            - **Why frequency:** Normalizes across different operational histories
            
            **Feature 4: Volume Decline Ratio**
            - **Calculation:** `(recent_avg_volume_7d / overall_avg_volume)`
            - **Scoring:** If ratio <0.3: +10 points (70% volume drop)
            - **Rationale:** Severe volume reduction even if still technically active
            - **Why 0.3 threshold:** 70% drop indicates fundamental operational problem
            
            ---
            
            **Action Items:**
            - **CRITICAL (70-100%):** Deploy mobile van within 48 hours, contact local operators immediately
            - **HIGH (50-69%):** Schedule field visit within 1 week, investigate equipment/staffing issues
            - **MEDIUM (30-49%):** Add to monitoring list, prepare contingency mobile unit
            - **Root cause investigation:** Check scatter plot position
              - High X (long inactivity): Equipment failure, staff attrition, or closure
              - Low Y (declining trend): Demand drop, competition, or operational inefficiency
            - **Preventive:** Contact pincodes at 7-day inactivity mark before reaching critical
            - **Resource allocation:** Use churn probability to prioritize limited mobile van deployments
            """)
    
    # Use preloaded data
    churn_predictions = preloaded_charts.get('churn_predictions', pd.DataFrame())
    
    if not churn_predictions.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            critical_churn = len(churn_predictions[churn_predictions['risk_level'] == 'CRITICAL'])
            st.metric("🔴 Critical Churn Risk", critical_churn)
        with col2:
            high_churn = len(churn_predictions[churn_predictions['risk_level'] == 'HIGH'])
            st.metric("🟠 High Churn Risk", high_churn)
        with col3:
            avg_churn = churn_predictions['churn_probability'].mean()
            st.metric("Average Churn Probability", f"{avg_churn:.1f}%")
        with col4:
            inactive_soon = len(churn_predictions[churn_predictions['churn_probability'] >= 70])
            st.metric("Likely Inactive (30d)", inactive_soon)
        
        # Visualization: Top 20 At-Risk Pincodes
        st.markdown("#### 🔻 Top 20 Pincodes at Risk of Becoming Inactive")
        st.markdown("*Higher bars = more likely to stop operations in next 30 days*")
        
        top_churn = churn_predictions.head(20)
        
        fig_churn = px.bar(
            top_churn,
            x='pincode',
            y='churn_probability',
            color='risk_level',
            color_discrete_map={'CRITICAL': '#c0392b', 'HIGH': '#d35400', 'MEDIUM': '#f39c12', 'LOW': '#16a085'},
            hover_data=['district', 'days_since_last_activity', 'activity_trend_7d', 'activity_frequency'],
            labels={'churn_probability': 'Churn Probability (%)', 'pincode': 'Pincode'},
            title="",
            template="plotly",
            height=450,
            text='churn_probability'
        )
        fig_churn.update_traces(width=0.8)
        fig_churn.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
        fig_churn.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Critical Risk (70%)", annotation_position="right")
        fig_churn.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="High Risk (50%)", annotation_position="right")
        fig_churn.update_xaxes(tickangle=45, title="Pincode")
        fig_churn.update_yaxes(title="Probability of Becoming Inactive (%)", range=[0, 105])
        fig_churn.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_churn, use_container_width=True)
        
        # Risk Matrix: 2x2 Classification
        st.markdown("#### 🎯 Risk Classification Matrix: Root Cause Analysis")
        st.markdown("*This matrix shows WHY pincodes are at risk - helps target the right intervention*")
        
        # Create risk categories
        churn_display = churn_predictions.head(100).copy()
        churn_display['inactivity_category'] = churn_display['days_since_last_activity'].apply(
            lambda x: 'Long Inactivity (>7 days)' if x > 7 else 'Recently Active (<7 days)'
        )
        churn_display['trend_category'] = churn_display['activity_trend_7d'].apply(
            lambda x: 'Declining Trend' if x < -10 else 'Stable/Growing'
        )
        churn_display['quadrant'] = churn_display['inactivity_category'] + ' + ' + churn_display['trend_category']
        
        # Color by quadrant risk
        quadrant_colors = {
            'Long Inactivity (>7 days) + Declining Trend': '#e74c3c',  # Red - Most Critical
            'Long Inactivity (>7 days) + Stable/Growing': '#e67e22',   # Orange - High
            'Recently Active (<7 days) + Declining Trend': '#f39c12',  # Yellow - Medium
            'Recently Active (<7 days) + Stable/Growing': '#27ae60'    # Green - Low
        }
        
        fig_matrix = px.scatter(
            churn_display,
            x='days_since_last_activity',
            y='activity_trend_7d',
            size='churn_probability',
            color='quadrant',
            color_discrete_map=quadrant_colors,
            hover_data=['pincode', 'district', 'churn_probability', 'activity_frequency'],
            labels={'days_since_last_activity': 'Days Since Last Activity', 
                   'activity_trend_7d': '7-Day Activity Change (%)',
                   'quadrant': 'Risk Category'},
            title="",
            template="plotly",
            height=550
        )
        
        # Add cleaner quadrant dividers only
        fig_matrix.add_hline(y=-10, line_dash="dash", line_color="rgba(150,150,150,0.5)", line_width=1.5)
        fig_matrix.add_vline(x=7, line_dash="dash", line_color="rgba(150,150,150,0.5)", line_width=1.5)
        fig_matrix.add_hline(y=0, line_dash="dot", line_color="rgba(100,100,100,0.3)")
        
        fig_matrix.update_xaxes(title="Days Inactive", showgrid=True, gridcolor="rgba(128,128,128,0.2)")
        fig_matrix.update_yaxes(title="Activity Trend %", showgrid=True, gridcolor="rgba(128,128,128,0.2)")
        fig_matrix.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_matrix, use_container_width=True)
        
        # Add interpretation guide
        with st.expander("📖 How to Read This Matrix"):
            st.markdown("""
            **Bubble Size** = Churn probability (bigger = higher risk)
            
            **Quadrants Explained:**
            - **🔴 Bottom-Right (RED):** Long inactive + declining trend = CRITICAL - likely permanent closure
                - *Action:* Deploy mobile van immediately, investigate equipment/staff issues
            
            - **🟠 Top-Right (ORANGE):** Long inactive but was stable = HIGH - sudden shutdown event
                - *Action:* Emergency field visit, check for equipment failure or staff attrition
            
            - **🟡 Bottom-Left (YELLOW):** Active but declining = MEDIUM - early warning signs
                - *Action:* Preventive contact, monitor for next 7 days
            
            - **🟢 Top-Left (GREEN):** Active and stable = LOW - normal operations
                - *Action:* Routine monitoring only
            """)
        
        st.markdown("---")
        
        # Detailed table
        with st.expander("📋 View Detailed Churn Predictions (Top 50)"):
            display_cols = ['pincode', 'district', 'churn_probability', 'risk_level', 
                          'days_since_last_activity', 'activity_trend_7d', 'activity_frequency', 
                          'avg_gap_days', 'recent_volume']
            st.dataframe(
                churn_predictions[display_cols].head(50).style.background_gradient(
                    subset=['churn_probability'], cmap='OrRd'
                ),
                use_container_width=True
            )
    else:
        st.warning("⚠️ Insufficient historical data for churn prediction")
    
    # ---------------------------------------------------------
    # SECTION 3: BIOMETRIC COMPLIANCE PREDICTION
    # ---------------------------------------------------------
    st.markdown("---")
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        st.subheader("🔐 Biometric Update Compliance Prediction")
    with col2:
        with st.popover("ℹ️"):
            st.markdown("""
            **What This Chart Shows:**
            - **Bottom 20 pincodes** with lowest predicted biometric compliance rates
            - **Predicted completion rate (0-100%)** = expected % of enrollments that will complete updates
            - **Two visualizations:** Bar chart (worst performers) + Scatter plot (backlog analysis)
            - **Scatter plot:** X-axis = pending volume, Y-axis = predicted rate, Size = total enrollment
            
            **Key Insights from This Analysis:**
            - **CRITICAL pincodes (<40% predicted rate)** need urgent SMS/mobile van campaigns
            - Large **pending backlogs** shown in scatter plot indicate resource allocation priorities
            - Scatter plot bottom-right quadrant (high pending + low rate) = **highest priority**
            - "Avg Days to Complete" metric reveals distance/awareness issues (>60 days = friction)
            - Campaign Prioritization table shows **total target population** for resource planning
            - Historical vs predicted gap shows impact of volume/time adjustments
            
            ---
            
            **Calculation Methodology:**
            
            **Predicted Completion Rate Formula:**
            ```
            Predicted Rate = Historical Rate × Volume Adjustment × Time Adjustment
            Capped at 0-100%
            ```
            
            **Step 1: Historical Baseline**
            - **Calculation:** `(total_biometric / total_enrollment) × 100`
            - **Rationale:** Past performance is best predictor of future compliance
            - **Handles missing data:** Pincodes with zero biometric get 0% historical rate
            
            **Step 2: Volume Adjustment**
            - **Condition:** If `enrollment_volume > 100`
            - **Adjustment:** Multiply by 0.9 (reduce prediction by 10%)
            - **Rationale:** High-volume centers face capacity constraints and resource limitations
            - **Why 100 threshold:** Empirical cutoff where operational overhead becomes significant
            
            **Step 3: Time Lag Adjustment**
            - **Calculation:** `avg_days_to_complete = (avg_bio_date - avg_enrol_date).days`
            - **Condition:** If `avg_days > 60`
            - **Adjustment:** Multiply by 0.85 (reduce prediction by 15%)
            - **Rationale:** Long delays indicate systematic issues (distance, awareness, capacity)
            - **Why 60 days:** Standard biometric update window; exceeding indicates friction
            - **Handles NaT:** Days set to 0 if biometric date missing
            
            **Pending Enrollments Calculation:**
            ```
            Pending = max(0, total_enrollment - total_biometric)
            ```
            
            ---
            
            **Action Items:**
            - **CRITICAL (<40%):** Launch SMS/WhatsApp campaign to all pending enrollees within 72 hours
            - **Deploy mobile vans:** Use pending_enrollments column to allocate vans to highest-volume pincodes
            - **HIGH (40-59%):** Extend center hours, add weekend shifts to clear backlog
            - **Target messaging:** High avg_days (>60) = awareness campaign; Low predicted rate = accessibility issue
            - **Use Campaign Prioritization table:** Shows recommended mobile vans + total SMS target population
            - **Budget planning:** Total pending × cost_per_update = campaign budget estimate
            - **Monitor weekly:** Re-run predictions after campaign to measure effectiveness
            """)
    
    # Use preloaded data
    compliance_predictions = preloaded_charts.get('compliance_predictions', pd.DataFrame())
    
    if not compliance_predictions.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            critical_compliance = len(compliance_predictions[compliance_predictions['risk_level'] == 'CRITICAL'])
            st.metric("🚨 Critical Compliance Risk", critical_compliance)
        with col2:
            avg_predicted = compliance_predictions['predicted_completion_rate'].mean()
            st.metric("Avg Predicted Compliance", f"{avg_predicted:.1f}%")
        with col3:
            total_pending = compliance_predictions['pending_enrollments'].sum()
            st.metric("Total Pending Updates", f"{total_pending:,}")
        with col4:
            avg_days = compliance_predictions['avg_days_to_complete'].mean()
            st.metric("Avg Days to Complete", f"{avg_days:.1f}")
        
        # Visualization: Bottom 20 (Lowest Compliance)
        st.markdown("#### ⚠️ Bottom 20 Pincodes: Lowest Predicted Compliance Rates")
        st.markdown("*These pincodes have the lowest expected biometric update completion rates - prioritize for intervention*")
        
        bottom_compliance = compliance_predictions.head(20)
        
        # Create a clearer comparison chart
        col1, col2 = st.columns(2)
        
        with col1:
            fig_compliance = px.bar(
                bottom_compliance,
                x='pincode',
                y='predicted_completion_rate',
                color='risk_level',
                color_discrete_map={'CRITICAL': '#c0392b', 'HIGH': '#e67e22', 'MEDIUM': '#f39c12', 'LOW': '#27ae60'},
                hover_data=['district', 'pending_enrollments', 'avg_days_to_complete', 'historical_completion_rate'],
                labels={'predicted_completion_rate': 'Predicted Rate (%)', 'pincode': 'Pincode'},
                title="Predicted Completion Rate",
                template="plotly",
                height=450,
                text='predicted_completion_rate'
            )
            fig_compliance.update_traces(width=0.8)
            fig_compliance.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_compliance.add_hline(y=40, line_dash="dash", line_color="red", annotation_text="CRITICAL", annotation_position="right")
            fig_compliance.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="HIGH", annotation_position="right")
            fig_compliance.update_xaxes(tickangle=45, title="Pincode")
            fig_compliance.update_yaxes(title="Completion Rate (%)", range=[0, 105])
            fig_compliance.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_compliance, use_container_width=True)
        
        with col2:
            # Show pending volumes for same pincodes
            fig_pending = px.bar(
                bottom_compliance,
                x='pincode',
                y='pending_enrollments',
                color='risk_level',
                color_discrete_map={'CRITICAL': '#c0392b', 'HIGH': '#e67e22', 'MEDIUM': '#f39c12', 'LOW': '#27ae60'},
                hover_data=['district', 'predicted_completion_rate', 'avg_days_to_complete'],
                labels={'pending_enrollments': 'Pending Updates', 'pincode': 'Pincode'},
                title="Pending Biometric Updates (Backlog Size)",
                template="plotly",
                height=450,
                text='pending_enrollments'
            )
            fig_pending.update_traces(width=0.8)
            fig_pending.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig_pending.update_xaxes(tickangle=45, title="Pincode")
            fig_pending.update_yaxes(title="Pending Updates Count")
            fig_pending.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pending, use_container_width=True)
        
        st.markdown("---")
        
        # Priority Matrix: Backlog vs Compliance
        st.markdown("#### 🎯 Priority Matrix: Which Pincodes Need Resources Most?")
        st.markdown("*Bottom-right quadrant = low completion + high backlog = HIGHEST PRIORITY for mobile vans*")
        
        # Create quadrant categories
        compliance_display = compliance_predictions.head(100).copy()
        median_pending = compliance_display['pending_enrollments'].median()
        compliance_display['backlog_category'] = compliance_display['pending_enrollments'].apply(
            lambda x: f'High Backlog (>{int(median_pending)})' if x > median_pending else f'Low Backlog (<{int(median_pending)})'
        )
        compliance_display['rate_category'] = compliance_display['predicted_completion_rate'].apply(
            lambda x: 'Low Rate (<50%)' if x < 50 else 'Moderate Rate (≥50%)'
        )
        
        fig_backlog = px.scatter(
            compliance_display,
            x='pending_enrollments',
            y='predicted_completion_rate',
            size='enrollment_volume',
            color='risk_level',
            color_discrete_map={'CRITICAL': '#e74c3c', 'HIGH': '#e67e22', 'MEDIUM': '#f39c12', 'LOW': '#27ae60'},
            hover_data=['pincode', 'district', 'avg_days_to_complete', 'historical_completion_rate'],
            labels={'pending_enrollments': 'Pending Enrollments (Backlog Size)', 
                   'predicted_completion_rate': 'Predicted Completion Rate (%)'},
            title="",
            template="plotly",
            height=550
        )
        
        # Add cleaner quadrant dividers
        fig_backlog.add_hline(y=50, line_dash="dash", line_color="rgba(150,150,150,0.5)", line_width=1.5)
        fig_backlog.add_vline(x=median_pending, line_dash="dash", line_color="rgba(150,150,150,0.5)", line_width=1.5)
        
        fig_backlog.update_xaxes(title="Pending Enrollments", showgrid=True, gridcolor="rgba(128,128,128,0.2)")
        fig_backlog.update_yaxes(title="Predicted Rate %", showgrid=True, gridcolor="rgba(128,128,128,0.2)")
        fig_backlog.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        
        st.plotly_chart(fig_backlog, use_container_width=True)
        
        # Add interpretation guide
        with st.expander("📖 How to Use This Priority Matrix"):
            st.markdown("""
            **Reading the Chart:**
            - **Bubble Size** = Total enrollment volume (bigger = more operational capacity)
            - **X-axis (right = worse):** More pending updates = larger backlog
            - **Y-axis (down = worse):** Lower predicted rate = less likely to clear backlog
            
            **Quadrant Actions:**
            - **🔴 Bottom-Right (HIGHEST PRIORITY):** Low rate + High backlog
                - *Deploy mobile vans immediately* - they won't clear backlog on their own
                - *Calculate vans needed:* pending ÷ 100 per van per week
            
            - **🟠 Top-Right:** Good rate + High backlog
                - *Extend center hours* or *add weekend shifts*
                - They're doing well but need capacity boost
            
            - **🟡 Bottom-Left:** Low rate + Low backlog
                - *SMS/WhatsApp awareness campaign* first
                - Monitor for 2 weeks before deploying vans
            
            - **🟢 Top-Left:** Good rate + Low backlog
                - *Routine monitoring* only - no intervention needed
            """)
        
        # Campaign prioritization
        st.markdown("#### 🎯 Campaign Prioritization")
        high_priority = compliance_predictions[compliance_predictions['predicted_completion_rate'] < 50]
        
        if not high_priority.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**High Priority Intervention Targets:**")
                st.dataframe(
                    high_priority[['pincode', 'district', 'predicted_completion_rate', 
                                 'pending_enrollments']].head(10),
                    use_container_width=True
                )
            with col2:
                total_target = high_priority['pending_enrollments'].sum()
                st.metric("Total Target Population", f"{total_target:,}")
                st.markdown(f"""
                **Recommended Actions:**
                - Deploy {len(high_priority)} mobile vans
                - Launch SMS campaign to {total_target:,} individuals
                - Extend center hours in flagged areas
                """)
        
        # Detailed table
        with st.expander("📋 View Detailed Compliance Predictions (All)"):
            display_cols = ['pincode', 'district', 'predicted_completion_rate', 'historical_completion_rate',
                          'risk_level', 'pending_enrollments', 'avg_days_to_complete', 'enrollment_volume']
            st.dataframe(
                compliance_predictions[display_cols].style.background_gradient(
                    subset=['predicted_completion_rate'], cmap='RdYlGn'
                ),
                use_container_width=True
            )
    else:
        st.warning("⚠️ Insufficient data for compliance prediction")
    
    # ---------------------------------------------------------
    # SECTION 4: CROSS-DATASET ANOMALY DETECTION
    # ---------------------------------------------------------
    st.markdown("---")
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        st.subheader("🔗 Cross-Dataset Integrity Analysis")
    with col2:
        with st.popover("ℹ️"):
            st.markdown("""
            **What This Chart Shows:**
            - **3D scatter plot** visualizing relationship between enrollment, biometric, and demographic data
            - **Red points = anomalies**, Blue points = normal operations
            - **Anomaly type distribution** bar chart showing most common integrity issues
            - **Top 20 critical anomalies** table ranked by severity score
            
            **Key Insights from This Analysis:**
            - **3D plot patterns:** Points far from diagonal trend line indicate data inconsistencies
            - **Ghost enrollments:** Clusters near enrollment axis but low on biometric axis
            - **Phantom updates:** Points above 1:1 ratio line (biometric > enrollment)
            - **Anomaly type chart** reveals dominant fraud/integrity pattern across system
            - **Severity score** combines anomaly magnitude with operational impact (volume)
            - Most pincodes cluster around normal ratios (0.7-1.2 bio/enroll, 0.3-0.8 demo/enroll)
            - **CRITICAL severity** flags pincodes with multiple simultaneous violations
            
            ---
            
            **Calculation Methodology:**
            
            **Cross-Dataset Ratio Analysis:**
            ```
            Bio_Ratio = total_biometric / total_enrollment
            Demo_Ratio = total_demographic / total_enrollment
            ```
            
            **Anomaly Detection Rules:**
            
            **1. Ghost Enrollments (Fake Registration Pattern)**
            - **Condition:** `Bio_Ratio < 0.3 AND enrollment_volume > 50`
            - **Formula:** Very low biometric update rate on moderate enrollment volume
            - **Rationale:** Suggests enrollments without real people (ghost registrations for quota gaming)
            - **Why 0.3 threshold:** Less than 30% update rate is far below operational norms
            - **Why 50 volume:** Filters noise; statistically significant sample size
            
            **2. Phantom Updates (Data Manipulation Pattern)**
            - **Condition:** `Bio_Ratio > 1.5`
            - **Formula:** More biometric records than enrollments
            - **Rationale:** Mathematically impossible unless duplicate biometric submissions or data dumping
            - **Why 1.5 threshold:** Allows 50% tolerance for legitimate re-updates beyond normal 1:1 ratio
            
            **3. Impossible Demographic Ratios**
            - **Condition:** `Demo_Ratio > 1.2`
            - **Formula:** Demographic records exceed enrollments by >20%
            - **Rationale:** Every demographic update should link to an enrollment; excess indicates data integrity failure
            - **Why 1.2 threshold:** 20% buffer for timing mismatches and legitimate re-updates
            
            **4. Zero Coverage**
            - **Condition:** `Bio_Ratio == 0 AND enrollment_volume > 0`
            - **Formula:** Enrollments exist but no biometric updates recorded
            - **Rationale:** Complete non-compliance or data sync failure
            - **Action:** Immediate investigation of pincode operations
            
            **5. Severe Lag**
            - **Condition:** `Bio_Ratio < 0.15 AND enrollment_volume > 100`
            - **Formula:** Less than 15% update rate on high volume
            - **Rationale:** Indicates capacity crisis or operational shutdown
            - **Why 0.15 & 100:** High-volume centers should maintain at least 15% throughput
            
            **Normal Expected Ratios:**
            - **Bio/Enroll: 0.7 - 1.2** (70-120% biometric coverage is operationally normal)
            - **Demo/Enroll: 0.3 - 0.8** (30-80% demographic updates typical)
            - Rationale: Accounts for timing lags, voluntary updates, and re-submission patterns
            
            ---
            
            **Action Items:**
            - **CRITICAL severity:** Immediate field verification + freeze operations until audit complete
            - **Ghost enrollments:** Cross-reference with physical attendance records, investigate enrollment officers
            - **Phantom updates:** Check for duplicate biometric submissions, review data entry logs
            - **Impossible ratios:** Trigger automated data sync verification across all 3 datasets
            - **Zero coverage:** Emergency technical support deployment, check equipment functionality
            - **Use 3D plot:** Rotate to identify clusters of similar anomalies (may indicate systemic issue)
            - **Anomaly type chart:** Prioritize most frequent pattern for systemic process improvement
            """)
    
    # Use preloaded data
    all_cross_data = preloaded_charts.get('cross_anomalies_all', pd.DataFrame())
    anomalies_only = preloaded_charts.get('cross_anomalies_only', pd.DataFrame())
    
    if not anomalies_only.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_anomalies = len(anomalies_only)
            st.metric("🔍 Anomalies Detected", total_anomalies)
        with col2:
            critical_anomalies = len(anomalies_only[anomalies_only['severity'] == 'CRITICAL'])
            st.metric("🚨 Critical Severity", critical_anomalies)
        with col3:
            ghost_enrollments = len(anomalies_only[anomalies_only['anomaly_type'].str.contains('Ghost', case=False)])
            st.metric("👻 Ghost Enrollments", ghost_enrollments)
        with col4:
            avg_anomaly_score = anomalies_only['anomaly_score'].mean()
            st.metric("Avg Anomaly Score", f"{avg_anomaly_score:.1f}")
        
        # Visualization: 2D Ratio Analysis (Easier to Read than 3D)
        st.markdown("#### 📊 Cross-Dataset Ratio Analysis: Finding Data Integrity Issues")
        st.markdown("*Red points = anomalies that violate normal operational ratios*")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📈 Biometric vs Enrollment", "📈 Demographic vs Enrollment", "🔍 Anomaly Distribution"])
        
        with tab1:
            st.markdown("**Reading This Chart:** Points should cluster around the diagonal line (1:1 ratio). Points far from this line indicate problems.")
            
            # Biometric vs Enrollment scatter
            fig_bio_enroll = px.scatter(
                all_cross_data.head(200),
                x='enroll_total',
                y='bio_total',
                color='is_anomaly',
                color_discrete_map={True: '#e74c3c', False: '#3498db'},
                hover_data=['pincode', 'district', 'anomaly_type', 'bio_enroll_ratio'],
                labels={'enroll_total': 'Total Enrollments', 'bio_total': 'Total Biometric Updates'},
                title="",
                template="plotly",
                height=500
            )
            
            # Add reference lines only
            max_val = max(all_cross_data['enroll_total'].max(), all_cross_data['bio_total'].max())
            fig_bio_enroll.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], 
                                                mode='lines', name='1:1 Ratio',
                                                line=dict(color='rgba(46,204,113,0.6)', dash='dash', width=2)))
            fig_bio_enroll.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val*1.2], 
                                                mode='lines', name='Upper Limit',
                                                line=dict(color='rgba(230,126,34,0.5)', dash='dot', width=1)))
            fig_bio_enroll.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val*0.7], 
                                                mode='lines', name='Lower Limit',
                                                line=dict(color='rgba(230,126,34,0.5)', dash='dot', width=1)))
            
            fig_bio_enroll.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            fig_bio_enroll.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
            fig_bio_enroll.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
            
            st.plotly_chart(fig_bio_enroll, use_container_width=True)
            
            with st.expander("💡 What This Shows"):
                st.markdown("""
                - **Green diagonal line** = Perfect match (every enrollment has exactly one biometric update)
                - **Orange zone** = Normal operations (70-120% coverage accounts for timing lags)
                - **Points above green line** = More biometric than enrollments (phantom updates - investigate for duplicates)
                - **Points below green line** = Fewer biometric than enrollments (low compliance or ghost enrollments)
                - **Red points far from diagonal** = Data integrity violations requiring audit
                """)
        
        with tab2:
            st.markdown("**Reading This Chart:** Demographic updates should be 30-80% of enrollments. Points outside this range need investigation.")
            
            # Demographic vs Enrollment scatter
            fig_demo_enroll = px.scatter(
                all_cross_data.head(200),
                x='enroll_total',
                y='demo_total',
                color='is_anomaly',
                color_discrete_map={True: '#e74c3c', False: '#2ecc71'},
                hover_data=['pincode', 'district', 'anomaly_type', 'demo_enroll_ratio'],
                labels={'enroll_total': 'Total Enrollments', 'demo_total': 'Total Demographic Updates'},
                title="",
                template="plotly",
                height=500
            )
            
            # Add reference lines only
            fig_demo_enroll.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val*0.8], 
                                                mode='lines', name='Upper Normal',
                                                line=dict(color='rgba(46,204,113,0.6)', dash='dash', width=2)))
            fig_demo_enroll.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val*0.3], 
                                                mode='lines', name='Lower Normal',
                                                line=dict(color='rgba(46,204,113,0.6)', dash='dash', width=2)))
            fig_demo_enroll.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val*1.2], 
                                                mode='lines', name='Critical Threshold',
                                                line=dict(color='rgba(231,76,60,0.6)', dash='dot', width=1)))
            
            fig_demo_enroll.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            fig_demo_enroll.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
            fig_demo_enroll.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
            
            st.plotly_chart(fig_demo_enroll, use_container_width=True)
            
            with st.expander("💡 What This Shows"):
                st.markdown("""
                - **Green zone (0.3-0.8)** = Normal demographic update patterns
                - **Points above 1.2 ratio** = Impossible - more demographics than enrollments (data error)
                - **Points near zero** = No demographic updates despite enrollments (operational issue)
                - **Red points** = Violations of expected ratios
                """)
        
        with tab3:
            # Anomaly type distribution
            if not anomalies_only.empty:
                st.markdown("**Most Common Data Integrity Issues:**")
                
                anomaly_type_counts = anomalies_only['anomaly_type'].value_counts().reset_index()
                anomaly_type_counts.columns = ['anomaly_type', 'count']
                
                fig_types = px.bar(
                    anomaly_type_counts,
                    y='anomaly_type',
                    x='count',
                    orientation='h',
                    color='count',
                    color_continuous_scale='Reds',
                    labels={'anomaly_type': 'Anomaly Type', 'count': 'Number of Pincodes'},
                    title="",
                    template="plotly",
                    height=400,
                    text='count'
                )
                fig_types.update_traces(width=0.7)
                fig_types.update_traces(texttemplate='%{text}', textposition='outside')
                st.plotly_chart(fig_types, use_container_width=True)
                
                st.markdown("""
                **What Each Anomaly Type Means:**
                - **Ghost Enrollments:** Fake registrations without real people (quota gaming)
                - **Phantom Updates:** More biometric records than enrollments (data dumping)
                - **Impossible Demographic Ratios:** Demographic > enrollment (database sync error)
                - **Zero Coverage:** Enrollments with no biometric updates (operational failure)
                - **Severe Lag:** Very low update rate on high volume (capacity crisis)
                """)
            else:
                st.success("✅ No anomalies detected - all pincodes show normal data patterns")
        
        st.markdown("---")
        
        # Top anomalies table
        st.markdown("#### 🚨 Top 20 Critical Cross-Dataset Anomalies")
        top_anomalies = anomalies_only.sort_values('anomaly_score', ascending=False).head(20)
        
        display_cols = ['pincode', 'district', 'anomaly_type', 'severity', 'anomaly_score',
                       'enroll_total', 'bio_total', 'demo_total', 'bio_enroll_ratio']
        st.dataframe(
            top_anomalies[display_cols].style.background_gradient(
                subset=['anomaly_score'], cmap='Reds'
            ),
            use_container_width=True
        )
        
        # Detailed view with all anomalies
        with st.expander("📋 View All Detected Anomalies"):
            st.dataframe(
                anomalies_only[display_cols].style.background_gradient(
                    subset=['anomaly_score'], cmap='OrRd'
                ),
                use_container_width=True
            )
    else:
        st.success("✅ No significant cross-dataset anomalies detected. Data integrity appears normal.")
        
        # Show summary stats even when no anomalies
        if not all_cross_data.empty:
            st.markdown("#### Dataset Relationship Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_bio_ratio = all_cross_data['bio_enroll_ratio'].mean()
                st.metric("Avg Bio/Enroll Ratio", f"{avg_bio_ratio:.2f}")
            with col2:
                avg_demo_ratio = all_cross_data['demo_enroll_ratio'].mean()
                st.metric("Avg Demo/Enroll Ratio", f"{avg_demo_ratio:.2f}")
            with col3:
                total_pincodes = len(all_cross_data)
                st.metric("Total Pincodes Analyzed", total_pincodes)
        else:
            st.info("Insufficient cross-dataset data for anomaly analysis")
    
    # ---------------------------------------------------------
    # SECTION 5: PEAK LOAD FORECASTING
    # ---------------------------------------------------------
    st.markdown("---")
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        st.subheader("📅 Peak Load Forecasting: Capacity Planning Intelligence")
    with col2:
        with st.popover("ℹ️"):
            st.markdown("""
            **What This Chart Shows:**
            - **30-day demand forecast** with daily predicted load and confidence intervals (shaded area)
            - **Color-coded markers:** Red = Peak days, Blue = Normal days, Green = Low days
            - **Staffing calendar:** 14-day view with recommended staff levels per day
            - **Peak day alerts:** Lists upcoming high-demand dates requiring extra staffing
            
            **Key Insights from This Analysis:**
            - **Peak days (red markers):** Require 20-30% additional staff scheduled 1 week in advance
            - **Weekday/weekend patterns:** Visible in forecast line (Prophet captures weekly seasonality)
            - **Confidence interval width:** Wider bands = higher uncertainty, narrow bands = reliable predictions
            - **Staffing calendar color coding:** Red rows = critical staffing days, Green = maintenance windows
            - **Trend direction:** Upward/downward slope shows long-term capacity planning needs
            - **Next 7 days most accurate:** Confidence intervals tighten for near-term forecasts
            - **Low-load days (green):** Optimal for equipment maintenance, staff training, or inventory tasks
            
            ---
            
            **Calculation Methodology:**
            
            **Time Series Forecasting with Prophet:**
            ```
            Model: Facebook Prophet (Additive Time Series Decomposition)
            Forecast Horizon: 30 days ahead
            Training Data: All historical daily aggregated activity
            ```
            
            **Prophet Configuration:**
            - **Seasonality:** `weekly_seasonality=True` (captures weekday vs weekend patterns)
            - **Confidence Interval:** 80% (uncertainty bounds shown as shaded area)
            - **Growth:** Linear trend with automatic changepoint detection
            - **Rationale:** Prophet handles missing data, holidays, and irregular patterns robustly
            
            **Data Preparation:**
            1. **Aggregate daily totals:** Sum of enrollment + biometric + demographic activity
            2. **Date formatting:** Convert to Prophet's required format (`ds`, `y`)
            3. **Historical window:** All available dates (2025-03-01 to 2025-12-29)
            4. **Why daily:** Balances granularity (catches patterns) with noise reduction
            
            **Load Level Classification:**
            - **Calculation:** `predicted_load_percentile = (predicted_value / historical_max) × 100`
            - **Peak (Red):** Predicted load > 75th percentile of historical distribution
            - **Normal (Blue):** 25th - 75th percentile
            - **Low (Green):** < 25th percentile
            - **Rationale:** Percentile-based (not absolute) accounts for seasonal variations
            
            **Recommended Staffing Formula:**
            ```
            Staff = ceiling(predicted_load / baseline_throughput_per_staff)
            baseline_throughput = historical_avg / historical_avg_staff
            ```
            - **Rationale:** Scales staffing linearly with predicted demand
            - **Buffer:** +1 staff for peak days to handle variance
            
            ---
            
            **Action Items:**
            - **Peak Days (listed below chart):** Pre-schedule additional staff 7 days in advance
            - **Use staffing calendar:** Lock in shift schedules based on recommended_staff column
            - **Low Days (green):** Schedule maintenance to avoid disrupting peak operations
            - **Budget planning:** Sum predicted_load × operational_cost_per_unit for monthly budget
            - **Capacity gaps:** If predicted_load > historical_max, consider temporary mobile units
            - **Weekly reviews:** Update forecast weekly as new data arrives to refine predictions
            - **Training:** Use low-load days for staff training to maintain high service quality on peak days
            """)
    
    # Use preloaded data
    peak_load_forecast = preloaded_charts.get('peak_load_forecast', pd.DataFrame())
    
    if not peak_load_forecast.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            peak_days = len(peak_load_forecast[peak_load_forecast['load_level'] == 'Peak'])
            st.metric("🔴 Peak Load Days", peak_days)
        with col2:
            low_days = len(peak_load_forecast[peak_load_forecast['load_level'] == 'Low'])
            st.metric("🟢 Low Load Days", low_days)
        with col3:
            avg_load = peak_load_forecast['predicted_load'].mean()
            st.metric("Avg Predicted Load", f"{avg_load:.0f}")
        with col4:
            max_staff = peak_load_forecast['recommended_staff'].max()
            st.metric("Max Staff Needed", int(max_staff))
        
        # Visualization: Forecast Line Chart with Load Levels
        fig_forecast = go.Figure()
        
        # Add prediction line
        fig_forecast.add_trace(go.Scatter(
            x=peak_load_forecast['date'],
            y=peak_load_forecast['predicted_load'],
            mode='lines+markers',
            name='Predicted Load',
            line=dict(color='#3498db', width=2),
            marker=dict(
                size=8,
                color=peak_load_forecast['load_level'].map({
                    'Peak': '#e74c3c',
                    'Normal': '#3498db',
                    'Low': '#27ae60'
                })
            ),
            hovertemplate='<b>%{x}</b><br>Load: %{y:.0f}<br><extra></extra>'
        ))
        
        # Add confidence interval
        fig_forecast.add_trace(go.Scatter(
            x=peak_load_forecast['date'],
            y=peak_load_forecast['upper_bound'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_forecast.add_trace(go.Scatter(
            x=peak_load_forecast['date'],
            y=peak_load_forecast['lower_bound'],
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(52, 152, 219, 0.2)',
            fill='tonexty',
            name='Confidence Interval',
            hoverinfo='skip'
        ))
        
        fig_forecast.update_layout(
            title="30-Day Load Forecast with Peak Detection",
            xaxis_title='Date',
            yaxis_title='Predicted Daily Load',
            template="plotly_white",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Staffing Calendar View
        st.markdown("#### 📋 Staffing Recommendations Calendar")
        
        # Create calendar-style table
        peak_load_forecast['date_str'] = pd.to_datetime(peak_load_forecast['date']).dt.strftime('%Y-%m-%d')
        calendar_view = peak_load_forecast[['date_str', 'day_name', 'predicted_load', 
                                           'load_level', 'recommended_staff']].head(14)
        
        # Color coding function
        def highlight_load_level(row):
            if row['load_level'] == 'Peak':
                return ['background-color: #ffcdd2'] * len(row)
            elif row['load_level'] == 'Low':
                return ['background-color: #c8e6c9'] * len(row)
            else:
                return ['background-color: #e3f2fd'] * len(row)
        
        st.dataframe(
            calendar_view.style.apply(highlight_load_level, axis=1).format({
                'predicted_load': '{:.0f}',
                'recommended_staff': '{:.0f}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Peak days alert
        peak_days_list = peak_load_forecast[peak_load_forecast['load_level'] == 'Peak']
        if not peak_days_list.empty:
            st.warning(f"⚠️ **{len(peak_days_list)} Peak Days Detected** - Schedule additional staff for:")
            for idx, row in peak_days_list.head(7).iterrows():
                st.markdown(f"- **{row['date'].strftime('%A, %b %d')}**: {row['predicted_load']:.0f} expected load → {int(row['recommended_staff'])} staff recommended")
        
        # Detailed table
        with st.expander("📋 View Complete 30-Day Forecast"):
            display_cols = ['date_str', 'day_name', 'predicted_load', 'load_level', 
                          'recommended_staff', 'lower_bound', 'upper_bound']
            st.dataframe(
                peak_load_forecast[display_cols].style.format({
                    'predicted_load': '{:.0f}',
                    'lower_bound': '{:.0f}',
                    'upper_bound': '{:.0f}',
                    'recommended_staff': '{:.0f}'
                }),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.warning("⚠️ Insufficient historical data for peak load forecasting")
    
    # ---------------------------------------------------------
    # SECTION 6: AGE GROUP TRANSITION PREDICTION
    # ---------------------------------------------------------
    st.markdown("---")
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        st.subheader("👶➡️👨 Age Group Transition Prediction")
    with col2:
        with st.popover("ℹ️"):
            st.markdown("""
            **What This Chart Shows:**
            - **Top 20 districts** ranked by urgency score (combination of cohort size + time until transition)
            - **Bar chart color coding:** Dark red (OVERDUE), Red (CRITICAL), Orange (HIGH), Yellow (MEDIUM), Green (LOW)
            - **Scatter plot:** X-axis = days until transition, Y-axis = cohort size, Size = urgency
            - **Campaign planning table:** Shows immediate-action districts with campaign start dates
            
            **Key Insights from This Analysis:**
            - **OVERDUE districts (dark red):** Cohort already 18+ but documents not updated - immediate backlog
            - **CRITICAL districts (red):** Transition within 1 year - campaigns must launch NOW
            - **Scatter plot quadrants:**
              - Top-left (large cohort, near transition) = highest resource priority
              - Bottom-right (small cohort, far transition) = low priority monitoring
            - **National monthly transition rate:** Total individuals turning 18 each month across all districts
            - **Campaign start dates:** Pre-calculated T-180 day launch dates for awareness campaigns
            - **Required monthly capacity:** Infrastructure needed to process transitions without backlog
            - Vertical lines (1yr, 2yr) help visualize planning horizons on scatter plot
            
            ---
            
            **Calculation Methodology:**
            
            **Age Group Transition Prediction (Cohort Survival Analysis):**
            ```
            Objective: Estimate when current 5-17 year-olds will turn 18
            Method: Age-based extrapolation with district-level grouping
            ```
            
            **Step 1: Identify Current Cohort**
            - **Calculation:** Filter demographic data for `age_group == '5-17'`
            - **Aggregation:** Group by district, sum total individuals
            - **Output:** `current_5_17_cohort` per district
            - **Rationale:** District-level planning (capacity varies by geography)
            
            **Step 2: Calculate Average Age of Cohort**
            - **Assumption:** Since exact birth dates unavailable, use midpoint of age group
            - **Formula:** `avg_age = (5 + 17) / 2 = 11 years`
            - **Rationale:** Statistical midpoint for cohort-level planning (individuals vary, but aggregate is predictable)
            
            **Step 3: Calculate Days Until Transition to 18+**
            - **Formula:** `days_until_18 = (18 - avg_age) × 365 = 7 × 365 = 2,555 days`
            - **Rationale:** Average member of 5-17 cohort is ~7 years from turning 18
            - **Conversion:** `.astype(int)` ensures whole days (no fractional days)
            
            **Step 4: Estimate Transition Date**
            - **Calculation:** `estimated_transition_date = last_activity_date + timedelta(days=days_until_18)`
            - **Formula:** Uses district's most recent activity date as baseline
            - **Rationale:** Accounts for data collection timing (newer data = more recent baseline)
            - **Output:** Target date for campaign launch (6 months prior)
            
            **Step 5: Calculate Monthly Transition Rate**
            - **Formula:** `monthly_rate = current_cohort / (days_until_18 / 30)`
            - **Interpretation:** Average number of individuals turning 18 each month
            - **Rationale:** Capacity planning metric (staff/infrastructure needed monthly)
            - **Example:** 1,000 cohort / 85 months = ~11.7 transitions/month
            
            **Priority Classification (Action Urgency):**
            - **OVERDUE (days ≤ 0):** Cohort already 18+ but not updated - immediate backlog processing
            - **CRITICAL (0 < days ≤ 365):** Transition within 1 year - launch campaigns NOW
            - **HIGH (365 < days ≤ 730):** 1-2 years out - begin planning and infrastructure prep
            - **MEDIUM (730 < days ≤ 1095):** 2-3 years out - budget allocation phase
            - **LOW (days > 1095):** 3+ years out - monitor and long-term planning
            
            ---
            
            **Action Items:**
            - **OVERDUE/CRITICAL (shown in Campaign Planning table):** Launch SMS campaigns on listed start dates
            - **Resource allocation:** Use monthly_transition_rate to calculate staff/infrastructure needs per district
            - **Mobile van deployment:** Prioritize top-left quadrant of scatter plot (high urgency + large cohort)
            - **Budget estimation:** Total cohort × update_cost = campaign budget per district
            - **Timeline execution:**
              - **T-180 days:** SMS/radio awareness campaign
              - **T-90 days:** Mobile van pre-positioning
              - **T-30 days:** Staff ramp-up
              - **T-day:** Begin processing transitions
            - **Capacity building:** Sum required_monthly_capacity across critical districts for infrastructure planning
            - **Quarterly reviews:** Re-run predictions to adjust as cohort ages and new data arrives
            """)
    
    # Use preloaded data
    transition_predictions = preloaded_charts.get('transition_predictions', pd.DataFrame())
    
    if not transition_predictions.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            critical_districts = len(transition_predictions[transition_predictions['priority'].isin(['CRITICAL', 'OVERDUE'])])
            st.metric("🚨 Critical Districts", critical_districts)
        with col2:
            total_cohort = transition_predictions['current_5_17_cohort'].sum()
            st.metric("Total 5-17 Cohort", f"{total_cohort:,}")
        with col3:
            total_monthly_rate = transition_predictions['monthly_transition_rate'].sum()
            st.metric("National Monthly Transition", f"{total_monthly_rate:.0f}")
        with col4:
            avg_urgency = transition_predictions['urgency_score'].mean()
            st.metric("Avg Urgency Score", f"{avg_urgency:.1f}")
        
        # Visualization: Urgency Score by District
        top_transitions = transition_predictions.head(20)
        
        fig_transitions = px.bar(
            top_transitions,
            x='district',
            y='urgency_score',
            color='priority',
            color_discrete_map={'OVERDUE': '#8b0000', 'CRITICAL': '#c0392b', 
                              'HIGH': '#e67e22', 'MEDIUM': '#f39c12', 'LOW': '#27ae60'},
            hover_data=['current_5_17_cohort', 'days_until_transition', 'monthly_transition_rate'],
            labels={'urgency_score': 'Urgency Score', 'district': 'District'},
            title="Top 20 Districts: Age Transition Urgency",
            template="plotly",
            height=500
        )
        fig_transitions.update_traces(width=0.8)
        fig_transitions.update_xaxes(tickangle=45)
        fig_transitions.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_transitions, use_container_width=True)
        
        # Timeline view
        st.markdown("#### 📅 Transition Timeline")
        fig_timeline = px.scatter(
            transition_predictions.head(50),
            x='days_until_transition',
            y='current_5_17_cohort',
            size='urgency_score',
            color='priority',
            color_discrete_map={'OVERDUE': '#8b0000', 'CRITICAL': '#c0392b', 
                              'HIGH': '#e67e22', 'MEDIUM': '#f39c12', 'LOW': '#27ae60'},
            hover_data=['district', 'predicted_transition_date', 'monthly_transition_rate'],
            labels={'days_until_transition': 'Days Until Transition', 
                   'current_5_17_cohort': 'Cohort Size'},
            title="Cohort Size vs Time to Transition",
            template="plotly_white",
            height=500
        )
        fig_timeline.add_vline(x=365, line_dash="dash", line_color="red", annotation_text="1 Year")
        fig_timeline.add_vline(x=730, line_dash="dash", line_color="orange", annotation_text="2 Years")
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Campaign Planning
        st.markdown("#### 🎯 Campaign Planning Recommendations")
        critical_campaigns = transition_predictions[
            transition_predictions['priority'].isin(['CRITICAL', 'OVERDUE'])
        ]
        
        if not critical_campaigns.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Immediate Action Required:**")
                st.dataframe(
                    critical_campaigns[['district', 'current_5_17_cohort', 'days_until_transition', 
                                      'campaign_start_date']].head(10),
                    use_container_width=True,
                    hide_index=True
                )
            with col2:
                total_critical_cohort = critical_campaigns['current_5_17_cohort'].sum()
                required_capacity = critical_campaigns['required_monthly_capacity'].sum()
                st.metric("Critical Cohort Size", f"{total_critical_cohort:,}")
                st.metric("Required Monthly Capacity", f"{required_capacity:.0f}")
                st.markdown(f"""
                **Campaign Strategy:**
                - Target {len(critical_campaigns)} districts
                - Reach {total_critical_cohort:,} individuals
                - Build capacity for {required_capacity:.0f} updates/month
                - Launch campaigns on listed start dates
                """)
        else:
            st.success("✅ No critical transitions in near term")
        
        # Detailed table
        with st.expander("📋 View All District Transition Predictions"):
            display_cols = ['district', 'current_5_17_cohort', 'avg_enrollment_date', 
                          'predicted_transition_date', 'days_until_transition', 
                          'priority', 'urgency_score', 'monthly_transition_rate', 
                          'campaign_start_date']
            st.dataframe(
                transition_predictions[display_cols].style.background_gradient(
                    subset=['urgency_score'], cmap='OrRd'
                ).format({
                    'current_5_17_cohort': '{:,.0f}',
                    'urgency_score': '{:.1f}',
                    'monthly_transition_rate': '{:.1f}'
                }),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.warning("⚠️ Insufficient cohort data for transition prediction")


# ==========================================
# 🤖 INTELLIGENT CHATBOT (Place at bottom of app.py)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("💬 AI Analyst")

# Initialize Session State for Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize Bot (Only once) - NO API KEY PASSED HERE
if "bot" not in st.session_state:
    # We don't pass api_key anymore, it's inside the class
    st.session_state.bot = AadhaarChatbot(df_enrol, df_bio, df_demo)

# Chat UI
user_input = st.sidebar.text_area("Ask about charts or data:", height=70)

if st.sidebar.button("Ask AI"):
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.spinner("Thinking..."):
            try:
                # PASS THE CURRENT PAGE TO THE BOT
                response = st.session_state.bot.ask(user_input, current_page=page)
            except Exception as e:
                response = f"I encountered an error: {e}"
            
        # Add bot response to history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# Display History (Latest on top)
for msg in reversed(st.session_state.chat_history):
    if msg["role"] == "user":
        st.sidebar.write(f"**You:** {msg['content']}")
    else:
        st.sidebar.info(f"**Bot:** {msg['content']}")