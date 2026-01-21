"""
Security & Integrity Cache Generator
Pre-generates all visualizations for Page 5 (Security & Integrity) and saves to JSON cache.
Run this script whenever data is updated to refresh the security cache.

Usage:
    python generate_security_cache.py
"""

import pandas as pd
import numpy as np
import json
import os
import glob
import math
from datetime import datetime
from data_preprocessing import preprocess_dataframe
import plotly
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')


# ==========================================
# SECURITY ANALYTICS FUNCTIONS
# ==========================================

def get_benfords_law_global(df):
    """Global Benford's Law analysis across all data"""
    # Get all age columns
    age_cols = [col for col in df.columns if col.startswith('age_')]
    
    # Calculate total activity per record
    df['total_activity'] = df[age_cols].sum(axis=1)
    
    # Get leading digits (excluding zeros)
    data = df[df['total_activity'] > 0]['total_activity']
    
    if len(data) < 10:
        return {"error": "Insufficient data"}
    
    # Extract first digit
    first_digits = data.astype(str).str[0].astype(int)
    actual_freq = first_digits.value_counts(normalize=True).sort_index()
    
    # Benford's theoretical distribution
    theoretical_freq = pd.Series({d: math.log10(1 + 1/d) for d in range(1, 10)})
    
    # Combine for plotting
    results = pd.DataFrame({
        'digit': range(1, 10),
        'actual_freq': [actual_freq.get(d, 0) for d in range(1, 10)],
        'benford_freq': [theoretical_freq.get(d, 0) for d in range(1, 10)]
    })
    
    # Calculate deviation score
    deviation = abs(results['actual_freq'] - results['benford_freq']).sum()
    
    return {
        'distribution': results,
        'deviation_score': float(deviation),
        'is_suspicious': bool(deviation > 0.15)
    }


def get_statistical_outliers(df):
    """Detect anomalous activity spikes over time"""
    age_cols = [col for col in df.columns if col.startswith('age_')]
    
    # Daily aggregation
    daily = df.groupby('date')[age_cols].sum().sum(axis=1).reset_index()
    daily.columns = ['date', 'total_activity']
    
    if len(daily) < 3:
        return pd.DataFrame()
    
    # Calculate statistical thresholds
    mean = daily['total_activity'].mean()
    std = daily['total_activity'].std()
    threshold = mean + (2.5 * std)
    
    daily['is_anomaly'] = daily['total_activity'] > threshold
    daily['threshold'] = threshold
    
    return daily


def get_volatility_analysis(df, top_n=20):
    """Analyze centers with erratic/inconsistent behavior"""
    age_cols = [col for col in df.columns if col.startswith('age_')]
    
    # Calculate variance score per pincode over time
    pincode_stats = df.groupby('pincode').agg({
        age_cols[0]: ['mean', 'std', 'count']
    }).reset_index()
    pincode_stats.columns = ['pincode', 'mean', 'std', 'count']
    
    # Calculate coefficient of variation
    pincode_stats['variance_score'] = (pincode_stats['std'] / pincode_stats['mean']).fillna(0)
    
    # Filter centers with sufficient data and sort by variance
    erratic = pincode_stats[pincode_stats['count'] > 5].sort_values('variance_score', ascending=False).head(top_n)
    
    return erratic


def get_state_variance_data(df):
    """Analyze consistency across states using box plots"""
    age_cols = [col for col in df.columns if col.startswith('age_')]
    
    # Calculate activity per district
    state_data = df.groupby(['state', 'district'])[age_cols].sum().sum(axis=1).reset_index()
    state_data.columns = ['state', 'district', 'total_activity']
    
    return state_data


def create_benford_chart(benford_data):
    """Create Benford's Law Plotly chart"""
    df_ben = benford_data['distribution']
    
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
        line={'color': 'red', 'width': 3}
    ))
    fig_ben.update_layout(
        xaxis_title="Leading Digit",
        yaxis_title="Relative Frequency",
        template="plotly_white",
        height=500,
        xaxis={'tickmode': 'linear', 'tick0': 1, 'dtick': 1}
    )
    
    return fig_ben


def create_outliers_chart(outliers_data):
    """Create Statistical Outliers Plotly chart"""
    anomaly_count = outliers_data['is_anomaly'].sum()
    
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
        line={'color': 'lightgray', 'width': 1},
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
    
    return fig_outliers, int(anomaly_count)


def create_volatility_chart(volatility_data):
    """Create Volatility Analysis Plotly chart"""
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
    
    return fig_volatility


def create_state_variance_chart(state_variance):
    """Create State Variance Box Plot"""
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
    
    return fig_variance


# ==========================================
# DATA LOADING & CACHE GENERATION
# ==========================================


def read_all_csvs_in_folder(folder_name):
    """Finds all .csv files in a folder and merges them into one DataFrame."""
    file_paths = glob.glob(os.path.join(folder_name, "*.csv"))
    
    if not file_paths:
        print(f"⚠️ No CSV files found in {folder_name}")
        return pd.DataFrame()
    
    df_list = []
    for file in file_paths:
        try:
            df_chunk = pd.read_csv(file, low_memory=False)
            df_list.append(df_chunk)
            print(f"  ✓ Loaded {file}")
        except Exception as e:
            print(f"  ✗ Error loading {file}: {e}")
    
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        print(f"  → Combined: {len(combined_df):,} rows")
        return combined_df
    else:
        return pd.DataFrame()


def generate_security_cache():
    """Generate all security analytics and save to JSON cache"""
    
    print("=" * 60)
    print("🚀 SECURITY & INTEGRITY CACHE GENERATOR")
    print("=" * 60)
    
    # Load enrollment data
    print("\n📂 Loading enrollment data...")
    df_enrol = read_all_csvs_in_folder("api_data_aadhar_enrolment")
    
    if df_enrol.empty:
        print("❌ Error: No enrollment data found!")
        return
    
    print(f"✓ Loaded {len(df_enrol):,} enrollment records")
    
    # Apply preprocessing
    print("\n🔧 Preprocessing data...")
    df_enrol = preprocess_dataframe(df_enrol, "Enrollment")
    print("✓ Preprocessing complete")
    
    # Initialize cache dictionary
    security_cache = {}
    
    # ==========================================
    # 1. BENFORD'S LAW ANALYSIS
    # ==========================================
    print("\n🔍 Analyzing Benford's Law distribution...")
    try:
        benford_data = get_benfords_law_global(df_enrol)
        
        if "error" not in benford_data:
            # Save data as serializable format
            benford_dict = {
                'digits': benford_data['distribution']['digit'].tolist(),
                'actual_freq': benford_data['distribution']['actual_freq'].tolist(),
                'benford_freq': benford_data['distribution']['benford_freq'].tolist(),
                'deviation_score': benford_data['deviation_score'],
                'is_suspicious': benford_data['is_suspicious']
            }
            security_cache['benford_data'] = benford_dict
            
            # Create and save Plotly chart
            fig_benford = create_benford_chart(benford_data)
            security_cache['benford_chart'] = plotly.io.to_json(fig_benford)
            
            status = "SUSPICIOUS" if benford_data['is_suspicious'] else "NORMAL"
            print(f"  ✓ Benford's Law analysis complete: {status} (deviation: {benford_data['deviation_score']:.3f})")
        else:
            print("  ⚠️ Insufficient data for Benford's Law analysis")
            security_cache['benford_data'] = {}
            security_cache['benford_chart'] = None
    except Exception as e:
        print(f"  ✗ Error in Benford's Law analysis: {e}")
        security_cache['benford_data'] = {}
        security_cache['benford_chart'] = None
    
    # ==========================================
    # 2. STATISTICAL OUTLIERS
    # ==========================================
    print("\n📊 Detecting statistical outliers...")
    try:
        outliers_data = get_statistical_outliers(df_enrol)
        
        if not outliers_data.empty:
            # Save data as serializable format
            outliers_dict = {
                'dates': outliers_data['date'].astype(str).tolist(),
                'total_activity': outliers_data['total_activity'].tolist(),
                'is_anomaly': outliers_data['is_anomaly'].tolist(),
                'threshold': outliers_data['threshold'].tolist()
            }
            security_cache['outliers_data'] = outliers_dict
            
            # Create and save Plotly chart
            fig_outliers, anomaly_count = create_outliers_chart(outliers_data)
            security_cache['outliers_chart'] = plotly.io.to_json(fig_outliers)
            security_cache['anomaly_count'] = anomaly_count
            
            print(f"  ✓ Outlier detection complete: {anomaly_count} suspicious days detected")
        else:
            print("  ⚠️ Insufficient time-series data for outlier detection")
            security_cache['outliers_data'] = {}
            security_cache['outliers_chart'] = None
            security_cache['anomaly_count'] = 0
    except Exception as e:
        print(f"  ✗ Error in outlier detection: {e}")
        security_cache['outliers_data'] = {}
        security_cache['outliers_chart'] = None
        security_cache['anomaly_count'] = 0
    
    # ==========================================
    # 3. VOLATILITY ANALYSIS
    # ==========================================
    print("\n📈 Analyzing center volatility...")
    try:
        volatility_data = get_volatility_analysis(df_enrol, top_n=20)
        
        if not volatility_data.empty:
            # Save data as serializable format
            volatility_dict = {
                'pincodes': volatility_data['pincode'].astype(str).tolist(),
                'mean': volatility_data['mean'].tolist(),
                'std': volatility_data['std'].tolist(),
                'count': volatility_data['count'].astype(int).tolist(),
                'variance_score': volatility_data['variance_score'].tolist()
            }
            security_cache['volatility_data'] = volatility_dict
            
            # Create and save Plotly chart
            fig_volatility = create_volatility_chart(volatility_data)
            security_cache['volatility_chart'] = plotly.io.to_json(fig_volatility)
            
            print(f"  ✓ Volatility analysis complete: {len(volatility_data)} most erratic centers identified")
        else:
            print("  ⚠️ Insufficient data for volatility analysis")
            security_cache['volatility_data'] = {}
            security_cache['volatility_chart'] = None
    except Exception as e:
        print(f"  ✗ Error in volatility analysis: {e}")
        security_cache['volatility_data'] = {}
        security_cache['volatility_chart'] = None
    
    # ==========================================
    # 4. STATE VARIANCE ANALYSIS
    # ==========================================
    print("\n🗺️  Analyzing state variance...")
    try:
        state_variance = get_state_variance_data(df_enrol)
        
        if not state_variance.empty and len(state_variance['state'].unique()) > 1:
            # Save data as serializable format
            state_dict = {
                'states': state_variance['state'].tolist(),
                'districts': state_variance['district'].tolist(),
                'total_activity': state_variance['total_activity'].tolist()
            }
            security_cache['state_variance_data'] = state_dict
            
            # Create and save Plotly chart
            fig_variance = create_state_variance_chart(state_variance)
            security_cache['state_variance_chart'] = plotly.io.to_json(fig_variance)
            
            print(f"  ✓ State variance analysis complete: {len(state_variance['state'].unique())} states analyzed")
        else:
            print("  ⚠️ Insufficient cross-state data for variance analysis")
            security_cache['state_variance_data'] = {}
            security_cache['state_variance_chart'] = None
    except Exception as e:
        print(f"  ✗ Error in state variance analysis: {e}")
        security_cache['state_variance_data'] = {}
        security_cache['state_variance_chart'] = None
    
    # ==========================================
    # SAVE TO CACHE FILE
    # ==========================================
    cache_file = 'security_cache.json'
    
    # Add metadata
    security_cache['metadata'] = {
        'generated_at': datetime.now().isoformat(),
        'total_records': len(df_enrol),
        'date_range': {
            'start': df_enrol['date'].min().isoformat() if 'date' in df_enrol.columns else None,
            'end': df_enrol['date'].max().isoformat() if 'date' in df_enrol.columns else None
        },
        'districts_analyzed': int(df_enrol['district'].nunique()) if 'district' in df_enrol.columns else 0,
        'states_analyzed': int(df_enrol['state'].nunique()) if 'state' in df_enrol.columns else 0,
        'pincodes_analyzed': int(df_enrol['pincode'].nunique()) if 'pincode' in df_enrol.columns else 0
    }
    
    print("\n💾 Saving cache to file...")
    try:
        with open(cache_file, 'w') as f:
            json.dump(security_cache, f, indent=2)
        
        file_size = os.path.getsize(cache_file) / (1024 * 1024)  # MB
        print(f"  ✓ Cache saved: {cache_file} ({file_size:.2f} MB)")
    except Exception as e:
        print(f"  ✗ Error saving cache: {e}")
        return
    
    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "=" * 60)
    print("✅ SECURITY CACHE GENERATION COMPLETE!")
    print("=" * 60)
    print("📊 Cached Items:")
    print(f"  • Benford's Law Analysis: {'✓' if security_cache.get('benford_data') else '✗'}")
    print(f"  • Statistical Outliers: {'✓' if security_cache.get('outliers_data') else '✗'}")
    print(f"  • Volatility Analysis: {'✓' if security_cache.get('volatility_data') else '✗'}")
    print(f"  • State Variance Analysis: {'✓' if security_cache.get('state_variance_data') else '✗'}")
    print(f"\n📁 Cache file: {cache_file}")
    print(f"📅 Generated: {security_cache['metadata']['generated_at']}")
    print(f"📊 Records analyzed: {security_cache['metadata']['total_records']:,}")
    print(f"🏙️ Districts: {security_cache['metadata']['districts_analyzed']}")
    print(f"🗺️  States: {security_cache['metadata']['states_analyzed']}")
    print(f"📍 Pincodes: {security_cache['metadata']['pincodes_analyzed']}")
    
    if security_cache.get('benford_data'):
        status = "⚠️ SUSPICIOUS" if security_cache['benford_data']['is_suspicious'] else "✓ NORMAL"
        print(f"\n🔍 Fraud Status: {status}")
        print(f"   Deviation Score: {security_cache['benford_data']['deviation_score']:.3f}")
    
    if security_cache.get('anomaly_count', 0) > 0:
        print(f"\n🚨 Anomalies Detected: {security_cache['anomaly_count']} suspicious activity days")
    
    print("\n💡 Next step: Run your Streamlit app to see the cached visualizations!")
    print("=" * 60)


if __name__ == "__main__":
    generate_security_cache()