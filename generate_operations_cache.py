"""
Operations & Logistics Cache Generator
Pre-generates all visualizations for Page 2 (Operations & Logistics) and saves to JSON cache.
Run this script whenever data is updated to refresh the operations cache.

Usage:
    python generate_operations_cache.py
"""

import pandas as pd
import numpy as np
import json
import os
import glob
from datetime import datetime
from data_preprocessing import preprocess_dataframe
import plotly
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')


# ==========================================
# OPERATIONS ANALYTICS FUNCTIONS
# ==========================================

def get_bcg_matrix_data(df_enrol, df_bio, df_demo):
    """Service Strain Matrix with BCG-style quadrants"""
    # Calculate metrics
    age_cols_enrol = [col for col in df_enrol.columns if col.startswith('age_')]
    age_cols_demo = [col for col in df_demo.columns if col.startswith('demo_age')]
    age_cols_bio = [col for col in df_bio.columns if col.startswith('bio_age')]
    
    enrollments = df_enrol.groupby('district')[age_cols_enrol].sum().sum(axis=1).reset_index()
    enrollments.columns = ['district', 'total_enrollments']
    
    updates_demo = df_demo.groupby('district')[age_cols_demo].sum().sum(axis=1).reset_index() if age_cols_demo else pd.DataFrame(columns=['district', 'total_updates'])
    updates_bio = df_bio.groupby('district')[age_cols_bio].sum().sum(axis=1).reset_index() if age_cols_bio else pd.DataFrame(columns=['district', 'total_updates'])
    
    if not updates_demo.empty:
        updates_demo.columns = ['district', 'total_updates']
    if not updates_bio.empty:
        updates_bio.columns = ['district', 'total_updates']
        
    # Merge updates
    if not updates_demo.empty and not updates_bio.empty:
        updates = pd.merge(updates_demo, updates_bio, on='district', how='outer', suffixes=('_demo', '_bio'))
        updates['total_updates'] = updates[['total_updates_demo', 'total_updates_bio']].sum(axis=1)
        updates = updates[['district', 'total_updates']]
    elif not updates_demo.empty:
        updates = updates_demo
    elif not updates_bio.empty:
        updates = updates_bio
    else:
        updates = pd.DataFrame(columns=['district', 'total_updates'])
    
    # Merge enrollments and updates
    matrix_data = pd.merge(enrollments, updates, on='district', how='outer').fillna(0)
    
    return matrix_data


def get_mobile_van_priority_data(df_enrol, df_bio, df_demo):
    """Identify high-priority areas for mobile van deployment"""
    age_cols_demo = [col for col in df_demo.columns if col.startswith('demo_age')]
    age_cols_bio = [col for col in df_bio.columns if col.startswith('bio_age')]
    
    # Combine enrollment and update data by pincode
    enrol_pincode = df_enrol.groupby(['district', 'pincode'])['age_18_greater'].sum().reset_index()
    enrol_pincode.columns = ['district', 'pincode', 'total_enrollments']
    
    updates = pd.DataFrame()
    if age_cols_demo:
        updates_demo = df_demo.groupby(['district', 'pincode'])[age_cols_demo].sum().sum(axis=1).reset_index()
        updates_demo.columns = ['district', 'pincode', 'total_updates']
        updates = updates_demo
    
    if age_cols_bio:
        updates_bio = df_bio.groupby(['district', 'pincode'])[age_cols_bio].sum().sum(axis=1).reset_index()
        updates_bio.columns = ['district', 'pincode', 'total_updates']
        if not updates.empty:
            updates = pd.merge(updates, updates_bio, on=['district', 'pincode'], how='outer', suffixes=('_demo', '_bio'))
            updates['total_updates'] = updates[['total_updates_demo', 'total_updates_bio']].sum(axis=1)
            updates = updates[['district', 'pincode', 'total_updates']]
        else:
            updates = updates_bio
    
    # Merge data
    van_data = pd.merge(enrol_pincode, updates, on=['district', 'pincode'], how='outer').fillna(0)
    van_data['total_activity'] = van_data['total_enrollments'] + van_data['total_updates']
    
    # Flag high-priority areas (high updates, indicating need for mobile service)
    if len(van_data) > 0:
        threshold = van_data['total_updates'].quantile(0.85)
        van_data['van_priority'] = van_data['total_updates'] > threshold
    else:
        van_data['van_priority'] = False
        
    return van_data


def get_center_productivity_data(van_data, top_n=20):
    """Top performing centers by activity volume"""
    top_centers = van_data.nlargest(top_n, 'total_activity')
    return top_centers


def get_weekly_capacity_data(df_enrol):
    """Weekly activity patterns for capacity planning"""
    if 'day_of_week' not in df_enrol.columns:
        return pd.DataFrame()
    
    age_cols = [col for col in df_enrol.columns if col.startswith('age_')]
    weekly_data = df_enrol.groupby('day_of_week')[age_cols].sum().sum(axis=1).reset_index()
    weekly_data.columns = ['day_of_week', 'total_activity']
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_data['day_of_week'] = pd.Categorical(weekly_data['day_of_week'], categories=day_order, ordered=True)
    weekly_data = weekly_data.sort_values('day_of_week')
    
    return weekly_data


def get_growth_velocity_data(df_enrol, top_n=5):
    """Enrollment growth velocity for top districts"""
    if 'date' not in df_enrol.columns:
        return pd.DataFrame()
    
    age_cols = [col for col in df_enrol.columns if col.startswith('age_')]
    
    # Get top districts by total volume
    top_districts = df_enrol.groupby('district')[age_cols].sum().sum(axis=1).nlargest(top_n).index
    
    # Time series data for top districts
    velocity_data = df_enrol[df_enrol['district'].isin(top_districts)].groupby(['date', 'district'])[age_cols].sum().sum(axis=1).reset_index()
    velocity_data.columns = ['date', 'district', 'total_enrollments']
    velocity_data = velocity_data.sort_values('date')
    
    return velocity_data


def create_bcg_scatter_chart(bcg_data):
    """Create BCG Matrix Plotly chart"""
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
    
    return fig_bcg


def create_mobile_van_chart(van_data):
    """Create Mobile Van Priority Plotly chart"""
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
    
    return fig_van


def create_productivity_chart(prod_data):
    """Create Center Productivity Plotly chart"""
    prod_data = prod_data.copy()
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
    
    return fig_prod


def create_weekly_heatmap(weekly_data):
    """Create Weekly Capacity Heatmap"""
    fig_heat = px.imshow(
        [weekly_data['total_activity'].values],
        x=weekly_data['day_of_week'].values,
        y=['Avg Activity'],
        color_continuous_scale='Viridis',
        labels={'color': 'Activity Level'},
        template="plotly_white",
        height=300
    )
    
    return fig_heat


def create_velocity_chart(velocity_data):
    """Create Growth Velocity Line Chart"""
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
    
    return fig_vel


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


def generate_operations_cache():
    """Generate all operations analytics and save to JSON cache"""
    
    print("=" * 60)
    print("🚀 OPERATIONS ANALYTICS CACHE GENERATOR")
    print("=" * 60)
    
    # Load all data
    print("\n📂 Loading enrollment data...")
    df_enrol = read_all_csvs_in_folder("api_data_aadhar_enrolment")
    
    print("\n📂 Loading biometric data...")
    df_bio = read_all_csvs_in_folder("api_data_aadhar_biometric")
    
    print("\n📂 Loading demographic data...")
    df_demo = read_all_csvs_in_folder("api_data_aadhar_demographic")
    
    if df_enrol.empty:
        print("❌ Error: No enrollment data found!")
        return
    
    print(f"\n✓ Loaded data:")
    print(f"  • Enrollment: {len(df_enrol):,} records")
    print(f"  • Biometric: {len(df_bio):,} records")
    print(f"  • Demographic: {len(df_demo):,} records")
    
    # Apply preprocessing
    print("\n🔧 Preprocessing data...")
    df_enrol = preprocess_dataframe(df_enrol, "Enrollment")
    df_bio = preprocess_dataframe(df_bio, "Biometric")
    df_demo = preprocess_dataframe(df_demo, "Demographic")
    print("✓ Preprocessing complete")
    
    # Initialize cache dictionary
    operations_cache = {}
    
    # ==========================================
    # 1. BCG SERVICE STRAIN MATRIX
    # ==========================================
    print("\n📊 Generating BCG Service Strain Matrix...")
    try:
        bcg_data = get_bcg_matrix_data(df_enrol, df_bio, df_demo)
        
        if not bcg_data.empty:
            # Save data as serializable format
            bcg_data_dict = {
                'districts': bcg_data['district'].tolist(),
                'total_enrollments': bcg_data['total_enrollments'].tolist(),
                'total_updates': bcg_data['total_updates'].tolist()
            }
            operations_cache['bcg_data'] = bcg_data_dict
            
            # Create and save Plotly chart
            fig_bcg = create_bcg_scatter_chart(bcg_data)
            operations_cache['bcg_chart'] = plotly.io.to_json(fig_bcg)
            
            print(f"  ✓ BCG matrix generated: {len(bcg_data)} districts")
        else:
            print("  ⚠️ Insufficient data for BCG matrix")
            operations_cache['bcg_data'] = {}
            operations_cache['bcg_chart'] = None
    except Exception as e:
        print(f"  ✗ Error generating BCG matrix: {e}")
        operations_cache['bcg_data'] = {}
        operations_cache['bcg_chart'] = None
    
    # ==========================================
    # 2. MOBILE VAN PRIORITY DATA
    # ==========================================
    print("\n🚐 Analyzing mobile van deployment priorities...")
    try:
        van_data = get_mobile_van_priority_data(df_enrol, df_bio, df_demo)
        
        if not van_data.empty:
            # Save data as serializable format
            van_data_dict = {
                'districts': van_data['district'].tolist(),
                'pincodes': van_data['pincode'].astype(str).tolist(),
                'total_enrollments': van_data['total_enrollments'].tolist(),
                'total_updates': van_data['total_updates'].tolist(),
                'total_activity': van_data['total_activity'].tolist(),
                'van_priority': van_data['van_priority'].tolist()
            }
            operations_cache['van_data'] = van_data_dict
            
            # Create and save Plotly chart
            fig_van = create_mobile_van_chart(van_data)
            operations_cache['van_chart'] = plotly.io.to_json(fig_van)
            
            print(f"  ✓ Mobile van data generated: {len(van_data)} pincodes")
        else:
            print("  ⚠️ No mobile van data available")
            operations_cache['van_data'] = {}
            operations_cache['van_chart'] = None
    except Exception as e:
        print(f"  ✗ Error generating mobile van data: {e}")
        operations_cache['van_data'] = {}
        operations_cache['van_chart'] = None
    
    # ==========================================
    # 3. CENTER PRODUCTIVITY RANKINGS
    # ==========================================
    print("\n🏆 Calculating center productivity rankings...")
    try:
        if not van_data.empty:
            prod_data = get_center_productivity_data(van_data, top_n=20)
            
            if not prod_data.empty:
                # Save data as serializable format
                prod_data_dict = {
                    'districts': prod_data['district'].tolist(),
                    'pincodes': prod_data['pincode'].astype(str).tolist(),
                    'total_activity': prod_data['total_activity'].tolist()
                }
                operations_cache['productivity_data'] = prod_data_dict
                
                # Create and save Plotly chart
                fig_prod = create_productivity_chart(prod_data)
                operations_cache['productivity_chart'] = plotly.io.to_json(fig_prod)
                
                print(f"  ✓ Productivity rankings generated: Top {len(prod_data)} centers")
            else:
                print("  ⚠️ No productivity data available")
                operations_cache['productivity_data'] = {}
                operations_cache['productivity_chart'] = None
        else:
            operations_cache['productivity_data'] = {}
            operations_cache['productivity_chart'] = None
    except Exception as e:
        print(f"  ✗ Error generating productivity data: {e}")
        operations_cache['productivity_data'] = {}
        operations_cache['productivity_chart'] = None
    
    # ==========================================
    # 4. WEEKLY CAPACITY UTILIZATION HEATMAP
    # ==========================================
    print("\n📅 Analyzing weekly capacity utilization...")
    try:
        weekly_data = get_weekly_capacity_data(df_enrol)
        
        if not weekly_data.empty:
            # Save data as serializable format
            weekly_data_dict = {
                'days': weekly_data['day_of_week'].tolist(),
                'total_activity': weekly_data['total_activity'].tolist()
            }
            operations_cache['weekly_data'] = weekly_data_dict
            
            # Create and save Plotly chart
            fig_heat = create_weekly_heatmap(weekly_data)
            operations_cache['weekly_chart'] = plotly.io.to_json(fig_heat)
            
            print(f"  ✓ Weekly capacity data generated: {len(weekly_data)} days")
        else:
            print("  ⚠️ No weekly capacity data available")
            operations_cache['weekly_data'] = {}
            operations_cache['weekly_chart'] = None
    except Exception as e:
        print(f"  ✗ Error generating weekly capacity data: {e}")
        operations_cache['weekly_data'] = {}
        operations_cache['weekly_chart'] = None
    
    # ==========================================
    # 5. GROWTH VELOCITY TRACKING
    # ==========================================
    print("\n📈 Tracking enrollment growth velocity...")
    try:
        velocity_data = get_growth_velocity_data(df_enrol, top_n=5)
        
        if not velocity_data.empty:
            # Save data as serializable format
            velocity_data_dict = {
                'dates': velocity_data['date'].astype(str).tolist(),
                'districts': velocity_data['district'].tolist(),
                'total_enrollments': velocity_data['total_enrollments'].tolist()
            }
            operations_cache['velocity_data'] = velocity_data_dict
            
            # Create and save Plotly chart
            fig_vel = create_velocity_chart(velocity_data)
            operations_cache['velocity_chart'] = plotly.io.to_json(fig_vel)
            
            print(f"  ✓ Growth velocity generated: {velocity_data['district'].nunique()} districts")
        else:
            print("  ⚠️ No growth velocity data available")
            operations_cache['velocity_data'] = {}
            operations_cache['velocity_chart'] = None
    except Exception as e:
        print(f"  ✗ Error generating growth velocity data: {e}")
        operations_cache['velocity_data'] = {}
        operations_cache['velocity_chart'] = None
    
    # ==========================================
    # SAVE TO CACHE FILE
    # ==========================================
    cache_file = 'operations_cache.json'
    
    # Add metadata
    operations_cache['metadata'] = {
        'generated_at': datetime.now().isoformat(),
        'total_records': {
            'enrollment': len(df_enrol),
            'biometric': len(df_bio),
            'demographic': len(df_demo)
        },
        'date_range': {
            'start': df_enrol['date'].min().isoformat() if 'date' in df_enrol.columns else None,
            'end': df_enrol['date'].max().isoformat() if 'date' in df_enrol.columns else None
        },
        'districts_analyzed': int(df_enrol['district'].nunique()) if 'district' in df_enrol.columns else 0,
        'pincodes_analyzed': int(df_enrol['pincode'].nunique()) if 'pincode' in df_enrol.columns else 0
    }
    
    print("\n💾 Saving cache to file...")
    try:
        with open(cache_file, 'w') as f:
            json.dump(operations_cache, f, indent=2)
        
        file_size = os.path.getsize(cache_file) / (1024 * 1024)  # MB
        print(f"  ✓ Cache saved: {cache_file} ({file_size:.2f} MB)")
    except Exception as e:
        print(f"  ✗ Error saving cache: {e}")
        return
    
    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "=" * 60)
    print("✅ OPERATIONS CACHE GENERATION COMPLETE!")
    print("=" * 60)
    print(f"📊 Cached Items:")
    print(f"  • BCG Service Strain Matrix: {'✓' if operations_cache.get('bcg_data') else '✗'}")
    print(f"  • Mobile Van Priority Data: {'✓' if operations_cache.get('van_data') else '✗'}")
    print(f"  • Center Productivity Rankings: {'✓' if operations_cache.get('productivity_data') else '✗'}")
    print(f"  • Weekly Capacity Utilization: {'✓' if operations_cache.get('weekly_data') else '✗'}")
    print(f"  • Growth Velocity Tracking: {'✓' if operations_cache.get('velocity_data') else '✗'}")
    print(f"\n📁 Cache file: {cache_file}")
    print(f"📅 Generated: {operations_cache['metadata']['generated_at']}")
    print(f"📊 Records analyzed:")
    print(f"  • Enrollment: {operations_cache['metadata']['total_records']['enrollment']:,}")
    print(f"  • Biometric: {operations_cache['metadata']['total_records']['biometric']:,}")
    print(f"  • Demographic: {operations_cache['metadata']['total_records']['demographic']:,}")
    print(f"🏙️ Districts: {operations_cache['metadata']['districts_analyzed']}")
    print(f"📍 Pincodes: {operations_cache['metadata']['pincodes_analyzed']}")
    print("\n💡 Next step: Run your Streamlit app to see the cached visualizations!")
    print("=" * 60)


if __name__ == "__main__":
    generate_operations_cache()
