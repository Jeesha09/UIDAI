"""
Trends & Forecasting Cache Generator
Pre-generates all visualizations for Page 3 (Trends & Forecasting) and saves to JSON cache.
Run this script whenever data is updated to refresh the trends cache.

Usage:
    python generate_trends_cache.py
"""

import pandas as pd
import numpy as np
import json
import os
import glob
from datetime import datetime
from data_preprocessing import preprocess_dataframe
import plotly
import plotly.graph_objects as go
from prophet import Prophet
import networkx as nx
import warnings

warnings.filterwarnings('ignore')


# ==========================================
# TRENDS ANALYTICS FUNCTIONS
# ==========================================

def get_forecast_data(df, days_forward=30):
    """Prepare forecast data - returns DataFrame for line chart"""
    age_cols = [col for col in df.columns if 'age_' in col]
    
    daily_data = df.groupby('date')[age_cols].sum().reset_index()
    daily_data['total'] = daily_data[age_cols].sum(axis=1)
    
    prophet_df = daily_data[['date', 'total']].rename(columns={'date': 'ds', 'total': 'y'})
    
    if len(prophet_df) < 2:
        return pd.DataFrame()
    
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True, interval_width=0.95)
    model.fit(prophet_df)
    
    future = model.make_future_dataframe(periods=days_forward)
    forecast = model.predict(future)
    
    result = pd.merge(
        prophet_df.rename(columns={'ds': 'date', 'y': 'Historical'}),
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={'ds': 'date', 'yhat': 'Forecast'}),
        on='date',
        how='outer'
    )
    
    return result.sort_values('date').set_index('date')


def get_dow_data(df):
    """Get day-of-week patterns - returns DataFrame"""
    age_cols = [col for col in df.columns if 'age_' in col]
    
    dow_data = df.groupby(['day_of_week', 'area_type'])[age_cols].sum().reset_index()
    dow_data['total'] = dow_data[age_cols].sum(axis=1)
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_data['day_of_week'] = pd.Categorical(dow_data['day_of_week'], categories=day_order, ordered=True)
    
    pivot = dow_data.pivot_table(index='day_of_week', columns='area_type', values='total', aggfunc='sum').fillna(0)
    
    # Ensure we have both Rural and Urban columns
    if 'Rural' not in pivot.columns:
        pivot['Rural'] = 0
    if 'Urban' not in pivot.columns:
        pivot['Urban'] = 0
    
    # Keep only Rural and Urban in specific order
    pivot = pivot[['Rural', 'Urban']]
    
    # Reindex to ensure correct day order
    pivot = pivot.reindex(day_order)
    
    return pivot


def get_growth_data(df, top_n=10):
    """Get cumulative growth - returns DataFrame"""
    age_cols = [col for col in df.columns if 'age_' in col]
    
    district_totals = df.groupby('district')[age_cols].sum().reset_index()
    district_totals['total'] = district_totals[age_cols].sum(axis=1)
    top_districts = district_totals.nlargest(top_n, 'total')['district'].values
    
    trend_data = df[df['district'].isin(top_districts)].groupby(['date', 'district'])[age_cols].sum().reset_index()
    trend_data['total'] = trend_data[age_cols].sum(axis=1)
    
    trend_data = trend_data.sort_values(['district', 'date'])
    trend_data['cumulative'] = trend_data.groupby('district')['total'].cumsum()
    
    pivot = trend_data.pivot_table(index='date', columns='district', values='cumulative', aggfunc='sum').reset_index()
    
    return pivot.set_index('date')


def create_seasonal_radar(df):
    """Radar chart - requires Plotly"""
    age_cols = [col for col in df.columns if 'age_' in col]
    
    monthly_data = df.groupby(['month', 'month_name', 'region'])[age_cols].sum().reset_index()
    monthly_data['total'] = monthly_data[age_cols].sum(axis=1)
    
    pivot_data = monthly_data.pivot_table(index=['month', 'month_name'], columns='region', values='total', aggfunc='sum').reset_index()
    pivot_data = pivot_data.sort_values('month')
    
    month_names = pivot_data['month_name'].tolist()
    
    fig = go.Figure()
    
    for region in ['North', 'South']:
        if region in pivot_data.columns:
            fig.add_trace(go.Scatterpolar(
                r=pivot_data[region].values,
                theta=month_names,
                fill='toself',
                name=f'{region} States',
                line=dict(width=2)
            ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, showticklabels=True)),
        title='Seasonal Enrollment Trends: North vs South India',
        showlegend=True,
        height=600,
        template='plotly_white'
    )
    
    return fig


def create_network_graph(df, top_n=25):
    """Network graph - requires Plotly + NetworkX"""
    age_cols = [col for col in df.columns if 'age_' in col]
    
    district_totals = df.groupby('district')[age_cols].sum().reset_index()
    district_totals['total'] = district_totals[age_cols].sum(axis=1)
    top_districts = district_totals.nlargest(top_n, 'total')['district'].values
    
    df_filtered = df[df['district'].isin(top_districts)]
    
    district_pivot = df_filtered.pivot_table(
        index='date',
        columns='district',
        values=age_cols[0],
        aggfunc='sum'
    ).fillna(0)
    
    corr_matrix = district_pivot.corr()
    
    G = nx.Graph()
    threshold = 0.7
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j])
    
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    
    node_x, node_y, node_text, node_size = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f'{node}<br>Connections: {G.degree(node)}')
        node_size.append(10 + G.degree(node) * 3)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[n.split()[0] for n in G.nodes()],
        hovertext=node_text,
        textposition='top center',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=node_size,
            color=[G.degree(n) for n in G.nodes()],
            colorbar=dict(thickness=15, title=dict(text='Connections', side='right'), xanchor='left'),
            line=dict(width=2, color='white')
        )
    )
    
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text='District Ripple Effect Network', font=dict(size=16)),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=0, r=0, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700,
            template='plotly_white'
        )
    )
    
    return fig


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


def generate_trends_cache():
    """Generate all trends analytics and save to JSON cache"""
    
    print("=" * 60)
    print("🚀 TRENDS ANALYTICS CACHE GENERATOR")
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
    trends_cache = {}
    
    # ==========================================
    # 1. 30-DAY ENROLLMENT FORECAST
    # ==========================================
    print("\n📈 Generating 30-day enrollment forecast...")
    try:
        forecast_data = get_forecast_data(df_enrol, days_forward=30)
        
        if not forecast_data.empty:
            # Convert to serializable format
            forecast_data_dict = {
                'dates': forecast_data.index.astype(str).tolist(),
                'Historical': forecast_data['Historical'].fillna(0).tolist(),
                'Forecast': forecast_data['Forecast'].fillna(0).tolist(),
                'yhat_lower': forecast_data.get('yhat_lower', [0] * len(forecast_data)).fillna(0).tolist() if 'yhat_lower' in forecast_data.columns else None,
                'yhat_upper': forecast_data.get('yhat_upper', [0] * len(forecast_data)).fillna(0).tolist() if 'yhat_upper' in forecast_data.columns else None
            }
            trends_cache['forecast_data'] = forecast_data_dict
            print(f"  ✓ Forecast generated: {len(forecast_data)} data points")
        else:
            print("  ⚠️ Insufficient data for forecasting")
            trends_cache['forecast_data'] = {}
    except Exception as e:
        print(f"  ✗ Error generating forecast: {e}")
        trends_cache['forecast_data'] = {}
    
    # ==========================================
    # 2. DAY-OF-WEEK PATTERNS
    # ==========================================
    print("\n📅 Analyzing day-of-week patterns...")
    try:
        dow_data = get_dow_data(df_enrol)
        
        if not dow_data.empty:
            # Convert to serializable format
            dow_data_dict = {
                'days': dow_data.index.tolist(),
                'columns': dow_data.columns.tolist(),
                'data': dow_data.values.tolist()
            }
            trends_cache['dow_data'] = dow_data_dict
            print(f"  ✓ Day-of-week data generated: {len(dow_data)} days")
        else:
            print("  ⚠️ No day-of-week data available")
            trends_cache['dow_data'] = {}
    except Exception as e:
        print(f"  ✗ Error generating day-of-week data: {e}")
        trends_cache['dow_data'] = {}
    
    # ==========================================
    # 3. CUMULATIVE GROWTH TRAJECTORIES
    # ==========================================
    print("\n📊 Generating cumulative growth trajectories...")
    try:
        growth_data = get_growth_data(df_enrol, top_n=10)
        
        if not growth_data.empty:
            # Convert to serializable format
            growth_data_dict = {
                'dates': growth_data.index.astype(str).tolist(),
                'districts': growth_data.columns.tolist(),
                'data': growth_data.values.tolist()
            }
            trends_cache['growth_data'] = growth_data_dict
            print(f"  ✓ Growth trajectories generated: {len(growth_data.columns)} districts")
        else:
            print("  ⚠️ No growth data available")
            trends_cache['growth_data'] = {}
    except Exception as e:
        print(f"  ✗ Error generating growth data: {e}")
        trends_cache['growth_data'] = {}
    
    # ==========================================
    # 4. SEASONAL RADAR CHART
    # ==========================================
    print("\n🌡️ Creating seasonal radar chart...")
    try:
        seasonal_radar = create_seasonal_radar(df_enrol)
        
        # Convert Plotly figure to JSON
        seasonal_radar_json = plotly.io.to_json(seasonal_radar)
        trends_cache['seasonal_radar'] = seasonal_radar_json
        print("  ✓ Seasonal radar chart generated")
    except Exception as e:
        print(f"  ✗ Error generating seasonal radar: {e}")
        trends_cache['seasonal_radar'] = None
    
    # ==========================================
    # 5. NETWORK GRAPH
    # ==========================================
    print("\n🔗 Building district correlation network...")
    try:
        network_graph = create_network_graph(df_enrol, top_n=25)
        
        # Convert Plotly figure to JSON
        network_graph_json = plotly.io.to_json(network_graph)
        trends_cache['network_graph'] = network_graph_json
        print("  ✓ Network graph generated (25 districts)")
    except Exception as e:
        print(f"  ✗ Error generating network graph: {e}")
        trends_cache['network_graph'] = None
    
    # ==========================================
    # SAVE TO CACHE FILE
    # ==========================================
    cache_file = 'trends_cache.json'
    
    # Add metadata
    trends_cache['metadata'] = {
        'generated_at': datetime.now().isoformat(),
        'total_records': len(df_enrol),
        'date_range': {
            'start': df_enrol['date'].min().isoformat() if 'date' in df_enrol.columns else None,
            'end': df_enrol['date'].max().isoformat() if 'date' in df_enrol.columns else None
        },
        'districts_analyzed': int(df_enrol['district'].nunique()) if 'district' in df_enrol.columns else 0
    }
    
    print("\n💾 Saving cache to file...")
    try:
        with open(cache_file, 'w') as f:
            json.dump(trends_cache, f, indent=2)
        
        file_size = os.path.getsize(cache_file) / (1024 * 1024)  # MB
        print(f"  ✓ Cache saved: {cache_file} ({file_size:.2f} MB)")
    except Exception as e:
        print(f"  ✗ Error saving cache: {e}")
        return
    
    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "=" * 60)
    print("✅ TRENDS CACHE GENERATION COMPLETE!")
    print("=" * 60)
    print(f"📊 Cached Items:")
    print(f"  • 30-Day Forecast: {'✓' if trends_cache.get('forecast_data') else '✗'}")
    print(f"  • Day-of-Week Patterns: {'✓' if trends_cache.get('dow_data') else '✗'}")
    print(f"  • Growth Trajectories: {'✓' if trends_cache.get('growth_data') else '✗'}")
    print(f"  • Seasonal Radar: {'✓' if trends_cache.get('seasonal_radar') else '✗'}")
    print(f"  • Network Graph: {'✓' if trends_cache.get('network_graph') else '✗'}")
    print(f"\n📁 Cache file: {cache_file}")
    print(f"📅 Generated: {trends_cache['metadata']['generated_at']}")
    print(f"📊 Records analyzed: {trends_cache['metadata']['total_records']:,}")
    print("\n💡 Next step: Run your Streamlit app to see the cached visualizations!")
    print("=" * 60)


if __name__ == "__main__":
    generate_trends_cache()
