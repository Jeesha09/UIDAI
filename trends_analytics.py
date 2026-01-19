"""
Trends & Forecasting Analytics Module  
Data preparation functions that return DataFrames for Streamlit native charts
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
import warnings

warnings.filterwarnings('ignore')


# ==========================================
# FEATURE 7A: FORECAST DATA
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


# ==========================================
# FEATURE 1B: DAY-OF-WEEK DATA
# ==========================================
def get_dow_data(df):
    """Get day-of-week patterns - returns DataFrame"""
    age_cols = [col for col in df.columns if 'age_' in col]
    
    dow_data = df.groupby(['day_of_week', 'area_type'])[age_cols].sum().reset_index()
    dow_data['total'] = dow_data[age_cols].sum(axis=1)
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_data['day_of_week'] = pd.Categorical(dow_data['day_of_week'], categories=day_order, ordered=True)
    
    pivot = dow_data.pivot_table(index='day_of_week', columns='area_type', values='total', aggfunc='sum').reset_index()
    
    return pivot.sort_values('day_of_week').set_index('day_of_week')


# ==========================================
# FEATURE 10A: GROWTH TRAJECTORY DATA
# ==========================================
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


# ==========================================
# PLOTLY CHARTS (FOR COMPLEX VISUALIZATIONS)
# ==========================================
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
    import networkx as nx
    
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
