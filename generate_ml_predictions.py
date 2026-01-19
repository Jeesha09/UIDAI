"""
ALL-IN-ONE ML PREDICTIONS GENERATOR
Run this script ONCE to generate all ML predictions and save to JSON.
The main app loads the static JSON file instantly.

This file contains ALL ML functions inline - no need for ml_predictive_analytics.py
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import glob
from data_preprocessing import preprocess_dataframe
from sklearn.ensemble import IsolationForest
import time
import math
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("ML PREDICTIONS GENERATOR")
print("=" * 60)
print("This script will run all ML models and save results to static files.")
print("The main app will then load these files instantly instead of recomputing.\n")

# ==========================================
# LOAD DATA
# ==========================================
print("📂 Loading CSV files...")

def load_data():
    """Load all CSV files from the three directories with optimizations"""
    
    # OPTIMIZATION: Read only required columns and use efficient dtypes
    base_cols = ['date', 'state', 'district', 'pincode']
    
    print("  Loading enrollment data...")
    enrol_files = glob.glob('api_data_aadhar_enrolment/*.csv')
    # Read first file to get age columns
    sample_df = pd.read_csv(enrol_files[0], nrows=5)
    age_cols_enrol = [col for col in sample_df.columns if col.startswith('age_')]
    enrol_cols = base_cols + age_cols_enrol
    
    # OPTIMIZATION: Use concat with ignore_index after reading all
    df_enrol = pd.concat(
        [pd.read_csv(f, usecols=enrol_cols, dtype={col: 'float32' for col in age_cols_enrol}) 
         for f in enrol_files],
        ignore_index=True
    )
    df_enrol = preprocess_dataframe(df_enrol, "Enrollment")
    
    print("  Loading biometric data...")
    bio_files = glob.glob('api_data_aadhar_biometric/*.csv')
    sample_df = pd.read_csv(bio_files[0], nrows=5)
    age_cols_bio = [col for col in sample_df.columns if col.startswith('bio_age')]
    bio_cols = base_cols + age_cols_bio
    
    df_bio = pd.concat(
        [pd.read_csv(f, usecols=bio_cols, dtype={col: 'float32' for col in age_cols_bio}) 
         for f in bio_files],
        ignore_index=True
    )
    df_bio = preprocess_dataframe(df_bio, "Biometric")
    
    print("  Loading demographic data...")
    demo_files = glob.glob('api_data_aadhar_demographic/*.csv')
    sample_df = pd.read_csv(demo_files[0], nrows=5)
    age_cols_demo = [col for col in sample_df.columns if col.startswith('demo_age')]
    demo_cols = base_cols + age_cols_demo
    
    df_demo = pd.concat(
        [pd.read_csv(f, usecols=demo_cols, dtype={col: 'float32' for col in age_cols_demo}) 
         for f in demo_files],
        ignore_index=True
    )
    df_demo = preprocess_dataframe(df_demo, "Demographic")
    
    return df_enrol, df_bio, df_demo

# ==========================================
# ML FUNCTIONS (ALL INLINE - NO SEPARATE FILE NEEDED)
# ==========================================

def calculate_multimodal_fraud_score(df_enrol, df_bio, df_demo):
    """Multi-modal fraud detection combining 5 signals"""
    age_cols_enrol = [col for col in df_enrol.columns if col.startswith('age_')]
    age_cols_bio = [col for col in df_bio.columns if col.startswith('bio_age')]
    age_cols_demo = [col for col in df_demo.columns if col.startswith('demo_age')]
    
    # SIGNAL 1: Benford's Law
    benford_scores = {}
    for pincode in df_enrol['pincode'].unique():
        pincode_activity = df_enrol[df_enrol['pincode'] == pincode][age_cols_enrol].sum().sum()
        if pincode_activity > 100:
            data = df_enrol[df_enrol['pincode'] == pincode][age_cols_enrol].sum(axis=1)
            data = data[data > 0]
            if len(data) >= 10:
                first_digits = data.astype(str).str[0].astype(int)
                actual_freq = first_digits.value_counts(normalize=True)
                benford_freq = pd.Series({d: math.log10(1 + 1/d) for d in range(1, 10)})
                deviation = sum([abs(actual_freq.get(d, 0) - benford_freq.get(d, 0)) for d in range(1, 10)])
                benford_scores[pincode] = min(deviation * 100, 100)
            else:
                benford_scores[pincode] = 0
        else:
            benford_scores[pincode] = 0
    
    # SIGNAL 2: Isolation Forest
    isolation_scores = {}
    if len(df_enrol) >= 10:
        pincode_agg = df_enrol.groupby('pincode')[age_cols_enrol].sum().sum(axis=1).reset_index()
        pincode_agg.columns = ['pincode', 'total_activity']
        if len(pincode_agg) >= 10:
            X = pincode_agg[['total_activity']].values
            clf = IsolationForest(contamination=0.1, random_state=42)
            scores = clf.fit(X).decision_function(X)
            scores_normalized = ((scores - scores.min()) / (scores.max() - scores.min())) * 100
            scores_normalized = 100 - scores_normalized
            for idx, pincode in enumerate(pincode_agg['pincode']):
                isolation_scores[pincode] = scores_normalized[idx]
        else:
            isolation_scores = {p: 0 for p in df_enrol['pincode'].unique()}
    else:
        isolation_scores = {p: 0 for p in df_enrol['pincode'].unique()}
    
    # SIGNAL 3: Volatility
    volatility_scores = {}
    for pincode in df_enrol['pincode'].unique():
        pincode_ts = df_enrol[df_enrol['pincode'] == pincode].groupby('date')[age_cols_enrol].sum().sum(axis=1)
        if len(pincode_ts) > 2:
            mean_val, std_val = pincode_ts.mean(), pincode_ts.std()
            cv = (std_val / mean_val) if mean_val > 0 else 0
            volatility_scores[pincode] = min(cv * 50, 100)
        else:
            volatility_scores[pincode] = 0
    
    # SIGNAL 4: Cross-Dataset Ratio
    ratio_scores = {}
    enrol_by_pin = df_enrol.groupby('pincode')[age_cols_enrol].sum().sum(axis=1) if age_cols_enrol else pd.Series()
    bio_by_pin = df_bio.groupby('pincode')[age_cols_bio].sum().sum(axis=1) if age_cols_bio else pd.Series()
    for pincode in df_enrol['pincode'].unique():
        enrol_val = enrol_by_pin.get(pincode, 0)
        bio_val = bio_by_pin.get(pincode, 0)
        if enrol_val > 0:
            ratio = bio_val / enrol_val
            if ratio < 0.7:
                deviation = (0.7 - ratio) * 100
            elif ratio > 1.2:
                deviation = (ratio - 1.2) * 50
            else:
                deviation = 0
            ratio_scores[pincode] = min(deviation, 100)
        else:
            ratio_scores[pincode] = 0
    
    # SIGNAL 5: Outlier Frequency
    outlier_scores = {}
    for pincode in df_enrol['pincode'].unique():
        pincode_ts = df_enrol[df_enrol['pincode'] == pincode].groupby('date')[age_cols_enrol].sum().sum(axis=1)
        if len(pincode_ts) > 3:
            mean_val, std_val = pincode_ts.mean(), pincode_ts.std()
            threshold = mean_val + (2.5 * std_val)
            outlier_days = (pincode_ts > threshold).sum()
            outlier_freq = (outlier_days / len(pincode_ts)) * 100
            outlier_scores[pincode] = min(outlier_freq * 2, 100)
        else:
            outlier_scores[pincode] = 0
    
    # Combine signals
    all_pincodes = list(set(list(benford_scores.keys()) + list(isolation_scores.keys()) + 
                             list(volatility_scores.keys()) + list(ratio_scores.keys()) + list(outlier_scores.keys())))
    results = []
    for pincode in all_pincodes:
        composite_score = (benford_scores.get(pincode, 0) * 0.30 + isolation_scores.get(pincode, 0) * 0.25 +
                          outlier_scores.get(pincode, 0) * 0.20 + volatility_scores.get(pincode, 0) * 0.15 +
                          ratio_scores.get(pincode, 0) * 0.10)
        risk_level = "CRITICAL" if composite_score >= 70 else "HIGH" if composite_score >= 50 else "MEDIUM" if composite_score >= 30 else "LOW"
        signals = {'Benford': benford_scores.get(pincode, 0), 'Isolation': isolation_scores.get(pincode, 0),
                  'Volatility': volatility_scores.get(pincode, 0), 'Ratio': ratio_scores.get(pincode, 0),
                  'Outlier': outlier_scores.get(pincode, 0)}
        results.append({'pincode': pincode, 'fraud_score': round(composite_score, 2), 'risk_level': risk_level,
                       'top_signal': max(signals, key=signals.get), 'benford_score': round(benford_scores.get(pincode, 0), 2),
                       'isolation_score': round(isolation_scores.get(pincode, 0), 2), 'volatility_score': round(volatility_scores.get(pincode, 0), 2),
                       'ratio_score': round(ratio_scores.get(pincode, 0), 2), 'outlier_score': round(outlier_scores.get(pincode, 0), 2)})
    return pd.DataFrame(results).sort_values('fraud_score', ascending=False)


def predict_pincode_churn(df_enrol, prediction_window=30):
    """OPTIMIZED churn prediction using vectorized operations"""
    age_cols = [col for col in df_enrol.columns if col.startswith('age_')]
    today = df_enrol['date'].max()
    df_enrol['total_activity'] = df_enrol[age_cols].sum(axis=1)
    
    last_activity = df_enrol.groupby('pincode')['date'].max()
    days_since_last = (today - last_activity).dt.days
    
    last_7d_mask = df_enrol['date'] >= (today - pd.Timedelta(days=7))
    prev_7d_mask = (df_enrol['date'] >= (today - pd.Timedelta(days=14))) & (df_enrol['date'] < (today - pd.Timedelta(days=7)))
    last_7d_activity = df_enrol[last_7d_mask].groupby('pincode')['total_activity'].sum()
    prev_7d_activity = df_enrol[prev_7d_mask].groupby('pincode')['total_activity'].sum()
    activity_trend_7d = ((last_7d_activity - prev_7d_activity) / prev_7d_activity * 100).fillna(0).replace([np.inf, -np.inf], 0)
    
    pincode_stats = df_enrol.groupby('pincode').agg({'date': ['min', 'max', 'count'], 'total_activity': ['sum', 'mean'], 'district': 'first'})
    pincode_stats.columns = ['_'.join(col).strip('_') for col in pincode_stats.columns]
    total_days = (pincode_stats['date_max'] - pincode_stats['date_min']).dt.days + 1
    activity_frequency = (pincode_stats['date_count'] / total_days).fillna(0)
    recent_avg_volume = last_7d_activity / 7
    overall_avg_volume = pincode_stats['total_activity_mean']
    
    # Align all series to the same index
    activity_trend_7d_aligned = activity_trend_7d.reindex(days_since_last.index, fill_value=0)
    activity_frequency_aligned = activity_frequency.reindex(days_since_last.index, fill_value=0)
    recent_avg_volume_aligned = recent_avg_volume.reindex(days_since_last.index, fill_value=0)
    overall_avg_volume_aligned = overall_avg_volume.reindex(days_since_last.index, fill_value=1)
    
    churn_risk = pd.Series(0.0, index=days_since_last.index)
    churn_risk = churn_risk + np.where(days_since_last.values > 14, 40, np.where(days_since_last.values > 7, 20, np.where(days_since_last.values > 3, 10, 0)))
    churn_risk = churn_risk + np.where(activity_trend_7d_aligned.values < -50, 30, np.where(activity_trend_7d_aligned.values < -20, 15, 0))
    churn_risk = churn_risk + np.where(activity_frequency_aligned.values < 0.2, 20, np.where(activity_frequency_aligned.values < 0.4, 10, 0))
    volume_ratio = recent_avg_volume_aligned / overall_avg_volume_aligned
    churn_risk = churn_risk + np.where(volume_ratio.values < 0.3, 10, 0)
    churn_risk = churn_risk.clip(upper=100)
    risk_level = pd.cut(churn_risk, bins=[-np.inf, 30, 50, 70, np.inf], labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])
    
    results = pd.DataFrame({
        'pincode': churn_risk.index,
        'churn_probability': churn_risk.round(2).values,
        'risk_level': risk_level.astype(str).values,
        'days_since_last_activity': days_since_last.values,
        'activity_trend_7d': activity_trend_7d_aligned.round(2).values,
        'activity_frequency': activity_frequency_aligned.round(3).values,
        'avg_gap_days': 0,
        'recent_volume': recent_avg_volume_aligned.astype(int).values,
        'district': pincode_stats['district_first'].reindex(churn_risk.index, fill_value='Unknown').values
    })
    return results[pincode_stats['date_count'].reindex(results['pincode'].values).values >= 3].sort_values('churn_probability', ascending=False).reset_index(drop=True)


def predict_biometric_compliance(df_enrol, df_bio):
    """OPTIMIZED compliance prediction"""
    age_cols_enrol = [col for col in df_enrol.columns if col.startswith('age_')]
    age_cols_bio = [col for col in df_bio.columns if col.startswith('bio_age')]
    df_enrol['total_activity'] = df_enrol[age_cols_enrol].sum(axis=1) if age_cols_enrol else 0
    df_bio['total_activity'] = df_bio[age_cols_bio].sum(axis=1) if age_cols_bio else 0
    
    enrol_stats = df_enrol.groupby('pincode').agg({'total_activity': 'sum', 'date': ['mean', 'count'], 'district': 'first'})
    enrol_stats.columns = ['total_enrollments', 'avg_enrol_date', 'enrollment_volume', 'district']
    bio_stats = df_bio.groupby('pincode').agg({'total_activity': 'sum', 'date': 'mean'})
    bio_stats.columns = ['total_biometric', 'avg_bio_date']
    results = enrol_stats.join(bio_stats, how='left')
    
    # Fill numeric columns with 0
    results['total_biometric'] = results['total_biometric'].fillna(0)
    
    results = results[results['total_enrollments'] > 0].copy()
    results['historical_rate'] = (results['total_biometric'] / results['total_enrollments'] * 100).clip(upper=100)
    
    # Calculate days properly using vectorized operations with proper NaT handling
    avg_enrol = pd.to_datetime(results['avg_enrol_date'])
    avg_bio = pd.to_datetime(results['avg_bio_date'])
    
    # Calculate days difference, handling NaT properly
    days_diff = (avg_bio - avg_enrol).dt.days
    results['avg_days_to_complete'] = days_diff.fillna(0).clip(lower=0)
    
    results['predicted_rate'] = results['historical_rate'].copy()
    results.loc[results['enrollment_volume'] > 100, 'predicted_rate'] *= 0.9
    results.loc[results['avg_days_to_complete'] > 60, 'predicted_rate'] *= 0.85
    results['predicted_rate'] = results['predicted_rate'].clip(0, 100)
    results['risk_level'] = pd.cut(results['predicted_rate'], bins=[-np.inf, 40, 60, 75, np.inf], labels=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'])
    results['pending_enrollments'] = (results['total_enrollments'] - results['total_biometric']).clip(lower=0).astype(int)
    
    output = pd.DataFrame({
        'pincode': results.index, 'predicted_completion_rate': results['predicted_rate'].round(2).values,
        'historical_completion_rate': results['historical_rate'].round(2).values, 'risk_level': results['risk_level'].astype(str).values,
        'pending_enrollments': results['pending_enrollments'].values, 'avg_days_to_complete': results['avg_days_to_complete'].round(1).values,
        'enrollment_volume': results['enrollment_volume'].astype(int).values, 'district': results['district'].values
    })
    return output.sort_values('predicted_completion_rate', ascending=True).reset_index(drop=True)


def detect_cross_dataset_anomalies(df_enrol, df_bio, df_demo):
    """Cross-dataset anomaly detection"""
    age_cols_enrol = [col for col in df_enrol.columns if col.startswith('age_')]
    age_cols_bio = [col for col in df_bio.columns if col.startswith('bio_age')]
    age_cols_demo = [col for col in df_demo.columns if col.startswith('demo_age')]
    
    enrol_by_pin = df_enrol.groupby('pincode')[age_cols_enrol].sum().sum(axis=1) if age_cols_enrol else pd.Series()
    bio_by_pin = df_bio.groupby('pincode')[age_cols_bio].sum().sum(axis=1) if age_cols_bio else pd.Series()
    demo_by_pin = df_demo.groupby('pincode')[age_cols_demo].sum().sum(axis=1) if age_cols_demo else pd.Series()
    all_pincodes = set(df_enrol['pincode'].unique()) | set(df_bio['pincode'].unique()) | set(df_demo['pincode'].unique())
    
    results = []
    for pincode in all_pincodes:
        enrol_total, bio_total, demo_total = enrol_by_pin.get(pincode, 0), bio_by_pin.get(pincode, 0), demo_by_pin.get(pincode, 0)
        bio_enrol_ratio = (bio_total / enrol_total) if enrol_total > 0 else 0
        demo_enrol_ratio = (demo_total / enrol_total) if enrol_total > 0 else 0
        
        anomaly_type, is_anomaly, severity = "Normal", False, "LOW"
        if enrol_total > 100 and bio_total < enrol_total * 0.3:
            anomaly_type, is_anomaly, severity = "Ghost Enrollments (High Enroll, Low Bio)", True, "CRITICAL"
        elif bio_total > enrol_total * 1.5 and enrol_total > 0:
            anomaly_type, is_anomaly, severity = "Phantom Updates (High Bio, Low Enroll)", True, "HIGH"
        elif demo_total > enrol_total * 1.2 and enrol_total > 0:
            anomaly_type, is_anomaly, severity = "Impossible Demo/Enroll Ratio", True, "HIGH"
        elif enrol_total > 50 and bio_total == 0:
            anomaly_type, is_anomaly, severity = "Zero Biometric Coverage", True, "CRITICAL"
        elif bio_enrol_ratio > 0 and bio_enrol_ratio < 0.2:
            anomaly_type, is_anomaly, severity = "Severe Update Lag", True, "MEDIUM"
        
        anomaly_score = 0
        if bio_enrol_ratio < 0.3 or bio_enrol_ratio > 1.5:
            anomaly_score += abs(bio_enrol_ratio - 0.85) * 50
        if demo_enrol_ratio > 1.0:
            anomaly_score += (demo_enrol_ratio - 1.0) * 30
        anomaly_score = min(anomaly_score, 100)
        
        results.append({
            'pincode': pincode, 'is_anomaly': is_anomaly, 'anomaly_type': anomaly_type, 'severity': severity,
            'anomaly_score': round(anomaly_score, 2), 'enroll_total': int(enrol_total), 'bio_total': int(bio_total),
            'demo_total': int(demo_total), 'bio_enroll_ratio': round(bio_enrol_ratio, 3), 'demo_enroll_ratio': round(demo_enrol_ratio, 3),
            'district': df_enrol[df_enrol['pincode'] == pincode]['district'].iloc[0] if pincode in df_enrol['pincode'].values and 'district' in df_enrol.columns else 'Unknown'
        })
    
    result_df = pd.DataFrame(results)
    anomalies_only = result_df[result_df['is_anomaly'] == True].sort_values('anomaly_score', ascending=False)
    return result_df, anomalies_only


def predict_peak_load_days(df_enrol, forecast_days=30):
    """Peak load forecasting - SIMPLIFIED (Prophet can be slow, returns empty if < 7 days data)"""
    try:
        from prophet import Prophet
        age_cols = [col for col in df_enrol.columns if col.startswith('age_')]
        daily_data = df_enrol.groupby('date')[age_cols].sum().sum(axis=1).reset_index()
        daily_data.columns = ['ds', 'y']
        if len(daily_data) < 7:
            return pd.DataFrame()
        
        daily_data['day_of_week'] = pd.to_datetime(daily_data['ds']).dt.dayofweek
        daily_data['is_weekend'] = daily_data['day_of_week'].isin([5, 6]).astype(int)
        daily_data['is_monday'] = (daily_data['day_of_week'] == 0).astype(int)
        
        model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False, interval_width=0.80)
        model.add_regressor('is_weekend')
        model.add_regressor('is_monday')
        model.fit(daily_data[['ds', 'y', 'is_weekend', 'is_monday']])
        
        future = model.make_future_dataframe(periods=forecast_days)
        future['day_of_week'] = pd.to_datetime(future['ds']).dt.dayofweek
        future['is_weekend'] = future['day_of_week'].isin([5, 6]).astype(int)
        future['is_monday'] = (future['day_of_week'] == 0).astype(int)
        forecast = model.predict(future)
        
        threshold_high, threshold_low = forecast['yhat'].quantile(0.80), forecast['yhat'].quantile(0.20)
        forecast['load_level'] = 'Normal'
        forecast.loc[forecast['yhat'] >= threshold_high, 'load_level'] = 'Peak'
        forecast.loc[forecast['yhat'] <= threshold_low, 'load_level'] = 'Low'
        forecast['day_name'] = pd.to_datetime(forecast['ds']).dt.day_name()
        forecast['recommended_staff'] = (forecast['yhat'] / forecast['yhat'].mean() * 10).round(0).astype(int)
        
        today = daily_data['ds'].max()
        future_predictions = forecast[forecast['ds'] > today].copy()
        results = future_predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'load_level', 'day_name', 'recommended_staff']].copy()
        results.columns = ['date', 'predicted_load', 'lower_bound', 'upper_bound', 'load_level', 'day_name', 'recommended_staff']
        return results
    except:
        return pd.DataFrame()


def predict_age_group_transitions(df_enrol):
    """Age group transition prediction"""
    age_cols = [col for col in df_enrol.columns if col.startswith('age_')]
    results, today = [], df_enrol['date'].max()
    
    for district in df_enrol['district'].unique():
        district_df = df_enrol[df_enrol['district'] == district].copy()
        current_5_17 = district_df['age_5_17'].sum()
        if current_5_17 == 0:
            continue
        
        district_df['total_5_17'] = district_df['age_5_17']
        # Convert dates to timestamp safely
        dates_as_int = pd.to_datetime(district_df['date']).astype(np.int64) / 10**9
        weighted_dates = (dates_as_int * district_df['total_5_17']).sum()
        total_weight = district_df['total_5_17'].sum()
        if total_weight == 0:
            continue
        
        avg_enrollment_timestamp = weighted_dates / total_weight
        avg_enrollment_date = pd.to_datetime(avg_enrollment_timestamp, unit='s')
        predicted_transition_date = avg_enrollment_date + pd.DateOffset(years=5)
        
        # Calculate days difference safely
        days_until_transition = int((predicted_transition_date - pd.Timestamp(today)).days)
        
        monthly_transition_rate = current_5_17 / 60
        if days_until_transition <= 0:
            priority, urgency_score = "OVERDUE", 100
        elif days_until_transition <= 365:
            priority, urgency_score = "CRITICAL", 90
        elif days_until_transition <= 730:
            priority, urgency_score = "HIGH", 70
        elif days_until_transition <= 1095:
            priority, urgency_score = "MEDIUM", 50
        else:
            priority, urgency_score = "LOW", 30
        
        cohort_size_factor = min(current_5_17 / 1000, 5)
        urgency_score = min(urgency_score * (1 + cohort_size_factor * 0.1), 100)
        
        results.append({
            'district': district, 'current_5_17_cohort': int(current_5_17), 'avg_enrollment_date': avg_enrollment_date.date(),
            'predicted_transition_date': predicted_transition_date.date(), 'days_until_transition': max(0, days_until_transition),
            'priority': priority, 'urgency_score': round(urgency_score, 2), 'monthly_transition_rate': round(monthly_transition_rate, 1),
            'required_monthly_capacity': round(monthly_transition_rate, 0), 'campaign_start_date': (predicted_transition_date - pd.DateOffset(months=6)).date()
        })
    
    return pd.DataFrame(results).sort_values('urgency_score', ascending=False) if results else pd.DataFrame()

# ==========================================
# DATA LOADING
# ==========================================

start_time = time.time()
df_enrol, df_bio, df_demo = load_data()
load_time = time.time() - start_time
print(f"✓ Loaded {len(df_enrol):,} enrollment records")
print(f"✓ Loaded {len(df_bio):,} biometric records")
print(f"✓ Loaded {len(df_demo):,} demographic records")
print(f"✓ Loading time: {load_time:.1f} seconds\n")

# OPTIMIZATION: Optional sampling for faster testing/development
USE_SAMPLING = False  # Set to True for faster testing
SAMPLE_SIZE = 100000

if USE_SAMPLING and len(df_enrol) > SAMPLE_SIZE:
    print(f"⚡ SAMPLING MODE: Using {SAMPLE_SIZE:,} records for faster processing")
    df_enrol = df_enrol.sample(n=SAMPLE_SIZE, random_state=42)
    # Filter other datasets to matching pincodes
    sampled_pincodes = df_enrol['pincode'].unique()
    df_bio = df_bio[df_bio['pincode'].isin(sampled_pincodes)]
    df_demo = df_demo[df_demo['pincode'].isin(sampled_pincodes)]
    print(f"  → Enrollment: {len(df_enrol):,} records")
    print(f"  → Biometric: {len(df_bio):,} records") 
    print(f"  → Demographic: {len(df_demo):,} records\n")

# ==========================================
# GENERATE ML PREDICTIONS (WITH INCREMENTAL SAVING)
# ==========================================
print("🤖 Running ML models (this may take several minutes)...\n")

output_file = 'ml_predictions_cache.json'
ml_results = {}
model_times = {}

# Load existing cache if available to resume from where we left off
import os
if os.path.exists(output_file):
    try:
        with open(output_file, 'r') as f:
            ml_results = json.load(f)
        print("📦 Found existing cache - will skip completed models\n")
    except:
        ml_results = {}

def save_incremental():
    """Save progress after each model"""
    with open(output_file, 'w') as f:
        json.dump(ml_results, f, indent=2)

# 1. Multi-Modal Fraud Score
if 'fraud_scores' not in ml_results or not ml_results['fraud_scores']:
    print("1/6 Calculating multi-modal fraud scores...")
    t0 = time.time()
    fraud_scores = calculate_multimodal_fraud_score(df_enrol, df_bio, df_demo)
    ml_results['fraud_scores'] = fraud_scores.to_dict(orient='records')
    model_times['fraud_scores'] = time.time() - t0
    print(f"    ✓ Analyzed {len(fraud_scores)} pincodes for fraud risk ({model_times['fraud_scores']:.1f}s)")
    save_incremental()
    print(f"    💾 Saved fraud scores to cache")
else:
    print("1/6 ⏭ Skipping fraud scores (already cached)")

# 2. Churn Prediction
if 'churn_predictions' not in ml_results or not ml_results['churn_predictions']:
    print("2/6 Predicting pincode churn risk...")
    t0 = time.time()
    churn_predictions = predict_pincode_churn(df_enrol, prediction_window=30)
    ml_results['churn_predictions'] = churn_predictions.to_dict(orient='records')
    model_times['churn_predictions'] = time.time() - t0
    print(f"    ✓ Generated churn predictions for {len(churn_predictions)} pincodes ({model_times['churn_predictions']:.1f}s)")
    save_incremental()
    print(f"    💾 Saved churn predictions to cache")
else:
    print("2/6 ⏭ Skipping churn predictions (already cached)")

# 3. Biometric Compliance Prediction
if 'compliance_predictions' not in ml_results or not ml_results['compliance_predictions']:
    print("3/6 Predicting biometric compliance...")
    t0 = time.time()
    compliance_predictions = predict_biometric_compliance(df_enrol, df_bio)
    ml_results['compliance_predictions'] = compliance_predictions.to_dict(orient='records')
    model_times['compliance_predictions'] = time.time() - t0
    print(f"    ✓ Predicted compliance for {len(compliance_predictions)} pincodes ({model_times['compliance_predictions']:.1f}s)")
    save_incremental()
    print(f"    💾 Saved compliance predictions to cache")
else:
    print("3/6 ⏭ Skipping compliance predictions (already cached)")

# 4. Cross-Dataset Anomaly Detection
if 'cross_anomalies_only' not in ml_results or not ml_results.get('cross_anomalies_all'):
    print("4/6 Detecting cross-dataset anomalies...")
    t0 = time.time()
    cross_anomalies_all, cross_anomalies_only = detect_cross_dataset_anomalies(df_enrol, df_bio, df_demo)
    ml_results['cross_anomalies_all'] = cross_anomalies_all.to_dict(orient='records')
    ml_results['cross_anomalies_only'] = cross_anomalies_only.to_dict(orient='records')
    model_times['cross_anomalies'] = time.time() - t0
    print(f"    ✓ Found {len(cross_anomalies_only)} anomalous pincodes ({model_times['cross_anomalies']:.1f}s)")
    save_incremental()
    print(f"    💾 Saved anomaly detection to cache")
else:
    print("4/6 ⏭ Skipping anomaly detection (already cached)")

# 5. Peak Load Forecasting
if 'peak_load_forecast' not in ml_results:
    print("5/6 Forecasting peak load days...")
    t0 = time.time()
    peak_load_forecast = predict_peak_load_days(df_enrol, forecast_days=30)
    if not peak_load_forecast.empty:
        # Convert datetime to string for JSON serialization
        peak_load_forecast['date'] = peak_load_forecast['date'].astype(str)
        ml_results['peak_load_forecast'] = peak_load_forecast.to_dict(orient='records')
        model_times['peak_load_forecast'] = time.time() - t0
        print(f"    ✓ Generated forecasts for {len(peak_load_forecast)} days ({model_times['peak_load_forecast']:.1f}s)")
    else:
        ml_results['peak_load_forecast'] = []
        model_times['peak_load_forecast'] = time.time() - t0
        print(f"    ⚠ Insufficient data for peak load forecasting ({model_times['peak_load_forecast']:.1f}s)")
    save_incremental()
    print(f"    💾 Saved peak load forecast to cache")
else:
    print("5/6 ⏭ Skipping peak load forecast (already cached)")

# 6. Age Group Transition Prediction
if 'transition_predictions' not in ml_results:
    print("6/6 Predicting age group transitions...")
    t0 = time.time()
    transition_predictions = predict_age_group_transitions(df_enrol)
    if not transition_predictions.empty:
        # Convert dates to strings for JSON serialization
        date_cols = ['avg_enrollment_date', 'predicted_transition_date', 'campaign_start_date']
        for col in date_cols:
            if col in transition_predictions.columns:
                transition_predictions[col] = transition_predictions[col].astype(str)
        ml_results['transition_predictions'] = transition_predictions.to_dict(orient='records')
        model_times['transition_predictions'] = time.time() - t0
        print(f"    ✓ Generated transitions for {len(transition_predictions)} districts ({model_times['transition_predictions']:.1f}s)")
    else:
        ml_results['transition_predictions'] = []
        model_times['transition_predictions'] = time.time() - t0
        print(f"    ⚠ Insufficient data for age group transitions ({model_times['transition_predictions']:.1f}s)")
    save_incremental()
    print(f"    💾 Saved age transitions to cache")
else:
    print("6/6 ⏭ Skipping age transitions (already cached)")

# ==========================================
# FINALIZE RESULTS WITH METADATA
# ==========================================
print("\n💾 Finalizing results with metadata...")

# Add/update metadata
ml_results['metadata'] = {
    'generated_at': datetime.now().isoformat(),
    'data_records': {
        'enrollment': len(df_enrol),
        'biometric': len(df_bio),
        'demographic': len(df_demo)
    },
    'predictions': {
        'fraud_scores': len(ml_results.get('fraud_scores', [])),
        'churn_predictions': len(ml_results.get('churn_predictions', [])),
        'compliance_predictions': len(ml_results.get('compliance_predictions', [])),
        'cross_anomalies': len(ml_results.get('cross_anomalies_only', [])),
        'peak_load_forecast': len(ml_results.get('peak_load_forecast', [])),
        'transition_predictions': len(ml_results.get('transition_predictions', []))
    }
}

# Final save with metadata
with open(output_file, 'w') as f:
    json.dump(ml_results, f, indent=2)

print(f"✓ Results saved to: {output_file}")
print(f"✓ File size: {round(len(json.dumps(ml_results)) / 1024, 2)} KB")

print("\n" + "=" * 60)
print("✅ ML PREDICTIONS GENERATED SUCCESSFULLY!")
print("=" * 60)
print(f"Generated at: {ml_results['metadata']['generated_at']}")
print("\nSummary:")
for key, value in ml_results['metadata']['predictions'].items():
    print(f"  • {key.replace('_', ' ').title()}: {value} records")

print("\n⏱ Performance Breakdown:")
total_ml_time = sum(model_times.values())
print(f"  • Data Loading: {load_time:.1f}s")
for model, duration in model_times.items():
    pct = (duration / total_ml_time * 100) if total_ml_time > 0 else 0
    print(f"  • {model.replace('_', ' ').title()}: {duration:.1f}s ({pct:.1f}%)")
print(f"  • Total ML Processing: {total_ml_time:.1f}s")
print(f"  • Total Runtime: {load_time + total_ml_time:.1f}s")

print("\n💡 The main app will now load these results instantly!")
print("   To regenerate, simply run this script again.")
print("=" * 60)
