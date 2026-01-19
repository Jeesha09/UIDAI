import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import grangercausalitytests
from prophet import Prophet
import math
from data_preprocessing import preprocess_dataframe

# ==========================================
# CLASS 1: Executive Summary Engine
# ==========================================
class ExecutiveSummaryEngine:
    def __init__(self, df_enrollment, df_biometric, df_demographic):
        self.df_enrol = df_enrollment
        self.df_bio = df_biometric
        self.df_demo = df_demographic
        
        # Merge key metrics for district-level analysis
        self.daily_district_stats = self.df_enrol.groupby(['date', 'state', 'district']).agg({
            'age_0_5': 'sum',
            'age_18_greater': 'sum'
        }).reset_index()

    def get_early_warning_system(self):
        df = self.daily_district_stats.copy()
        # Calculate 7-day Moving Average
        df['7_day_avg'] = df.groupby('district')['age_0_5'].transform(lambda x: x.rolling(window=7).mean())
        
        latest_date = df['date'].max()
        last_week_start = latest_date - pd.Timedelta(days=7)
        prev_week_start = last_week_start - pd.Timedelta(days=7)

        recent_data = df[df['date'] > last_week_start].groupby('district')['age_0_5'].sum()
        prev_data = df[(df['date'] <= last_week_start) & (df['date'] > prev_week_start)].groupby('district')['age_0_5'].sum()

        risk_df = pd.DataFrame({'recent': recent_data, 'previous': prev_data})
        risk_df['change_pct'] = ((risk_df['recent'] - risk_df['previous']) / risk_df['previous']) * 100
        
        high_risk_districts = risk_df[risk_df['change_pct'] < -20].copy()
        high_risk_districts['status'] = 'Critical Decline'
        
        return {
            "metric_value": len(high_risk_districts),
            "details_df": high_risk_districts.sort_values('change_pct').head(10)
        }

    def get_stagnation_detection(self):
        latest_date = self.df_enrol['date'].max()
        cutoff_date = latest_date - pd.Timedelta(days=30)
        all_pincodes = self.df_enrol['pincode'].unique()
        active_pincodes = self.df_enrol[self.df_enrol['date'] > cutoff_date]['pincode'].unique()
        inactive_pincodes = np.setdiff1d(all_pincodes, active_pincodes)
        
        stagnant_df = pd.DataFrame(inactive_pincodes, columns=['pincode'])
        stagnant_df['status'] = 'Inactive (30+ Days)'
        
        return {
            "total_stagnant": len(stagnant_df),
            "pincode_list": stagnant_df['pincode'].tolist()
        }

    def get_peer_benchmarking(self):
        district_ranks = self.df_enrol.groupby(['state', 'district'])['age_0_5'].sum().reset_index()
        district_ranks.rename(columns={'age_0_5': 'total_performance'}, inplace=True)
        district_ranks['rank'] = district_ranks['total_performance'].rank(ascending=False)
        
        top_5 = district_ranks.sort_values('total_performance', ascending=False).head(5)
        bottom_5 = district_ranks.sort_values('total_performance', ascending=True).head(5)
        
        return {"top_performers": top_5, "bottom_performers": bottom_5}

    def get_location_clusters(self):
        # FIX: Group by BOTH State and District to handle duplicate district names
        features = self.df_enrol.groupby(['state', 'district']).agg({
            'age_0_5': 'sum', 'age_18_greater': 'sum'
        }).reset_index()
        
        features['total_volume'] = features['age_0_5'] + features['age_18_greater']
        # Avoid division by zero
        features['adult_ratio'] = features.apply(
            lambda x: x['age_18_greater'] / x['total_volume'] if x['total_volume'] > 0 else 0, axis=1
        )
        features = features.fillna(0)
        
        # Dynamic Cluster Count
        n_districts = len(features)
        n_clusters = max(1, min(4, n_districts))
        
        if n_districts < 1:
            return pd.DataFrame() 

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features[['total_volume', 'adult_ratio']])
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        features['cluster_id'] = kmeans.fit_predict(X_scaled)
        
        def label_cluster(row):
            return f"Cluster {row['cluster_id']}"

        features['cluster_label'] = features.apply(label_cluster, axis=1)
        
        # FIX: Return 'state' in the final output
        return features[['state', 'district', 'total_volume', 'cluster_label', 'cluster_id']]
# ==========================================
# CLASS 2: Aadhaar Analytics Engine
# ==========================================
class AadhaarAnalyticsEngine:
    def __init__(self, df_enrollment, df_biometric, df_demographic):
        self.df_enrol = df_enrollment
        self.df_bio = df_biometric
        self.df_demo = df_demographic
        
        # Ensure date format (preprocessing should have already done this, but double-check)
        if 'date' in self.df_enrol.columns:
            self.df_enrol['date'] = pd.to_datetime(self.df_enrol['date'], errors='coerce')
        if 'date' in self.df_bio.columns:
            self.df_bio['date'] = pd.to_datetime(self.df_bio['date'], errors='coerce')
        if 'date' in self.df_demo.columns:
            self.df_demo['date'] = pd.to_datetime(self.df_demo['date'], errors='coerce')

    def get_center_of_gravity(self, district_name):
        district_data = self.df_bio[self.df_bio['district'] == district_name].copy()
        
        # Mock Lat/Long if missing (Replace with real geocoding in production)
        if 'lat' not in district_data.columns:
            district_data['lat'] = np.random.uniform(18.5, 19.5, len(district_data))
            district_data['long'] = np.random.uniform(72.5, 73.5, len(district_data))

        district_data['weight'] = district_data['bio_age_5_17']
        total_weight = district_data['weight'].sum()
        
        if total_weight == 0: return None

        optimal_lat = (district_data['lat'] * district_data['weight']).sum() / total_weight
        optimal_long = (district_data['long'] * district_data['weight']).sum() / total_weight
        
        return {
            "optimal_location": (optimal_lat, optimal_long),
            "existing_locations": district_data[['lat', 'long', 'weight']].to_dict('records')
        }

    def get_service_strain_matrix(self):
        enrollment = self.df_enrol.groupby('district')['age_0_5'].sum().reset_index()
        enrollment.rename(columns={'age_0_5': 'new_demand'}, inplace=True)
        
        updates = self.df_demo.groupby('district')['demo_age_17_'].sum().reset_index()
        updates.rename(columns={'demo_age_17_': 'maintenance_load'}, inplace=True)
        
        matrix_df = pd.merge(enrollment, updates, on='district')
        
        median_x = matrix_df['new_demand'].median()
        median_y = matrix_df['maintenance_load'].median()
        
        def assign_quadrant(row):
            if row['new_demand'] >= median_x and row['maintenance_load'] >= median_y: return "Warzone"
            elif row['new_demand'] >= median_x: return "Nursery"
            elif row['maintenance_load'] >= median_y: return "Hub"
            else: return "Dormant"
                
        matrix_df['category'] = matrix_df.apply(assign_quadrant, axis=1)
        return matrix_df

    def get_forecast_prophet(self, district_name, days_forward=30):
        df = self.df_enrol[self.df_enrol['district'] == district_name].copy()
        if df.empty: return pd.DataFrame(columns=['ds', 'yhat', 'yhat_lower', 'yhat_upper'])
        
        daily_data = df.groupby('date')['age_0_5'].sum().reset_index()
        daily_data.columns = ['ds', 'y']
        
        if len(daily_data) < 2: return pd.DataFrame(columns=['ds', 'yhat', 'yhat_lower', 'yhat_upper'])

        model = Prophet(yearly_seasonality=True, daily_seasonality=False)
        model.fit(daily_data)
        future = model.make_future_dataframe(periods=days_forward)
        forecast = model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days_forward)

    def get_ripple_effect(self, district_a, district_b):
        ts_a = self.df_demo[self.df_demo['district'] == district_a].groupby('date')['demo_age_17_'].sum()
        ts_b = self.df_demo[self.df_demo['district'] == district_b].groupby('date')['demo_age_17_'].sum()
        
        combined = pd.concat([ts_a, ts_b], axis=1).dropna()
        combined.columns = ['A', 'B']
        
        if len(combined) < 5: return {"error": "Not enough data"}

        try:
            test_result = grangercausalitytests(combined[['B', 'A']], maxlag=[3], verbose=False)
            p_value = test_result[3][0]['ssr_chi2test'][1]
            return {"is_significant": p_value < 0.05, "p_value": p_value}
        except:
            return {"error": "Calculation failed"}

    def get_age_distribution(self, district_name):
        df = self.df_enrol[self.df_enrol['district'] == district_name]
        return {
            "0-5 Years": df['age_0_5'].sum(),
            "5-17 Years": df['age_5_17'].sum(),
            "18+ Years": df['age_18_greater'].sum()
        }

    def get_update_lag_analysis(self, district_name):
        new_adults = self.df_enrol[self.df_enrol['district'] == district_name]['age_18_greater'].sum()
        updated_adults = self.df_demo[self.df_demo['district'] == district_name]['demo_age_17_'].sum()
        ratio = updated_adults / new_adults if new_adults > 0 else 0
        return {"engagement_ratio": round(ratio, 2)}

    def check_benfords_law(self, district_name):
        df = self.df_enrol[self.df_enrol['district'] == district_name]
        data = df[df['age_18_greater'] > 0]['age_18_greater']
        
        if len(data) < 10: return {"error": "Not enough data points"}
            
        first_digits = data.astype(str).str[0].astype(int)
        digit_counts = first_digits.value_counts(normalize=True).sort_index()
        benford_probs = {d: math.log10(1 + 1/d) for d in range(1, 10)}
        
        results = []
        for d in range(1, 10):
            actual = digit_counts.get(d, 0)
            expected = benford_probs[d]
            results.append({"digit": d, "actual_freq": actual, "benford_freq": expected, "deviation": abs(actual - expected)})
            
        fraud_score = sum([r['deviation'] for r in results])
        return {"fraud_score": fraud_score, "distribution": pd.DataFrame(results)}

    def detect_anomalies_isolation_forest(self):
        data = self.df_enrol.groupby(['date', 'pincode'])['age_18_greater'].sum().reset_index()
        if len(data) < 10: return pd.DataFrame()
        
        X = data[['age_18_greater']].values
        clf = IsolationForest(contamination=0.01, random_state=42)
        data['anomaly'] = clf.fit_predict(X)
        return data[data['anomaly'] == -1].sort_values('age_18_greater', ascending=False)

    def get_benfords_law_global(self):
        """Global Benford's Law analysis across all data"""
        import math
        
        # Get all age columns
        age_cols = [col for col in self.df_enrol.columns if col.startswith('age_')]
        
        # Calculate total activity per record
        self.df_enrol['total_activity'] = self.df_enrol[age_cols].sum(axis=1)
        
        # Get leading digits (excluding zeros)
        data = self.df_enrol[self.df_enrol['total_activity'] > 0]['total_activity']
        
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
            'deviation_score': deviation,
            'is_suspicious': deviation > 0.15
        }

    def get_statistical_outliers(self):
        """Detect anomalous activity spikes over time"""
        age_cols = [col for col in self.df_enrol.columns if col.startswith('age_')]
        
        # Daily aggregation
        daily = self.df_enrol.groupby('date')[age_cols].sum().sum(axis=1).reset_index()
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

    def get_volatility_analysis(self, top_n=20):
        """Analyze centers with erratic/inconsistent behavior"""
        age_cols = [col for col in self.df_enrol.columns if col.startswith('age_')]
        
        # Aggregate by pincode
        volatility = self.df_enrol.groupby('pincode')[age_cols].sum().sum(axis=1).reset_index()
        volatility.columns = ['pincode', 'total_activity']
        
        # Calculate variance score per pincode over time
        pincode_stats = self.df_enrol.groupby('pincode').agg({
            age_cols[0]: ['mean', 'std', 'count']
        }).reset_index()
        pincode_stats.columns = ['pincode', 'mean', 'std', 'count']
        
        # Calculate coefficient of variation
        pincode_stats['variance_score'] = (pincode_stats['std'] / pincode_stats['mean']).fillna(0)
        
        # Filter centers with sufficient data and sort by variance
        erratic = pincode_stats[pincode_stats['count'] > 5].sort_values('variance_score', ascending=False).head(top_n)
        
        return erratic

    def get_state_variance_data(self):
        """Analyze consistency across states using box plots"""
        age_cols = [col for col in self.df_enrol.columns if col.startswith('age_')]
        
        # Calculate activity per district
        state_data = self.df_enrol.groupby(['state', 'district'])[age_cols].sum().sum(axis=1).reset_index()
        state_data.columns = ['state', 'district', 'total_activity']
        
        return state_data

    def get_bcg_matrix_data(self):
        """Service Strain Matrix with BCG-style quadrants"""
        # Calculate metrics
        age_cols_enrol = [col for col in self.df_enrol.columns if col.startswith('age_')]
        age_cols_demo = [col for col in self.df_demo.columns if col.startswith('demo_age')]
        age_cols_bio = [col for col in self.df_bio.columns if col.startswith('bio_age')]
        
        enrollments = self.df_enrol.groupby('district')[age_cols_enrol].sum().sum(axis=1).reset_index()
        enrollments.columns = ['district', 'total_enrollments']
        
        updates_demo = self.df_demo.groupby('district')[age_cols_demo].sum().sum(axis=1).reset_index() if age_cols_demo else pd.DataFrame(columns=['district', 'total_updates'])
        updates_bio = self.df_bio.groupby('district')[age_cols_bio].sum().sum(axis=1).reset_index() if age_cols_bio else pd.DataFrame(columns=['district', 'total_updates'])
        
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

    def get_mobile_van_priority_data(self):
        """Identify high-priority areas for mobile van deployment"""
        age_cols_demo = [col for col in self.df_demo.columns if col.startswith('demo_age')]
        age_cols_bio = [col for col in self.df_bio.columns if col.startswith('bio_age')]
        
        # Combine enrollment and update data by pincode
        enrol_pincode = self.df_enrol.groupby(['district', 'pincode'])['age_18_greater'].sum().reset_index()
        enrol_pincode.columns = ['district', 'pincode', 'total_enrollments']
        
        updates = pd.DataFrame()
        if age_cols_demo:
            updates_demo = self.df_demo.groupby(['district', 'pincode'])[age_cols_demo].sum().sum(axis=1).reset_index()
            updates_demo.columns = ['district', 'pincode', 'total_updates']
            updates = updates_demo
        
        if age_cols_bio:
            updates_bio = self.df_bio.groupby(['district', 'pincode'])[age_cols_bio].sum().sum(axis=1).reset_index()
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

    def get_center_productivity_data(self, top_n=20):
        """Top performing centers by activity volume"""
        van_data = self.get_mobile_van_priority_data()
        top_centers = van_data.nlargest(top_n, 'total_activity')
        return top_centers

    def get_weekly_capacity_data(self):
        """Weekly activity patterns for capacity planning"""
        if 'day_of_week' not in self.df_enrol.columns:
            return pd.DataFrame()
        
        age_cols = [col for col in self.df_enrol.columns if col.startswith('age_')]
        weekly_data = self.df_enrol.groupby('day_of_week')[age_cols].sum().sum(axis=1).reset_index()
        weekly_data.columns = ['day_of_week', 'total_activity']
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_data['day_of_week'] = pd.Categorical(weekly_data['day_of_week'], categories=day_order, ordered=True)
        weekly_data = weekly_data.sort_values('day_of_week')
        
        return weekly_data

    def get_growth_velocity_data(self, top_n=5):
        """Enrollment growth velocity for top districts"""
        if 'date' not in self.df_enrol.columns:
            return pd.DataFrame()
        
        age_cols = [col for col in self.df_enrol.columns if col.startswith('age_')]
        
        # Get top districts by total volume
        top_districts = self.df_enrol.groupby('district')[age_cols].sum().sum(axis=1).nlargest(top_n).index
        
        # Time series data for top districts
        velocity_data = self.df_enrol[self.df_enrol['district'].isin(top_districts)].groupby(['date', 'district'])[age_cols].sum().sum(axis=1).reset_index()
        velocity_data.columns = ['date', 'district', 'total_enrollments']
        velocity_data = velocity_data.sort_values('date')
        
        return velocity_data