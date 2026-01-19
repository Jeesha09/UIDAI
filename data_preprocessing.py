"""
Centralized Data Preprocessing Module for UIDAI Analytics
Handles normalization, cleaning, and feature engineering for all datasets
"""

import pandas as pd
import numpy as np
import re
import unicodedata

# ==========================================
# VALID STATES AND UNION TERRITORIES
# ==========================================
VALID_STATES_UTS = {
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
    "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi", "Puducherry", "Chandigarh",
    "Lakshadweep", "Andaman And Nicobar Islands",
    "Dadra And Nagar Haveli And Daman And Diu",
    "Jammu And Kashmir", "Ladakh"
}

# ==========================================
# STATE ALIASES MAPPING
# ==========================================
STATE_ALIASES = {
    "Andaman & Nicobar Islands": "Andaman And Nicobar Islands",
    "Dadra & Nagar Haveli": "Dadra And Nagar Haveli And Daman And Diu",
    "Dadra And Nagar Haveli": "Dadra And Nagar Haveli And Daman And Diu",
    "Daman & Diu": "Dadra And Nagar Haveli And Daman And Diu",
    "Daman And Diu": "Dadra And Nagar Haveli And Daman And Diu",
    "The Dadra And Nagar Haveli And Daman And Diu": "Dadra And Nagar Haveli And Daman And Diu",
    "Jammu & Kashmir": "Jammu And Kashmir",
    "Orissa": "Odisha",
    "Pondicherry": "Puducherry",
    "Westbengal": "West Bengal",
    "West  Bengal": "West Bengal",
    "West Bangal": "West Bengal",
    "West Bengli": "West Bengal",
    "Andhra Pradesh ": "Andhra Pradesh",
    "Chhatisgarh": "Chhattisgarh",
    "Uttaranchal": "Uttarakhand",
    
    # Spelling variations
    "Tamilnadu": "Tamil Nadu",
    
    # Districts mistakenly entered as states (map to correct state)
    "Darbhanga": "Bihar",
    "Jaipur": "Rajasthan",
    "Nagpur": "Maharashtra",
    
    # Neighborhoods/localities mistakenly entered as states (map to correct state)
    "Balanagar": "Telangana",
    "Madanapalle": "Andhra Pradesh",
    "Puttenahalli": "Karnataka",
    "Raja Annamalai Puram": "Tamil Nadu",
}

# ==========================================
# DISTRICT ALIASES MAPPING
# ==========================================
DISTRICT_ALIASES = {
    # 1. BANGALORE / BENGALURU VARIANTS
    "Bangalore": "Bengaluru",
    "Bangalore Rural": "Bengaluru Rural",
    "Bengaluru Urban": "Bengaluru",

    # 1.5. SIKKIM DISTRICTS (cardinal directions need state suffix)
    "East": "East Sikkim",
    "West": "West Sikkim",
    "North": "North Sikkim",
    "South": "South Sikkim",

    # 2. INVERSIONS (West Bengal/Bihar alignment)
    "24 Paraganas North": "North 24 Parganas",
    "24 Paraganas South": "South 24 Parganas",
    "Dinajpur Dakshin": "Dakshin Dinajpur",
    "Dinajpur Uttar": "Uttar Dinajpur",
    "North Twenty Four Parganas": "North 24 Parganas",
    "South Twenty Four Parganas": "South 24 Parganas",
    "South 24 Pargana": "South 24 Parganas",

    # 3. ADMINISTRATIVE RENAMES
    "Ahmadnagar": "Ahilyanagar", "Ahmed Nagar": "Ahilyanagar", "Ahmednagar": "Ahilyanagar",
    "Aurangabad": "Chhatrapati Sambhajinagar", "Chatrapati Sambhaji Nagar": "Chhatrapati Sambhajinagar",
    "Osmanabad": "Dharashiv", "Bid": "Beed",
    "Allahabad": "Prayagraj", "Faizabad": "Ayodhya",
    "Hoshangabad": "Narmadapuram",
    "Gurgaon": "Gurugram",
    "Mewat": "Nuh",
    "Spsr Nellore": "Nellore",

    # 4. SPELLING & TYPO FIXES
    "East Singhbum": "East Singhbhum",
    "Purbi Champaran": "East Champaran",
    "Purba Champaran": "East Champaran",
    "Kushi Nagar": "Kushinagar",
    "Gaurella Pendra Marwahi": "Gaurela Pendra Marwahi",
    "Visakhapatanam": "Visakhapatnam",
    "Ananthapur": "Anantapur", "Ananthapuramu": "Anantapur",
    "Anugal": "Angul", "Ashok Nagar": "Ashoknagar", "Buldana": "Buldhana",
    "Darjiling": "Darjeeling", "Davanagere": "Davangere", "Hooghiy": "Hooghly",
    "Jangoan": "Jangaon", "Jhunjhunun": "Jhunjhunu", "Kasargod": "Kasaragod",
    "Khorda": "Khordha", "Maldah": "Malda", "Nabarangapur": "Nabarangpur",
    "Nicobars": "Nicobar", "Puruliya": "Purulia", "Sheikpura": "Sheikhpura",
    "Shrawasti": "Shravasti", "Sundergarh": "Sundargarh",
    "Baleswar": "Baleshwar", "Banas Kantha": "Banaskantha", "Sabar Kantha": "Sabarkantha",
    "Ferozepur": "Firozpur", "Hazaribag": "Hazaribagh",
    "Jagatsinghapur": "Jagatsinghpur", "Mysore": "Mysuru", "Tumkur": "Tumakuru",
    "Yamuna Nagar": "Yamunanagar", "Haridwar": "Hardwar", "Shimoga": "Shivamogga",
    "Kancheepuram": "Kanchipuram",

    # 5. TELANGANA / ANDHRA ALIGNMENT
    "K.V. Rangareddy": "Rangareddy", "K.V.Rangareddy": "Rangareddy",
    "Ranga Reddy": "Rangareddy", "Rangareddi": "Rangareddy",
    "Hanumakonda": "Hanamkonda", "Karim Nagar": "Karimnagar", "Mahabub Nagar": "Mahabubnagar",

    # 6. ENCODING / MOJIBAKE CLEANUP
    "Medchal-Malkajgiri": "Medchal Malkajgiri",
    "Medchal?Malkajgiri": "Medchal Malkajgiri",
    "Medchalâˆ'Malkajgiri": "Medchal Malkajgiri",
    "Medchala Malkajgiri": "Medchal Malkajgiri",
    "Medchalmalkajgiri": "Medchal Malkajgiri",

    # 7. REGIONAL CONSISTENCY
    "Cooch Behar": "Koch Bihar", "Coochbehar": "Koch Bihar",
    "East Midnapur": "East Midnapore", "West Midnapore": "West Medinipur",
    "Medinipur West": "Paschim Medinipur"
}

# ==========================================
# NORMALIZATION FUNCTIONS
# ==========================================
def normalize_district_name(district):
    """Clean and standardize district names"""
    if pd.isna(district) or str(district).strip() in ['?', '']:
        return None

    # Fix Encoding/Unicode
    d = unicodedata.normalize("NFKD", str(district))
    
    # Remove mojibake/encoding artifacts
    d = d.encode('ascii', 'ignore').decode('ascii')
    
    # Strip footnotes, parentheticals, and dots/stars
    d = re.sub(r"\(.*?\)|\*+|\.+", "", d)
    
    # Handle special separators and "Dist :" prefixes
    d = d.replace("Dist :", "").replace("&", "and")
    d = re.sub(r"[-–—−?]", " ", d)  # Replace dashes with space
    
    # Standardize whitespace and case
    d = " ".join(d.split()).title()
    
    # Map aliases first (before filtering generic names)
    d = DISTRICT_ALIASES.get(d, d)
    
    # Filter out invalid entries after alias mapping
    if d.isdigit() or len(d) < 2:
        return None
        
    return d


def normalize_state_name(state):
    """Clean and standardize state names"""
    if pd.isna(state):
        return None
    s = " ".join(str(state).strip().title().split())
    if s.isdigit():
        return None
    s = STATE_ALIASES.get(s, s)
    return s if s in VALID_STATES_UTS else None


def normalize_district_vectorized(series):
    """Vectorized district normalization - faster than .apply()"""
    # Remove NaN and invalid values upfront
    series = series.fillna('')
    series = series.astype(str).str.strip()
    series = series.replace(['?', ''], None)
    
    # Basic cleaning (vectorized)
    series = series.str.replace(r"\(.*?\)|\*+|\.+", "", regex=True)
    series = series.str.replace("Dist :", "", regex=False)
    series = series.str.replace("&", "and", regex=False)
    series = series.str.replace(r"[-–—−?]", " ", regex=True)
    series = series.str.strip().str.title()
    series = series.apply(lambda x: " ".join(str(x).split()) if pd.notna(x) else None)
    
    # Apply aliases
    series = series.map(lambda x: DISTRICT_ALIASES.get(x, x) if pd.notna(x) else None)
    
    # Filter invalid
    series = series.map(lambda x: None if (pd.notna(x) and (str(x).isdigit() or len(str(x)) < 2)) else x)
    
    return series


def normalize_state_vectorized(series):
    """Vectorized state normalization - faster than .apply()"""
    # Basic cleaning (vectorized)
    series = series.fillna('')
    series = series.astype(str).str.strip().str.title()
    series = series.apply(lambda x: " ".join(str(x).split()) if x else None)
    
    # Filter numeric
    series = series.map(lambda x: None if (pd.notna(x) and str(x).isdigit()) else x)
    
    # Apply aliases
    series = series.map(lambda x: STATE_ALIASES.get(x, x) if pd.notna(x) else None)
    
    # Validate
    series = series.map(lambda x: x if (pd.notna(x) and x in VALID_STATES_UTS) else None)
    
    return series


# ==========================================
# MAIN PREPROCESSING FUNCTION
# ==========================================
def preprocess_dataframe(df, dataset_name="Dataset"):
    """
    Clean and prepare data for analysis using centralized normalization
    
    Args:
        df: pandas DataFrame with raw data
        dataset_name: Name for logging purposes
        
    Returns:
        Cleaned and enriched DataFrame
    """
    print(f"\n{'='*60}")
    print(f"PREPROCESSING {dataset_name}")
    print(f"{'='*60}")
    
    initial_len = len(df)
    
    # 1. TEMPORAL CONVERSION - DO THIS FIRST!
    if 'date' in df.columns:
        before_date = len(df)
        # Store original dates before parsing
        df['date_original'] = df['date'].astype(str)
        
        # Try multiple date formats (YYYY-MM-DD, DD-MM-YYYY)
        df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
        
        # Check for invalid dates
        invalid_date_mask = df['date'].isna()
        invalid_date_count = invalid_date_mask.sum()
        valid_dates = before_date - invalid_date_count
        
        if invalid_date_count > 0:
            # Show sample of invalid date formats
            invalid_date_samples = df[invalid_date_mask]['date_original'].value_counts().head(10)
            print(f"⚠ Date parsing: {valid_dates:,}/{before_date:,} valid ({invalid_date_count:,} FAILED TO PARSE)")
            print(f"   Invalid date formats found:")
            for date_val, count in invalid_date_samples.items():
                print(f"     - '{date_val}': {count:,} rows")
        else:
            print(f"✓ Parsed {valid_dates:,} valid dates out of {before_date:,} records")
        
        if valid_dates > 0:
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['month_name'] = df['date'].dt.month_name()
            df['day_of_week'] = df['date'].dt.day_name()
            df['week'] = df['date'].dt.isocalendar().week
            
            # Drop records with invalid dates
            df = df.dropna(subset=['date'])
            df.drop(columns=['date_original'], inplace=True)
    
    # 2. GEOGRAPHY NORMALIZATION (VECTORIZED FOR SPEED)
    if 'state' in df.columns:
        before_state = len(df)
        # Store original values before normalization
        df['state_original'] = df['state'].copy()
        df['state'] = normalize_state_vectorized(df['state'])
        after_state = df['state'].notna().sum()
        invalid_states = before_state - after_state
        
        if invalid_states > 0:
            # Show actual invalid state values (where normalize returned None)
            invalid_mask = df['state'].isna() & df['state_original'].notna()
            if invalid_mask.sum() > 0:
                invalid_state_samples = df[invalid_mask]['state_original'].value_counts().head(10)
                print(f"  → State normalization: {after_state:,}/{before_state:,} valid ({invalid_states:,} invalid)")
                print(f"     Invalid state values found:")
                for state, count in invalid_state_samples.items():
                    print(f"       - '{state}': {count:,} rows")
            else:
                print(f"  → State normalization: {after_state:,}/{before_state:,} valid")
        else:
            print(f"  → State normalization: {after_state:,}/{before_state:,} valid")
        
        # Drop temporary column
        df.drop(columns=['state_original'], inplace=True)
        
    if 'district' in df.columns:
        before_district = len(df)
        # Store original values before normalization
        df['district_original'] = df['district'].copy()
        df['district'] = normalize_district_vectorized(df['district'])
        after_district = df['district'].notna().sum()
        invalid_districts = before_district - after_district
        
        if invalid_districts > 0:
            # Show actual invalid district values (where normalize returned None)
            invalid_mask = df['district'].isna() & df['district_original'].notna()
            if invalid_mask.sum() > 0:
                invalid_district_samples = df[invalid_mask]['district_original'].value_counts().head(10)
                print(f"  → District normalization: {after_district:,}/{before_district:,} valid ({invalid_districts:,} invalid)")
                print(f"     Invalid district values found:")
                for district, count in invalid_district_samples.items():
                    print(f"       - '{district}': {count:,} rows")
            else:
                print(f"  → District normalization: {after_district:,}/{before_district:,} valid")
        else:
            print(f"  → District normalization: {after_district:,}/{before_district:,} valid")
        
        # Drop temporary column
        df.drop(columns=['district_original'], inplace=True)
    
    # 3. DROP INVALID RECORDS
    if 'state' in df.columns and 'district' in df.columns:
        before_drop = len(df)
        df = df.dropna(subset=['state', 'district'])
        after_drop = len(df)
        dropped_count = before_drop - after_drop
        if dropped_count > 0:
            print(f"  → Dropped rows with missing state/district: {dropped_count:,}")
    
    # 4. REGIONAL CLASSIFICATION
    if 'state' in df.columns:
        north_states = {
            'Uttar Pradesh', 'Punjab', 'Haryana', 'Himachal Pradesh',
            'Jammu And Kashmir', 'Uttarakhand', 'Delhi', 'Rajasthan'
        }
        df['region'] = df['state'].apply(lambda x: 'North' if x in north_states else 'South')
    
    # 5. URBAN/RURAL CLASSIFICATION
    if 'district' in df.columns:
        urban_keywords = ['Urban', 'City', 'Municipal', 'Corporation', 'Metro']
        df['area_type'] = df['district'].apply(
            lambda x: 'Urban' if any(kw in str(x) for kw in urban_keywords) else 'Rural'
        )
    
    # 6. SUMMARY STATISTICS
    final_len = len(df)
    dropped = initial_len - final_len
    
    print(f"✓ Records: {final_len:,} (Dropped {dropped:,} invalid rows)")
    if 'state' in df.columns:
        print(f"✓ Unique States: {df['state'].nunique()}")
    if 'district' in df.columns:
        print(f"✓ Unique Districts: {df['district'].nunique()}")
    if 'date' in df.columns:
        print(f"✓ Date Range: {df['date'].min()} to {df['date'].max()}")
    
    return df


# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def get_valid_states():
    """Return list of valid Indian states and UTs"""
    return sorted(list(VALID_STATES_UTS))


def get_state_aliases():
    """Return state alias mapping dictionary"""
    return STATE_ALIASES.copy()


def get_district_aliases():
    """Return district alias mapping dictionary"""
    return DISTRICT_ALIASES.copy()
