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
    "Andhra Pradesh ": "Andhra Pradesh",
}

# ==========================================
# DISTRICT ALIASES MAPPING
# ==========================================
DISTRICT_ALIASES = {
    # 1. BANGALORE / BENGALURU VARIANTS
    "Bangalore": "Bengaluru",
    "Bangalore Rural": "Bengaluru Rural",
    "Bengaluru Urban": "Bengaluru",

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
    
    # Filter out invalid entries
    if d in {"East", "West", "North", "South"} or d.isdigit() or len(d) < 2:
        return None
        
    # Map aliases
    return DISTRICT_ALIASES.get(d, d)


def normalize_state_name(state):
    """Clean and standardize state names"""
    if pd.isna(state):
        return None
    s = " ".join(str(state).strip().title().split())
    if s.isdigit():
        return None
    s = STATE_ALIASES.get(s, s)
    return s if s in VALID_STATES_UTS else None


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
        # Try multiple date formats
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Check if dates were parsed
        valid_dates = df['date'].notna().sum()
        print(f"✓ Parsed {valid_dates:,} valid dates out of {len(df):,} records")
        
        if valid_dates > 0:
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['month_name'] = df['date'].dt.month_name()
            df['day_of_week'] = df['date'].dt.day_name()
            df['week'] = df['date'].dt.isocalendar().week
            
            # Drop records with invalid dates
            df = df.dropna(subset=['date'])
    
    # 2. GEOGRAPHY NORMALIZATION
    if 'state' in df.columns:
        df['state'] = df['state'].apply(normalize_state_name)
    if 'district' in df.columns:
        df['district'] = df['district'].apply(normalize_district_name)
    
    # 3. DROP INVALID RECORDS
    if 'state' in df.columns and 'district' in df.columns:
        df = df.dropna(subset=['state', 'district'])
    
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
