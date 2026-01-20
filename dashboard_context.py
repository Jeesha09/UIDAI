import re

# We define the context with simple keys (no emojis needed here)
RAW_CONTEXT = {
    "Executive Summary": """
    - **Location Archetypes (Cluster Map)**: 
        - **What is it?**: A map of India where districts are colored by their 'Cluster' (0, 1, 2, 3).
        - **Cluster 0 (Blue)**: High Volume / Balanced Growth. These are stable major districts.
        - **Cluster 1 (Orange)**: Rapid Growth / High Churn. These areas are growing fast but have many updates (people moving or correcting data).
        - **Cluster 2 (Green)**: Stagnant / Rural. Low volume and low updates.
        - **Cluster 3 (Red)**: Critical Decline. Enrollment numbers are dropping here.
    - **Critical Alerts**: Shows districts with >20% decline in enrollment.
    """,
    
    "Operations & Logistics": """
    - **Mobile Van Solver**: Shows the optimal lat/long coordinates to park a mobile service van to cover the most people.
    - **Service Strain Matrix**: A scatter plot comparing New Enrollments vs Update Requests.
    """,
    
    "Trends & Forecasting": """
    - **Forecast Chart**: Uses the Facebook Prophet model to predict enrollment volume for the next 30 days.
    - **Ripple Effect**: Shows how a spike in one district affects neighbors.
    """,
    
    "Demographics & Policy": """
    - **Age Cohort**: Shows the split between 0-5, 5-17, and 18+ age groups.
    """,
    
    "Security & Integrity": """
    - **Benford's Law**: Detects fraud. If the blue bars don't match the red curve, the data might be fake/manipulated.
    """
}

def normalize_key(text):
    """Removes emojis and extra spaces to ensure matching works."""
    # Remove non-ASCII characters (emojis) and strip whitespace
    return re.sub(r'[^\x00-\x7F]+', '', text).strip()

def get_page_context(page_name):
    # 1. Clean the incoming page name (e.g. "🏠 Executive Summary" -> "Executive Summary")
    clean_name = normalize_key(page_name)
    
    # 2. Look up in our dictionary
    # We loop through keys to find a partial match (e.g. "Executive" in "Executive Summary")
    for key, context in RAW_CONTEXT.items():
        if key in clean_name:
            return context
            
    return "No specific chart context available for this page."