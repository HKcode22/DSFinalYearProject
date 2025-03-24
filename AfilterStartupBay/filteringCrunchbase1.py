import pandas as pd
import numpy as np
import os
from datetime import datetime

# Configuration
INPUT_FILE = "./AMergedCsvFiles/merged_crunchbase.csv"
OUTPUT_DIR = "AMergedCsvFiles2"
OUTPUT_FILE = "cleaned_crunchbase.csv"

# Comprehensive Bay Area definition
BAY_AREA_REGIONS = {
    'San Francisco': ['San Francisco', 'Daly City', 'South San Francisco'],
    'Peninsula': ['San Mateo', 'Burlingame', 'Redwood City', 'Palo Alto', 'Menlo Park'],
    'South Bay': ['San Jose', 'Santa Clara', 'Sunnyvale', 'Mountain View', 'Cupertino'],
    'East Bay': ['Oakland', 'Berkeley', 'Emeryville', 'Fremont', 'Hayward'],
    'North Bay': ['San Rafael', 'Novato', 'Santa Rosa'],
    'Special Zones': ['Silicon Valley', 'SF Bay Area', 'Bay Area']
}

def load_data():
    """Load data with proper dtype handling and date parsing"""
    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=['founded_at', 'first_funding_at', 'last_funding_at'],
        dtype={
            'funding_total_usd': str,
            'employee_count': str,
            'growth_potential_score': str
        },
        low_memory=False
    )
    return df

def clean_numeric(df):
    """Robust numeric cleaning with multiple invalid pattern handling"""
    numeric_columns = ['funding_total_usd', 'employee_count', 'growth_potential_score']
    
    for col in numeric_columns:
        # Remove all non-numeric characters except digits, commas, and periods
        df[col] = df[col].astype(str).str.replace(r'[^\d.,-]', '', regex=True)
        
        # Replace long sequences of # symbols and other invalid patterns
        invalid_patterns = r'^#+$|^-+$|^n/?a$|^undefined$'
        df[col] = df[col].replace(invalid_patterns, np.nan, regex=True)
        
        # Convert to numeric with error coercing
        df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
        
        # Fill remaining NaNs with 0 for integer columns
        if col in ['employee_count']:
            df[col] = df[col].fillna(0).astype('Int64')
        else:
            df[col] = df[col].fillna(0.0)
    
    return df

def filter_bay_area(df):
    """Advanced geographic filtering using provided region definitions"""
    # Normalize region names
    df['region'] = df['region'].replace({
        'SF Bay Area': 'San Francisco Bay Area',
        'Bay Area': 'San Francisco Bay Area'
    })
    
    # Create comprehensive location filter
    bay_cities = [city for sublist in BAY_AREA_REGIONS.values() for city in sublist]
    region_filter = df['region'].isin(BAY_AREA_REGIONS.keys()) | df['city'].isin(bay_cities)
    
    return df[region_filter].copy()

def identify_startups(df):
    """Startup identification logic"""
    current_year = datetime.now().year
    df['company_age'] = current_year - df['founded_at'].dt.year.fillna(current_year)
    
    # Startup criteria: <10 years old and operational
    df['is_startup'] = (df['company_age'] < 10) & (df['status'] == 'operating')
    
    return df

def enhance_dataset(df):
    """Add analytical features"""
    # Funding stage classification
    df['funding_stage'] = pd.cut(
        df['funding_total_usd'],
        bins=[-np.inf, 1e6, 10e6, 50e6, np.inf],
        labels=['Seed', 'Early', 'Growth', 'Late'],
        include_lowest=True
    )
    
    # Industry categorization
    tech_keywords = r'Tech|Software|Cloud|AI|SaaS|Internet|Data'
    df['primary_industry'] = np.where(
        df['category_list'].str.contains(tech_keywords, case=False, na=False),
        'Technology',
        df['category_list'].str.split('|').str[0].fillna('Other')
    )
    
    return df

def handle_missing_data(df):
    """Comprehensive missing value treatment"""
    # Numeric columns
    df['funding_total_usd'] = df['funding_total_usd'].fillna(0)
    df['employee_count'] = df['employee_count'].fillna(0).astype(int)
    
    # Categorical columns
    df['status'] = df['status'].fillna('Unknown')
    df['category_list'] = df['category_list'].fillna('Uncategorized')
    
    return df

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and process data
    print("Loading data...")
    df = load_data()
    
    print("Cleaning numeric columns...")
    df = clean_numeric(df)
    
    print("Filtering Bay Area companies...")
    bay_df = filter_bay_area(df)
    
    print("Identifying startups...")
    startup_df = identify_startups(bay_df)
    
    print("Handling missing values...")
    cleaned_df = handle_missing_data(startup_df)
    
    print("Adding enhanced features...")
    final_df = enhance_dataset(cleaned_df)
    
    # Save results
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    final_df.to_csv(output_path, index=False)
    print(f"Saved {len(final_df)} Bay Area startups to {output_path}")

if __name__ == "__main__":
    main()