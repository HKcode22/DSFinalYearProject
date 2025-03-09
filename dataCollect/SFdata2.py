import requests
import pandas as pd
import os
from datetime import datetime, timedelta
import time

# Create directory for data storage
os.makedirs('datasf_startup_data', exist_ok=True)

# Your DataSF API token
APP_TOKEN = "7zYjZxGENqKygDNg8QbTmmtIB"  # Replace with your actual token

# Common headers for all API requests
headers = {
    "X-App-Token": APP_TOKEN
}

# Dictionary of confirmed dataset IDs from DataSF
datasets = {
    "registered_businesses": {
        "id": "g8m3-pdis",  # Registered Business Locations
        "description": "All registered business locations in San Francisco"
    },
    "business_register": {
        "id": "u6xu-nkq2",  # Business Register
        "description": "Business Register with additional details"
    },
    "building_permits": {
        "id": "i98e-djp9",  # Building Permits
        "description": "Building Permits"
    }
}

# Separate function for getting sample data (no pagination)
def get_sample_data(dataset_id, limit=5):
    """Get a small sample of data to understand structure"""
    base_url = f"https://data.sfgov.org/resource/{dataset_id}.json"
    params = {"$limit": limit}
    
    try:
        response = requests.get(base_url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting sample: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Error: {e}")
        return []

# Function for full data retrieval with pagination
def get_all_data(dataset_id, where_clause=None, select_clause=None, limit=1000, max_records=50000):
    """
    Retrieves data from a dataset with pagination up to max_records
    """
    base_url = f"https://data.sfgov.org/resource/{dataset_id}.json"
    offset = 0
    all_data = []
    
    while True:
        params = {
            "$limit": limit,
            "$offset": offset
        }
        
        if where_clause:
            params["$where"] = where_clause
            
        if select_clause:
            params["$select"] = select_clause
        
        try:
            print(f"Requesting data batch (offset {offset}, limit {limit})...")
            response = requests.get(base_url, headers=headers, params=params)
            
            if response.status_code == 429:
                print("Rate limit exceeded. Waiting 60 seconds...")
                time.sleep(60)
                continue
                
            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.text}")
                print(f"URL that caused error: {response.url}")
                break
                
            data_chunk = response.json()
            
            if not data_chunk:  # No more data
                break
                
            all_data.extend(data_chunk)
            print(f"Retrieved {len(data_chunk)} records. Total so far: {len(all_data)}")
            
            # If we got fewer records than the limit, we've reached the end
            if len(data_chunk) < limit:
                break
                
            # Check if we've reached the maximum records limit
            if len(all_data) >= max_records:
                print(f"Reached maximum records limit ({max_records})")
                break
                
            offset += limit
            
            # Be respectful of API limits
            time.sleep(1)
            
        except Exception as e:
            print(f"Error retrieving data: {e}")
            break
            
    return all_data

# 1. Get sample data to understand the structure
print("Examining registered businesses dataset structure...")
sample_business = get_sample_data(datasets["registered_businesses"]["id"])
if sample_business:
    print(f"Sample data fields: {list(sample_business[0].keys())}")
    
    # Check if NAICS code field exists
    naics_fields = [field for field in sample_business[0].keys() if 'naic' in field.lower()]
    print(f"Possible NAICS code fields: {naics_fields}")

# 2. Get registered businesses data
print("\nRetrieving registered businesses data...")
# Look for businesses started in the last 3 years
three_years_ago = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%dT00:00:00')
where_clause = f"location_start_date >= '{three_years_ago}'"

business_data = get_all_data(
    datasets["registered_businesses"]["id"], 
    where_clause=where_clause,
    limit=1000,  # Increased to 1000 records per request
    max_records=10000  # Set a reasonable maximum
)

# Convert to DataFrame
df_businesses = pd.DataFrame(business_data)

# Save the raw data
df_businesses.to_csv('datasf_startup_data/recent_businesses.csv', index=False)
print(f"Saved {len(df_businesses)} recent businesses to CSV")

# 3. Get additional Business data from Business Register
print("\nExamining business register dataset structure...")
sample_register = get_sample_data(datasets["business_register"]["id"])
if sample_register:
    print(f"Business register fields: {list(sample_register[0].keys())}")

# Try to find NAICS data or industry classification
print("\nRetrieving business register data for industry classification...")
register_data = get_all_data(
    datasets["business_register"]["id"],
    limit=1000,
    max_records=5000  # Limited sample
)

df_register = pd.DataFrame(register_data)
if not df_register.empty:
    # Look for industry classification fields
    industry_fields = [col for col in df_register.columns if any(term in col.lower() for term in ['industry', 'naic', 'business_type', 'class'])]
    print(f"Potential industry classification fields: {industry_fields}")
    
    df_register.to_csv('datasf_startup_data/business_register_sample.csv', index=False)
    print(f"Saved {len(df_register)} business register records to CSV")

# 4. Get Building Permits data
print("\nExamining building permits dataset structure...")
sample_permits = get_sample_data(datasets["building_permits"]["id"])
if sample_permits:
    print(f"Building permits fields: {list(sample_permits[0].keys())}")
    
    # Identify date and type fields
    date_fields = [field for field in sample_permits[0].keys() if 'date' in field.lower()]
    type_fields = [field for field in sample_permits[0].keys() if 'type' in field.lower()]
    
    print(f"Date fields: {date_fields}")
    print(f"Type fields: {type_fields}")

# Get recent permits
print("\nRetrieving recent building permits...")
one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%dT00:00:00')

# Use the identified date field, if available
date_field = date_fields[0] if 'date_fields' in locals() and date_fields else 'filed_date'
where_clause = f"{date_field} >= '{one_year_ago}'"

permit_data = get_all_data(
    datasets["building_permits"]["id"],
    where_clause=where_clause,
    limit=1000,
    max_records=5000
)

df_permits = pd.DataFrame(permit_data)
if not df_permits.empty:
    df_permits.to_csv('datasf_startup_data/recent_building_permits.csv', index=False)
    print(f"Saved {len(df_permits)} recent building permits to CSV")

# 5. Create a merged dataset with available fields
print("\nCreating a basic predictive dataset...")

if not df_businesses.empty:
    # Start with business data
    df_predictive = df_businesses.copy()
    
    # Calculate business age
    if 'location_start_date' in df_predictive.columns:
        try:
            df_predictive['location_start_date'] = pd.to_datetime(df_predictive['location_start_date'])
            current_date = datetime.now()
            df_predictive['business_age_years'] = (current_date - df_predictive['location_start_date']).dt.days / 365.25
        except Exception as e:
            print(f"Error calculating business age: {e}")
    
    # Save the dataset
    df_predictive.to_csv('datasf_startup_data/sf_business_dataset.csv', index=False)
    print(f"Saved initial dataset with {len(df_predictive)} records")
    
    # Print summary of available data
    print("\nAvailable data summary:")
    print(f"Total businesses in dataset: {len(df_predictive)}")
    if 'location_start_date' in df_predictive.columns:
        year_counts = df_predictive['location_start_date'].dt.year.value_counts().sort_index()
        print("\nBusinesses by year of establishment:")
        for year, count in year_counts.items():
            print(f"  {year}: {count}")
            
    print("\nTop business neighborhoods:")
    if 'neighborhoods_analysis_boundaries' in df_predictive.columns:
        neighborhood_counts = df_predictive['neighborhoods_analysis_boundaries'].value_counts().head(10)
        for neighborhood, count in neighborhood_counts.items():
            print(f"  {neighborhood}: {count}")
    elif 'neighborhood' in df_predictive.columns:
        neighborhood_counts = df_predictive['neighborhood'].value_counts().head(10)
        for neighborhood, count in neighborhood_counts.items():
            print(f"  {neighborhood}: {count}")
    
print("\nData collection complete.")
