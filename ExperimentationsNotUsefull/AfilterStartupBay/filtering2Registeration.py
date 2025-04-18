import pandas as pd
import numpy as np
import json
from datetime import datetime

# Load the dataset
df = pd.read_csv("./AMergedCsvFiles/merged_business_registrations.csv")

# Create destination directory
# os.makedirs("AMergedCsvFiles2", exist_ok=True)

# 1. Bay Area Filtering
bay_area_cities = [
    'San Francisco', 'Oakland', 'San Jose', 'Berkeley',
    'Redwood City', 'San Mateo', 'Palo Alto', 'Millbrae',
    'South San Francisco', 'Daly City', 'San Rafael'
]

bay_area_df = df[
    (df['city'].isin(bay_area_cities)) |
    (df['mail_city'].isin(bay_area_cities))
].copy()

# 2. Startup Identification Criteria
is_startup = (
    (bay_area_df['business_type'] == 'Tech') |
    (bay_area_df['naic_code_description'].str.contains('Technology|Information|Software', na=False)) |
    (pd.to_datetime(bay_area_df['dba_start_date']).dt.year >= 2020)
)

bay_area_df['is_startup'] = is_startup

# 3. Handle Missing Values
# Calculate business age from start date
bay_area_df['dba_start_date'] = pd.to_datetime(bay_area_df['dba_start_date'])
current_year = datetime.now().year
bay_area_df['business_age_years'] = current_year - bay_area_df['dba_start_date'].dt.year

# Fill empty numeric fields
bay_area_df['supervisor_district'] = bay_area_df['supervisor_district'].fillna(-1)

# 4. GeoJSON Processing
def extract_coordinates(geojson):
    try:
        coords = json.loads(geojson.replace("'", '"'))['coordinates']
        return pd.Series({'longitude': coords[0], 'latitude': coords[1]})
    except:
        return pd.Series({'longitude': np.nan, 'latitude': np.nan})

bay_area_df[['longitude', 'latitude']] = bay_area_df['location'].apply(extract_coordinates)

# 5. Industry Classification
tech_industries = [
    'Information', 'Software', 'Technology', 'Computer',
    'Internet', 'Web Services', 'Artificial Intelligence'
]

bay_area_df['tech_industry'] = bay_area_df['naic_code_description'].str.contains(
    '|'.join(tech_industries), case=False, na=False
)

# 6. Enhanced Columns
bay_area_df['has_website'] = ~bay_area_df['ownership_name'].str.contains(
    'LLC|Inc|Corp|Ltd', regex=True
)

bay_area_df['employee_size'] = np.select(
    [
        bay_area_df['business_age_years'] < 3,
        bay_area_df['business_age_years'].between(3, 5),
        bay_area_df['business_age_years'] > 5
    ],
    ['Micro (1-10)', 'Small (11-50)', 'Medium (51-200)'],
    default='Unknown'
)

# 7. Final Cleanup
columns_to_keep = [
    'uniqueid', 'ownership_name', 'dba_name', 'full_business_address',
    'city', 'naic_code_description', 'latitude', 'longitude',
    'dba_start_date', 'business_age_years', 'is_startup', 'tech_industry',
    'employee_size', 'has_website'
]

clean_df = bay_area_df[columns_to_keep].drop_duplicates('uniqueid')

# Save enhanced file
clean_df.to_csv("AMergedCsvFiles2/enhanced_business_registrations.csv", index=False)

print(f"Cleaned dataset saved with {len(clean_df)} records")
print("New Features Added:")
print("- Tech industry classification")
print("- Startup identification flag")
print("- Business age calculation")
print("- Employee size estimation")
print("- Website presence indicator")
print("- Geographic coordinates extraction")
