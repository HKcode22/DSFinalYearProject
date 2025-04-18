import pandas as pd
import numpy as np
import os
import re
from datetime import datetime

# Create destination directory if it doesn't exist
os.makedirs("AMergedCsvFiles2", exist_ok=True)

# Load the data
print("Loading merged_base_companies.csv...")
df = pd.read_csv("./AMergedCsvFiles/merged_base_companies.csv")

print(f"Original dataset: {len(df)} companies")

# 1. Filter for Bay Area companies only
bay_area_locations = ['San Francisco', 'East Bay', 'South Bay', 'Peninsula', 'North Bay']
bay_area_df = df[df['Location'].isin(bay_area_locations)].copy()

print(f"After Bay Area filtering: {len(bay_area_df)} companies")

# 2. Clean and standardize the data
# Fix missing values and convert data types
bay_area_df['Founded Year'] = pd.to_numeric(bay_area_df['Founded Year'], errors='coerce')
current_year = datetime.now().year
bay_area_df['Company Age'] = current_year - bay_area_df['Founded Year']

# 3. Create company stage classification
conditions = [
    # Early-stage: <5 years OR 1-60 employees
    ((bay_area_df['Company Age'] < 5) | 
     (bay_area_df['Company Size'].isin(['1-12', '13-60']))),
    
    # Growth-stage: 5-10 years AND 61-250 employees
    ((bay_area_df['Company Age'] >= 5) & (bay_area_df['Company Age'] < 10) & 
     (bay_area_df['Company Size'].isin(['61-150', '151-250']))),
    
    # Mature: >10 years OR >250 employees
    ((bay_area_df['Company Age'] >= 10) | 
     (bay_area_df['Company Size'].isin(['251-499', '500-999', '1000-5000', '5001-9999', '10000'])))
]
choices = ['Early-stage', 'Growth-stage', 'Mature']
bay_area_df['Company Stage'] = np.select(conditions, choices, default='Unknown')

# 4. Industry Classification
# Extract primary industry from Tags
def extract_primary_industry(tags):
    if pd.isna(tags):
        return "Unknown"
    
    # Priority industries to check for
    priority_industries = [
        'Financial Technology', 'FinTech', 
        'Healthcare', 'Biotech', 
        'Consumer Goods', 
        'B2B Software', 
        'Education',
        'Real Estate',
        'Developer Tool',
        'AI', 'Artificial Intelligence',
        'Marketing',
        'Security'
    ]
    
    for industry in priority_industries:
        if industry in tags:
            return industry
    
    # If no priority industry found, return first tag
    first_tag = tags.split(',')[0]
    return first_tag

bay_area_df['Primary Industry'] = bay_area_df['Tags'].apply(extract_primary_industry)

# 5. Funding and Investor Status
bay_area_df['Has YC Funding'] = bay_area_df['Investors'].str.contains('Y Combinator', na=False)
bay_area_df['Has Known Investors'] = bay_area_df['Investors'].notna() & (bay_area_df['Investors'] != '')

# 6. Tech stack analysis
bay_area_df['Uses React'] = bay_area_df['Tech stack'].str.contains('React', na=False)
bay_area_df['Uses Ruby'] = bay_area_df['Tech stack'].str.contains('Ruby', na=False)
bay_area_df['Uses Marketo'] = bay_area_df['Marketing Stack'].str.contains('Marketo', na=False)

# 7. Final cleanup and save
enhanced_df = bay_area_df.reset_index(drop=True)

print(f"Enhanced dataset: {len(enhanced_df)} companies with additional classifications")
print("New columns added: Company Age, Company Stage, Primary Industry, Has YC Funding, Has Known Investors, Uses React, Uses Ruby, Uses Marketo")

# Save to the new directory
output_path = os.path.join("AMergedCsvFiles2", "enhanced_bay_area_companies.csv")
enhanced_df.to_csv(output_path, index=False)

print(f"Enhanced dataset saved to: {output_path}")


