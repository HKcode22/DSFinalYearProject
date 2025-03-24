import pandas as pd
import os

# Define base path for all files
base_path = "./AKnownData/"   # Go up one level from `dataCollectTesting1`


# --- Group 1: Merge duplicate base files ---
print("Merging base company datasets...")
base_cols = ['Company Name', 'Tags', 'Location', 'Investors', 'Description', 
             'Website', 'Founded Year', 'Address', 'Lat', 'Long', 
             'Company Size', 'Tech stack', 'Marketing Stack', 
             'Design Stack', 'Product Stack']

ba_base = pd.read_csv(os.path.join(base_path, 'bay_area_startups_base.csv'), usecols=base_cols)
ba_list = pd.read_csv(os.path.join(base_path, 'Bay-Area-Companies-List.csv'), usecols=base_cols)
merged_base = pd.concat([ba_base, ba_list]).drop_duplicates().reset_index(drop=True)

# --- Group 2: Merge Employbl-structured datasets ---
print("Merging Employbl-structured datasets...")
employbl_cols = ['Employbl Company ID', 'Company Name', 'Website', 'Address 1',
                 'City', 'State', 'Zip', 'Latitude', 'Longitude', 
                 'Company Description', 'Thumbnail URL']

datasets = [
    ('companies.csv', 'General Tech'),
    ('companies-that-use-laravel.csv', 'Laravel Users'),
    ('silicon-valley-companies.csv', 'Silicon Valley'),
    ('venture-capital.csv', 'Venture Capital'),
    ('y-combinator-companies.csv', 'Y Combinator')
]

merged_employbl = pd.DataFrame()
for file, category in datasets:
    df = pd.read_csv(os.path.join(base_path, file), usecols=employbl_cols)
    df['source_category'] = category
    merged_employbl = pd.concat([merged_employbl, df])

# --- Group 3: Merge Crunchbase datasets ---
print("Merging Crunchbase variants...")
crunch_alt = pd.read_csv(os.path.join(base_path, 'crunchbase_alternative_data.csv'))
crunch_comp = pd.read_csv(os.path.join(base_path, 'crunchbase_companies.csv'))
ba_crunch = pd.read_csv(os.path.join(base_path, 'bay_area_startups_crunchbase_dataset.csv'))

merged_crunch = pd.concat([crunch_alt, crunch_comp, ba_crunch], ignore_index=True, sort=False)

# --- Group 4: Merge geographic/registration files ---
print("Merging geographic datasets...")
oakland = pd.read_csv(os.path.join(base_path, 'tech-companies-in-oakland-06-20-2021.csv'))
oakland['city'] = 'Oakland'
sf_tech = pd.read_csv(os.path.join(base_path, 'san-francisco-tech-companies-06-30-2021.csv'))
sf_tech['city'] = 'San Francisco'
merged_snapshots = pd.concat([oakland, sf_tech])

registration_files = [
    ('datasf_businesses_raw.csv', 'Non-Tech'),
    ('sf_business_dataset.csv', 'Non-Tech'),
    ('datasf_tech_businesses.csv', 'Tech'),
    ('sf_tech_businesses1.csv', 'Tech')
]

merged_registrations = pd.DataFrame()
for file, biz_type in registration_files:
    df = pd.read_csv(os.path.join(base_path, file))
    df['business_type'] = biz_type
    merged_registrations = pd.concat([merged_registrations, df])

# --- Group 5: Merge miscellaneous datasets ---
print("Merging remaining files...")
misc_files = [
    os.path.join(base_path, f) for f in [
        'github_bay_area_startups.csv',
        'growth_list_startups.csv',
        'high_potential_startups.csv',
        'business_register_sample.csv',
        'recent_building_permits.csv'
    ]
]

links = pd.read_csv(os.path.join(base_path, 'bay_area_startups_combined_Links.csv'))
if 'website' in links.columns:
    links = links[['website']]

master_cols = ['company_name', 'location', 'state', 'founding_year', 'data_source']
ba_master = pd.read_csv(os.path.join(base_path, 'bay_area_startups_master.csv'), usecols=master_cols)

misc_dfs = [pd.read_csv(f) for f in misc_files] + [links, ba_master]
merged_misc = pd.concat(misc_dfs, axis=0, sort=False, ignore_index=True)

# --- Save merged datasets ---
merged_base.to_csv(os.path.join(base_path, 'merged_base_companies.csv'), index=False)
merged_employbl.to_csv(os.path.join(base_path, 'merged_employbl_companies.csv'), index=False)
merged_crunch.to_csv(os.path.join(base_path, 'merged_crunchbase.csv'), index=False)
merged_snapshots.to_csv(os.path.join(base_path, 'merged_tech_snapshots.csv'), index=False)
merged_registrations.to_csv(os.path.join(base_path, 'merged_business_registrations.csv'), index=False)
merged_misc.to_csv(os.path.join(base_path, 'merged_miscellaneous_data.csv'), index=False)

print("""
Merge completed successfully. Final files:
- merged_base_companies.csv
- merged_employbl_companies.csv
- merged_crunchbase.csv
- merged_tech_snapshots.csv
- merged_business_registrations.csv
- merged_miscellaneous_data.csv
""")
