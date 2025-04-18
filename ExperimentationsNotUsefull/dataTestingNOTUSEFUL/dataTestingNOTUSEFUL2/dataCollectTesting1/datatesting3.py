# import pandas as pd

# # Load the CSV files
# bay_area_df = pd.read_csv('bay_area_startups_base.csv')
# sv_df = pd.read_csv('silicon_valley_companies.csv')

# # Standardize column names (adjust based on actual columns in files)
# standard_columns = ['company_name', 'website', 'industry', 'location', 
#                    'description', 'founding_year', 'employee_count', 
#                    'funding_status', 'last_funding_amount']

# # Map existing columns to standard columns for each dataset
# # This step depends on the actual columns in your CSVs
# # Example:
# bay_area_df = bay_area_df.rename(columns={
#     'Company': 'company_name',
#     'Website': 'website',
#     # Add other mappings as needed
# })

# # Similarly for sv_df
# # ...

# # Fill missing columns with None
# for col in standard_columns:
#     if col not in bay_area_df.columns:
#         bay_area_df[col] = None
#     if col not in sv_df.columns:
#         sv_df[col] = None

# # Select only the standard columns
# bay_area_df = bay_area_df[standard_columns]
# sv_df = sv_df[standard_columns]

# # Merge the datasets and remove duplicates
# combined_df = pd.concat([bay_area_df, sv_df])
# combined_df = combined_df.drop_duplicates(subset=['company_name', 'website'])

# # Save the combined dataset
# combined_df.to_csv('bay_area_startups_combined.csv', index=False)


# import requests
# from bs4 import BeautifulSoup
# import pandas as pd
# import time

# # URLs for Y Combinator Bay Area startup pages
# yc_urls = [
#     "https://www.ycombinator.com/companies/location/san-francisco-bay-area",
#     "https://www.ycombinator.com/companies/industry/developer-tools/san-francisco-bay-area",
#     "https://www.ycombinator.com/companies/industry/saas/san-francisco-bay-area",
#     "https://www.ycombinator.com/companies/industry/data-engineering/san-francisco-bay-area"
# ]

# yc_startups = []

# for url in yc_urls:
#     # Add delay to avoid overwhelming the server
#     time.sleep(2)
    
#     response = requests.get(url, headers={
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
#     })
    
#     if response.status_code == 200:
#         soup = BeautifulSoup(response.text, 'html.parser')
        
#         # This selector will need to be adjusted based on actual HTML structure
#         startup_cards = soup.select('.company-card')
        
#         for card in startup_cards:
#             startup = {}
            
#             # Extract name, description, etc. (adjust selectors as needed)
#             name_elem = card.select_one('h3')
#             if name_elem:
#                 startup['company_name'] = name_elem.text.strip()
                
#             desc_elem = card.select_one('.description')
#             if desc_elem:
#                 startup['description'] = desc_elem.text.strip()
                
#             # Extract other available information
#             # ...
            
#             yc_startups.append(startup)

# # Convert to DataFrame
# yc_df = pd.DataFrame(yc_startups)

# # Save to CSV
# yc_df.to_csv('yc_bay_area_startups.csv', index=False)

# # Merge with previously combined data
# combined_df = pd.read_csv('bay_area_startups_combined.csv')
# yc_df = pd.read_csv('yc_bay_area_startups.csv')

# # Add a source column to track data origin
# combined_df['source'] = 'GitHub repos'
# yc_df['source'] = 'Y Combinator'

# # Merge and remove duplicates
# master_df = pd.concat([combined_df, yc_df])
# master_df = master_df.drop_duplicates(subset=['company_name', 'website'])

# # Save the updated master dataset
# master_df.to_csv('bay_area_startups_master.csv', index=False)


# import requests
# import pandas as pd
# import time
# from bs4 import BeautifulSoup

# # Example for StartupBlink (would need to be adjusted based on actual site structure)
# startupblink_url = "https://www.startupblink.com/top-startups/san-francisco-ca-us"

# response = requests.get(startupblink_url, headers={
#     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
# })

# startupblink_companies = []

# if response.status_code == 200:
#     soup = BeautifulSoup(response.text, 'html.parser')
    
#     # Find and extract startup information
#     # (Selectors would need to be adjusted based on actual HTML structure)
#     startup_elements = soup.select('.startup-card')
    
#     for element in startup_elements:
#         company = {}
        
#         # Extract available information
#         name_elem = element.select_one('.company-name')
#         if name_elem:
#             company['company_name'] = name_elem.text.strip()
        
#         # Extract other fields...
        
#         startupblink_companies.append(company)

# # Convert to DataFrame and save
# sb_df = pd.DataFrame(startupblink_companies)
# sb_df['source'] = 'StartupBlink'
# sb_df.to_csv('startupblink_companies.csv', index=False)

# # Merge with master dataset
# master_df = pd.read_csv('bay_area_startups_master.csv')
# sb_df = pd.read_csv('startupblink_companies.csv')

# final_df = pd.concat([master_df, sb_df])
# final_df = final_df.drop_duplicates(subset=['company_name', 'website'])

# final_df.to_csv('bay_area_startups_final.csv', index=False)


import pandas as pd
import requests
import os
import json
import time
from datetime import datetime, timedelta
import re
import logging
from bs4 import BeautifulSoup
import concurrent.futures

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bay_area_startup_collection.log"),
        logging.StreamHandler()
    ]
)

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Define Bay Area counties and major cities for filtering
BAY_AREA_COUNTIES = [
    'San Francisco', 'San Mateo', 'Santa Clara', 'Alameda', 
    'Contra Costa', 'Marin', 'Napa', 'Sonoma', 'Solano'
]

BAY_AREA_CITIES = [
    'San Francisco', 'Oakland', 'San Jose', 'Palo Alto', 'Mountain View',
    'Sunnyvale', 'Santa Clara', 'Redwood City', 'Menlo Park', 'Berkeley',
    'Fremont', 'Walnut Creek', 'San Mateo', 'South San Francisco', 'Emeryville',
    'Cupertino', 'Campbell', 'Los Gatos', 'Milpitas', 'Hayward',
    'San Rafael', 'San Bruno', 'Burlingame', 'Foster City', 'Sausalito',
    'Alameda', 'Richmond', 'San Leandro', 'Pleasanton', 'Livermore',
    'Dublin', 'Newark', 'Union City', 'Novato', 'Larkspur'
]

# Technology and startup keywords for filtering
TECH_KEYWORDS = [
    'tech', 'software', 'app', 'digital', 'ai', 'artificial intelligence',
    'machine learning', 'blockchain', 'crypto', 'data', 'platform', 'saas',
    'cloud', 'internet', 'web', 'mobile', 'robotics', 'automation', 'biotech',
    'fintech', 'healthtech', 'edtech', 'cyber', 'security', 'analytics',
    'ecommerce', 'startup', 'ventures', 'innovation', 'labs', 'technologies',
    'solutions', 'systems', 'networks', 'computing', 'semiconductor', 'hardware',
    'api', 'database', 'quantum', 'virtual reality', 'augmented reality', 'ar', 'vr'
]

# Standard columns for the final dataset
STANDARD_COLUMNS = [
    'company_name', 'description', 'website', 'location', 'city', 'county',
    'address', 'industry', 'industry_tags', 'founding_year', 'employee_count',
    'status', 'is_tech', 'naics_code', 'data_source', 'source_url'
]

def clean_company_name(name):
    """Clean and standardize company names."""
    if not name or pd.isna(name):
        return None
    
    # Convert to string if not already
    name = str(name)
    
    # Remove common business suffixes
    suffixes = [' inc', ' llc', ' corporation', ' corp', ' incorporated', ' ltd', ' limited']
    cleaned_name = name.lower()
    for suffix in suffixes:
        if cleaned_name.endswith(suffix):
            cleaned_name = cleaned_name[:-len(suffix)]
    
    # Remove special characters and extra whitespace
    cleaned_name = re.sub(r'[^\w\s]', ' ', cleaned_name)
    cleaned_name = re.sub(r'\s+', ' ', cleaned_name).strip()
    
    # Title case for final format
    return cleaned_name.title()

def is_likely_startup(row, tech_keywords=TECH_KEYWORDS):
    """Determine if a business is likely a startup based on various signals."""
    # Check if any description, industry, or name fields match tech keywords
    text_fields = []
    
    for field in ['description', 'industry', 'industry_tags', 'company_name']:
        if field in row and pd.notnull(row[field]):
            text_fields.append(str(row[field]).lower())
    
    combined_text = ' '.join(text_fields)
    
    # Initialize startup likelihood score
    startup_score = 0
    
    # Check for tech keywords (1 point for each match, up to 3)
    matched_keywords = [keyword for keyword in tech_keywords if keyword in combined_text]
    startup_score += min(len(matched_keywords), 3)
    
    # Check founding year (if available)
    if 'founding_year' in row and pd.notnull(row['founding_year']):
        try:
            founding_year = int(row['founding_year'])
            current_year = datetime.now().year
            years_old = current_year - founding_year
            
            # Startups are typically younger companies
            if years_old <= 3:  # Very recent = very likely startup
                startup_score += 3
            elif years_old <= 7:  # Recent = likely startup
                startup_score += 2
            elif years_old <= 15:  # Moderately recent = possibly startup
                startup_score += 1
        except:
            pass
    
    # Check explicit startup keywords
    startup_keywords = ['startup', 'start-up', 'start up', 'venture', 'seed', 'founded', 'founder']
    if any(keyword in combined_text for keyword in startup_keywords):
        startup_score += 2
    
    # Check NAICS codes for tech industries
    tech_naics_prefixes = ['518', '519', '5415', '5417', '3341', '3342', '3344', '3345']
    if 'naics_code' in row and pd.notnull(row['naics_code']):
        naics = str(row['naics_code'])
        if any(naics.startswith(prefix) for prefix in tech_naics_prefixes):
            startup_score += 1
    
    # Companies from VC portfolios are almost certainly startups
    if 'data_source' in row and pd.notnull(row['data_source']):
        source = str(row['data_source']).lower()
        if 'portfolio' in source or 'venture' in source or 'vc' in source:
            startup_score += 3
    
    # Threshold for considering a company a startup
    return startup_score >= 2

def in_bay_area(location):
    """Check if a location is in the Bay Area."""
    if not location or pd.isna(location):
        return False
    
    location = str(location).lower()
    
    # Check for Bay Area counties
    for county in BAY_AREA_COUNTIES:
        if county.lower() in location:
            return True
    
    # Check for major Bay Area cities
    for city in BAY_AREA_CITIES:
        if city.lower() in location:
            return True
    
    # Check for common Bay Area terms
    if any(term in location for term in ['bay area', 'silicon valley', 'sf', 'sfo']):
        return True
    
    return False

def extract_city_from_location(location):
    """Extract city name from location string."""
    if not location or pd.isna(location):
        return None
    
    location = str(location).lower()
    
    # Try to match with known Bay Area cities
    for city in BAY_AREA_CITIES:
        city_pattern = r'\b' + city.lower() + r'\b'
        if re.search(city_pattern, location):
            return city
    
    # For San Francisco, also check for 'SF'
    if 'san francisco' in location or re.search(r'\bsf\b', location):
        return 'San Francisco'
    
    return None

def fetch_github_bay_area_companies():
    """Fetch Bay Area companies from GitHub repository."""
    logging.info("Fetching Bay Area companies from GitHub repositories...")
    
    github_datasets = [
        {
            'url': "https://raw.githubusercontent.com/nihalrai/tech-companies-bay-area/main/Bay-Area-Companies-List.csv",
            'name': "tech-companies-bay-area"
        },
        {
            'url': "https://raw.githubusercontent.com/connor11528/tech-companies-and-startups/main/silicon-valley-companies.csv",
            'name': "silicon-valley-companies"
        }
    ]
    
    all_companies = []
    
    for dataset in github_datasets:
        try:
            logging.info(f"Fetching from {dataset['name']}...")
            df = pd.read_csv(dataset['url'])
            
            # Save raw data
            raw_file = f"data/{dataset['name']}_raw.csv"
            df.to_csv(raw_file, index=False)
            logging.info(f"Saved raw data to {raw_file}")
            
            # Standardize column names based on actual columns in the file
            if 'Company' in df.columns:
                df = df.rename(columns={'Company': 'company_name'})
            if 'Description' in df.columns:
                df = df.rename(columns={'Description': 'description'})
            if 'Website' in df.columns:
                df = df.rename(columns={'Website': 'website'})
            if 'Location' in df.columns:
                df = df.rename(columns={'Location': 'location'})
            if 'Tags' in df.columns:
                df = df.rename(columns={'Tags': 'industry_tags'})
            
            # Add source information
            df['data_source'] = dataset['name']
            df['source_url'] = dataset['url']
            
            # Extract city from location if possible
            if 'location' in df.columns:
                df['city'] = df['location'].apply(extract_city_from_location)
            
            # Add standardized columns if missing
            for col in STANDARD_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            
            # Append to all companies
            all_companies.append(df)
            logging.info(f"Added {len(df)} companies from {dataset['name']}")
            
        except Exception as e:
            logging.error(f"Error fetching from {dataset['name']}: {str(e)}")
    
    if all_companies:
        # Combine all datasets
        combined_df = pd.concat(all_companies, ignore_index=True)
        
        # Clean company names
        combined_df['company_name'] = combined_df['company_name'].apply(clean_company_name)
        
        # Determine if likely a startup
        combined_df['is_tech'] = combined_df.apply(is_likely_startup, axis=1)
        
        # Save to file
        output_file = "data/github_bay_area_companies.csv"
        combined_df.to_csv(output_file, index=False)
        logging.info(f"Saved {len(combined_df)} companies to {output_file}")
        
        return combined_df
    else:
        logging.warning("No data retrieved from GitHub repositories")
        return pd.DataFrame(columns=STANDARD_COLUMNS)

def fetch_datasf_businesses():
    """Fetch business data from DataSF API."""
    logging.info("Fetching business data from DataSF...")
    
    # Registered Business Locations dataset
    url = "https://data.sfgov.org/resource/g8m3-pdis.json"
    
    # Define parameters for recent businesses (last 10 years)
    ten_years_ago = (datetime.now() - timedelta(days=10*365)).strftime("%Y-%m-%d")
    
    params = {
        "$limit": 50000,  # Maximum number of records to retrieve
        "$where": f"business_start_date >= '{ten_years_ago}'",
        "$order": "business_start_date DESC"
    }
    
    try:
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            
            # Save raw data
            raw_file = "data/datasf_businesses_raw.csv"
            df.to_csv(raw_file, index=False)
            logging.info(f"Saved raw data to {raw_file} with {len(df)} records")
            
            # Map column names to standard format
            column_mapping = {
                'business_name': 'company_name',
                'business_dba': 'dba_name',  # Doing Business As name
                'naics': 'naics_code',
                'naics_description': 'industry',
                'business_start_date': 'founding_year',
                'business_location': 'address',
                'neighborhoods_analysis_boundaries': 'neighborhood'
            }
            
            # Rename columns that exist in the DataFrame
            rename_cols = {old: new for old, new in column_mapping.items() if old in df.columns}
            df = df.rename(columns=rename_cols)
            
            # Extract founding year from date if available
            if 'founding_year' in df.columns:
                df['founding_year'] = pd.to_datetime(df['founding_year'], errors='coerce').dt.year
            
            # Set location to San Francisco
            df['location'] = 'San Francisco, CA'
            df['city'] = 'San Francisco'
            df['county'] = 'San Francisco'
            
            # Add source information
            df['data_source'] = 'DataSF Registered Businesses'
            df['source_url'] = url
            
            # Identify likely tech startups
            df['is_tech'] = df.apply(is_likely_startup, axis=1)
            
            # Add missing standard columns
            for col in STANDARD_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            
            # Clean company names
            df['company_name'] = df['company_name'].apply(clean_company_name)
            
            # Filter to keep only tech/startup companies
            tech_businesses = df[df['is_tech'] == True].copy()
            
            # Save tech businesses to file
            output_file = "data/datasf_tech_businesses.csv"
            tech_businesses.to_csv(output_file, index=False)
            logging.info(f"Saved {len(tech_businesses)} tech businesses to {output_file}")
            
            return tech_businesses
            
        else:
            logging.error(f"Error accessing DataSF API: {response.status_code}")
            logging.error(response.text)
            return pd.DataFrame(columns=STANDARD_COLUMNS)
            
    except Exception as e:
        logging.error(f"Error fetching DataSF businesses: {str(e)}")
        return pd.DataFrame(columns=STANDARD_COLUMNS)

def fetch_datasf_building_permits():
    """Fetch building permit data from DataSF."""
    logging.info("Fetching building permit data from DataSF...")
    
    url = "https://data.sfgov.org/resource/i98e-djp9.json"
    
    # Focus on commercial permits in recent years
    three_years_ago = (datetime.now() - timedelta(days=3*365)).strftime("%Y-%m-%d")
    
    params = {
        "$limit": 10000,
        "$where": f"permit_creation_date >= '{three_years_ago}' AND permit_type_definition like '%COMMERCIAL%'",
        "$order": "permit_creation_date DESC"
    }
    
    try:
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            
            # Save raw data
            raw_file = "data/datasf_building_permits_raw.csv"
            df.to_csv(raw_file, index=False)
            logging.info(f"Saved raw data to {raw_file} with {len(df)} records")
            
            # Extract companies from description and create entries
            companies = []
            
            if 'description' in df.columns:
                for idx, row in df.iterrows():
                    if pd.isna(row['description']):
                        continue
                    
                    description = str(row['description']).upper()
                    
                    # Look for company indicators in description (FOR TENANT X, NEW OFFICE FOR X)
                    company_patterns = [
                        r'FOR\s+TENANT\s+([A-Z0-9\s&]+)',
                        r'NEW\s+OFFICE\s+FOR\s+([A-Z0-9\s&]+)',
                        r'TENANT\s+IMPROVEMENT\s+FOR\s+([A-Z0-9\s&]+)',
                        r'INTERIOR\s+RENOVATION\s+FOR\s+([A-Z0-9\s&]+)',
                        r'BUILDOUT\s+FOR\s+([A-Z0-9\s&]+)'
                    ]
                    
                    for pattern in company_patterns:
                        match = re.search(pattern, description)
                        if match:
                            company_name = match.group(1).strip()
                            
                            # Skip generic names
                            generic_terms = ['TENANT', 'CLIENT', 'BUSINESS', 'OFFICE', 'N/A', 'TBD']
                            if any(term == company_name for term in generic_terms):
                                continue
                                
                            companies.append({
                                'company_name': company_name,
                                'address': row.get('address', None),
                                'description': row.get('description', None),
                                'location': 'San Francisco, CA',
                                'city': 'San Francisco',
                                'county': 'San Francisco',
                                'data_source': 'DataSF Building Permits',
                                'source_url': url
                            })
            
            if companies:
                companies_df = pd.DataFrame(companies)
                
                # Clean company names
                companies_df['company_name'] = companies_df['company_name'].apply(clean_company_name)
                
                # Add missing standard columns
                for col in STANDARD_COLUMNS:
                    if col not in companies_df.columns:
                        companies_df[col] = None
                
                # Remove duplicates
                companies_df = companies_df.drop_duplicates(subset=['company_name'])
                
                # Flag tech companies
                companies_df['is_tech'] = companies_df.apply(is_likely_startup, axis=1)
                
                # Filter to tech companies
                tech_companies = companies_df[companies_df['is_tech'] == True].copy()
                
                # Save to file
                output_file = "data/datasf_permit_companies.csv"
                tech_companies.to_csv(output_file, index=False)
                logging.info(f"Extracted {len(tech_companies)} potential tech companies from building permits")
                
                return tech_companies
            else:
                logging.info("No companies extracted from building permits")
                return pd.DataFrame(columns=STANDARD_COLUMNS)
            
        else:
            logging.error(f"Error accessing DataSF Building Permits API: {response.status_code}")
            return pd.DataFrame(columns=STANDARD_COLUMNS)
            
    except Exception as e:
        logging.error(f"Error fetching DataSF building permits: {str(e)}")
        return pd.DataFrame(columns=STANDARD_COLUMNS)

def fetch_bay_area_economic_data():
    """Fetch business data from Bay Area county economic development sources."""
    logging.info("Fetching data from Bay Area economic development sources...")
    
    # This is a simplified version. A full implementation would need to access
    # multiple county/city business registries across the Bay Area
    
    # Example for San Mateo County 
    url = "https://data.smcgov.org/resource/wv2b-n3bd.json"
    
    params = {
        "$limit": 25000,
        "$where": "license_status='Active'"
    }
    
    try:
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            
            # Save raw data
            raw_file = "data/smc_businesses_raw.csv"
            df.to_csv(raw_file, index=False)
            logging.info(f"Saved raw data to {raw_file} with {len(df)} records")
            
            # Map columns to standard format
            column_mapping = {
                'business_name': 'company_name',
                'naics_description': 'industry',
                'naics_code': 'naics_code',
                'business_address': 'address',
                'license_issue_date': 'founding_year'
            }
            
            # Rename columns that exist in the DataFrame
            rename_cols = {old: new for old, new in column_mapping.items() if old in df.columns}
            df = df.rename(columns=rename_cols)
            
            # Extract founding year from date if available
            if 'founding_year' in df.columns:
                df['founding_year'] = pd.to_datetime(df['founding_year'], errors='coerce').dt.year
            
            # Set location to San Mateo County
            df['location'] = 'San Mateo County, CA'
            df['county'] = 'San Mateo'
            
            # Extract city if available
            if 'city' in df.columns:
                pass  # Already has city column
            elif 'address' in df.columns:
                # Try to extract city from address
                def extract_city(address):
                    if pd.isna(address):
                        return None
                    for city in BAY_AREA_CITIES:
                        if city.upper() in address.upper():
                            return city
                    return None
                df['city'] = df['address'].apply(extract_city)
            
            # Add source information
            df['data_source'] = 'San Mateo County Business Licenses'
            df['source_url'] = url
            
            # Add missing standard columns
            for col in STANDARD_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            
            # Clean company names
            df['company_name'] = df['company_name'].apply(clean_company_name)
            
            # Identify likely tech startups
            df['is_tech'] = df.apply(is_likely_startup, axis=1)
            
            # Filter to keep only tech/startup companies
            tech_businesses = df[df['is_tech'] == True].copy()
            
            # Save tech businesses to file
            output_file = "data/smc_tech_businesses.csv"
            tech_businesses.to_csv(output_file, index=False)
            logging.info(f"Saved {len(tech_businesses)} San Mateo County tech businesses to {output_file}")
            
            return tech_businesses
            
        else:
            logging.error(f"Error accessing San Mateo County API: {response.status_code}")
            return pd.DataFrame(columns=STANDARD_COLUMNS)
            
    except Exception as e:
        logging.error(f"Error fetching San Mateo County businesses: {str(e)}")
        return pd.DataFrame(columns=STANDARD_COLUMNS)

def scrape_bay_area_vc_portfolios():
    """Scrape portfolio companies from Bay Area VC websites."""
    logging.info("Scraping Bay Area VC portfolios...")
    
    # List of major Bay Area VC firms with portfolio pages
    vc_firms = [
        {"name": "Andreessen Horowitz", "url": "https://a16z.com/portfolio/"},
        {"name": "Sequoia Capital", "url": "https://www.sequoiacap.com/companies/"},
        {"name": "Greylock", "url": "https://greylock.com/companies/"},
        {"name": "Khosla Ventures", "url": "https://www.khoslaventures.com/portfolio"},
        {"name": "Accel", "url": "https://www.accel.com/companies"}
    ]
    
    all_portfolio_companies = []
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for vc in vc_firms:
        logging.info(f"Checking portfolio for {vc['name']}...")
        try:
            response = requests.get(vc['url'], headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # This would need to be customized for each VC website's structure
                # Here's a generic approach that might work for some sites
                
                # Look for company cards or portfolio entries
                company_elements = soup.select('.portfolio-company, .company-card, .portfolio-item')
                
                for element in company_elements:
                    company = {
                        'data_source': f"{vc['name']} Portfolio",
                        'source_url': vc['url']
                    }
                    
                    # Try to find company name - adjust selectors based on actual HTML
                    name_elem = element.select_one('h3, h4, .company-name, .name')
                    if name_elem:
                        company['company_name'] = name_elem.text.strip()
                    
                    # Try to find description
                    desc_elem = element.select_one('p, .description, .company-description')
                    if desc_elem:
                        company['description'] = desc_elem.text.strip()
                    
                    # Try to find website link
                    link_elem = element.select_one('a[href^="http"]')
                    if link_elem and 'href' in link_elem.attrs:
                        company['website'] = link_elem['href']
                    
                    # Only add if we found a company name
                    if 'company_name' in company and company['company_name']:
                        all_portfolio_companies.append(company)
                
                logging.info(f"Found {len(company_elements)} potential companies from {vc['name']}")
                
            else:
                logging.warning(f"Could not access {vc['name']} portfolio page: {response.status_code}")
                
        except Exception as e:
            logging.error(f"Error scraping {vc['name']} portfolio: {str(e)}")
            
        # Add a delay between requests to avoid overloading servers
        time.sleep(2)
    
    if all_portfolio_companies:
        portfolio_df = pd.DataFrame(all_portfolio_companies)
        
        # Clean company names
        portfolio_df['company_name'] = portfolio_df['company_name'].apply(clean_company_name)
        
        # Assume all VC-backed companies are startups
        portfolio_df['is_tech'] = True
        
        # Add missing standard columns
        for col in STANDARD_COLUMNS:
            if col not in portfolio_df.columns:
                portfolio_df[col] = None
        
        # Remove duplicates
        portfolio_df = portfolio_df.drop_duplicates(subset=['company_name'])
        
        # Save to file
        output_file = "data/vc_portfolio_companies.csv"
        portfolio_df.to_csv(output_file, index=False)
        logging.info(f"Saved {len(portfolio_df)} VC portfolio companies to {output_file}")
        
        return portfolio_df
    else:
        logging.warning("No VC portfolio companies found")
        return pd.DataFrame(columns=STANDARD_COLUMNS)

def fetch_startup_news_mentions():
    """Scrape news sources for Bay Area startup mentions."""
    logging.info("Fetching startup mentions from news sources...")
    
    # This would typically use a news API, but for demonstration,
    # we'll create a simplified version using TechCrunch's Bay Area tag
    
    url = "https://techcrunch.com/tag/san-francisco/"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find article elements
            articles = soup.select('article')
            
            news_companies = []
            
            for article in articles:
                # Extract title
                title_elem = article.select_one('h2, .post-title')
                if not title_elem:
                    continue
                
                title = title_elem.text.strip()
                
                # Extract potential company names from title
                # This is a simplistic approach and would need refinement
                # to extract actual company names more accurately
                words = title.split()
                potential_companies = []
                
                for i in range(len(words) - 1):
                    if words[i][0].isupper() and words[i+1][0].isupper():
                        potential_company = f"{words[i]} {words[i+1]}"
                        if len(potential_company) > 5:  # Avoid very short names
                            potential_companies.append(potential_company)
                
                # Extract article URL
                link_elem = article.select_one('a')
                article_url = link_elem['href'] if link_elem and 'href' in link_elem.attrs else None
                
                for company_name in potential_companies:
                    news_companies.append({
                        'company_name': company_name,
                        'description': title,
                        'source_url': article_url,
                        'data_source': 'TechCrunch News Mentions',
                        'location': 'San Francisco Bay Area',
                        'city': 'San Francisco'  # Default, would need verification
                    })
            
            if news_companies:
                news_df = pd.DataFrame(news_companies)
                
                # Clean company names
                news_df['company_name'] = news_df['company_name'].apply(clean_company_name)
                
                # Assume companies mentioned in tech news are likely tech companies
                news_df['is_tech'] = True
                
                # Add missing standard columns
                for col in STANDARD_COLUMNS:
                    if col not in news_df.columns:
                        news_df[col] = None
                
                # Remove duplicates
                news_df = news_df.drop_duplicates(subset=['company_name'])
                
                # Save to file
                output_file = "data/news_mentioned_companies.csv"
                news_df.to_csv(output_file, index=False)
                logging.info(f"Saved {len(news_df)} news-mentioned companies to {output_file}")
                
                return news_df
            else:
                logging.info("No companies extracted from news mentions")
                return pd.DataFrame(columns=STANDARD_COLUMNS)
            
        else:
            logging.error(f"Error accessing news source: {response.status_code}")
            return pd.DataFrame(columns=STANDARD_COLUMNS)
            
    except Exception as e:
        logging.error(f"Error fetching startup news mentions: {str(e)}")
        return pd.DataFrame(columns=STANDARD_COLUMNS)

def merge_all_sources():
    """Merge all data sources into a single dataset."""
    logging.info("Merging all data sources...")
    
    data_files = [
        "data/github_bay_area_companies.csv",
        "data/datasf_tech_businesses.csv",
        "data/datasf_permit_companies.csv",
        "data/smc_tech_businesses.csv",
        "data/vc_portfolio_companies.csv",
        "data/news_mentioned_companies.csv"
    ]
    
    all_dataframes = []
    
    for file in data_files:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file)
                logging.info(f"Loaded {len(df)} records from {file}")
                all_dataframes.append(df)
            except Exception as e:
                logging.error(f"Error loading {file}: {str(e)}")
        else:
            logging.warning(f"File not found: {file}")
    
    if all_dataframes:
        # Combine all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        logging.info(f"Combined dataset has {len(combined_df)} records before deduplication")
        
        # Clean company names for better matching
        combined_df['company_name_clean'] = combined_df['company_name'].apply(clean_company_name)
        
        # Remove duplicates based on cleaned company name
        deduplicated_df = combined_df.drop_duplicates(subset=['company_name_clean'])
        logging.info(f"After deduplication: {len(deduplicated_df)} unique companies")
        
        # Check company location is in Bay Area
        if 'location' in deduplicated_df.columns:
            bay_area_df = deduplicated_df[
                deduplicated_df['location'].apply(lambda x: in_bay_area(x) if pd.notnull(x) else False)
            ].copy()
            logging.info(f"Filtered to {len(bay_area_df)} Bay Area companies")
        else:
            bay_area_df = deduplicated_df.copy()
        
        # Remove temporary columns
        if 'company_name_clean' in bay_area_df.columns:
            bay_area_df = bay_area_df.drop(columns=['company_name_clean'])
        
        # Ensure only standard columns are in the final output
        final_df = bay_area_df[
            [col for col in STANDARD_COLUMNS if col in bay_area_df.columns]
        ]
        
        # Save to CSV
        output_file = "bay_area_startups_master.csv"
        final_df.to_csv(output_file, index=False)
        logging.info(f"Saved final dataset with {len(final_df)} Bay Area startups to {output_file}")
        
        return final_df
    else:
        logging.error("No data files found to merge")
        return None
def main():
    """Main function to run the entire data collection pipeline."""
    logging.info("Starting Bay Area startup data collection...")
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Use concurrent.futures to parallelize some of the data collection
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Start concurrent tasks
        github_future = executor.submit(fetch_github_bay_area_companies)
        datasf_future = executor.submit(fetch_datasf_businesses)
        permits_future = executor.submit(fetch_datasf_building_permits)
        
        # Wait for these tasks to complete
        github_result = github_future.result()
        datasf_result = datasf_future.result()
        permits_result = permits_future.result()
        
        logging.info(f"Completed initial data collection:")
        logging.info(f"  - GitHub companies: {len(github_result) if isinstance(github_result, pd.DataFrame) else 0}")
        logging.info(f"  - DataSF businesses: {len(datasf_result) if isinstance(datasf_result, pd.DataFrame) else 0}")
        logging.info(f"  - Building permits: {len(permits_result) if isinstance(permits_result, pd.DataFrame) else 0}")
    
    # Run the remaining data collection tasks sequentially
    # These may involve more complex web scraping that benefits from sequential execution
    logging.info("Starting secondary data collection...")
    
    economic_data = fetch_bay_area_economic_data()
    logging.info(f"Collected {len(economic_data) if isinstance(economic_data, pd.DataFrame) else 0} records from economic data")
    
    vc_portfolio_data = scrape_bay_area_vc_portfolios()
    logging.info(f"Collected {len(vc_portfolio_data) if isinstance(vc_portfolio_data, pd.DataFrame) else 0} records from VC portfolios")
    
    news_data = fetch_startup_news_mentions()
    logging.info(f"Collected {len(news_data) if isinstance(news_data, pd.DataFrame) else 0} records from news mentions")
    
    # Merge all the collected data
    final_dataset = merge_all_sources()
    
    if final_dataset is not None and len(final_dataset) > 0:
        logging.info(f"Successfully collected data on {len(final_dataset)} Bay Area startups")
        logging.info(f"Final dataset saved to 'bay_area_startups_master.csv'")
        
        # Generate basic statistics
        if 'founding_year' in final_dataset.columns:
            year_counts = final_dataset['founding_year'].value_counts().sort_index()
            logging.info(f"Startups by founding year: {year_counts.to_dict()}")
        
        if 'city' in final_dataset.columns:
            city_counts = final_dataset['city'].value_counts().head(10)
            logging.info(f"Top startup cities: {city_counts.to_dict()}")
        
        if 'industry' in final_dataset.columns:
            industry_counts = final_dataset['industry'].value_counts().head(10)
            logging.info(f"Top industries: {industry_counts.to_dict()}")
            
        return True
    else:
        logging.error("Failed to create final dataset")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logging.info("Data collection completed successfully")
    else:
        logging.error("Data collection failed")
