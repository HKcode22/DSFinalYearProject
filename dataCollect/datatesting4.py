# import os
# import pandas as pd
# import requests
# import json
# import time
# import logging
# import ssl
# import certifi
# import urllib.request
# import concurrent.futures
# from datetime import datetime, timedelta
# from bs4 import BeautifulSoup

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )

# # Create data directory
# os.makedirs('data', exist_ok=True)

# def fix_ssl_issue():
#     """Fix SSL certificate issues for macOS."""
#     # This is specific to macOS - you can comment this out if not using macOS
#     try:
#         import subprocess
#         # Run the Python certificate install script
#         subprocess.run([
#             '/Applications/Python 3.12/Install Certificates.command'
#         ], check=True, capture_output=True)
#         logging.info("Installed SSL certificates for macOS")
#     except Exception as e:
#         logging.warning(f"Could not run certificate installation: {e}")
#         # Alternative fix - create a custom SSL context
#         ssl_context = ssl.create_default_context()
#         ssl_context.check_hostname = False
#         ssl_context.verify_mode = ssl.CERT_NONE
#         # Patch urllib.request to use our context
#         urllib.request.urlopen = lambda url, *args, **kwargs: \
#             urllib.request.__original_urlopen(url, *args, context=ssl_context, **kwargs)
#         logging.info("Applied SSL verification workaround")

# def fetch_github_bay_area_companies():
#     """Fetch Bay Area companies from GitHub repositories with SSL fix."""
#     logging.info("Fetching Bay Area companies from GitHub repositories...")
    
#     # URLs for GitHub repositories with Bay Area company data
#     urls = {
#         'tech-companies-bay-area': 'https://raw.githubusercontent.com/nihalrai/tech-companies-bay-area/main/Bay-Area-Companies-List.csv',
#         'silicon-valley-companies': 'https://raw.githubusercontent.com/connor11528/tech-companies-and-startups/master/silicon-valley-companies.csv'
#     }
    
#     combined_df = pd.DataFrame()
    
#     for name, url in urls.items():
#         logging.info(f"Fetching from {name}...")
#         try:
#             # Try with requests first
#             response = requests.get(url, verify=True)
#             if response.status_code == 200:
#                 df = pd.read_csv(pd.StringIO(response.text))
#                 logging.info(f"Retrieved {len(df)} companies from {name}")
#                 combined_df = pd.concat([combined_df, df], ignore_index=True)
#             else:
#                 # If requests fails, try with urllib, disabling SSL verification if needed
#                 try:
#                     context = ssl._create_unverified_context()
#                     with urllib.request.urlopen(url, context=context) as response:
#                         content = response.read().decode('utf-8')
#                     df = pd.read_csv(pd.StringIO(content))
#                     logging.info(f"Retrieved {len(df)} companies from {name} (fallback method)")
#                     combined_df = pd.concat([combined_df, df], ignore_index=True)
#                 except Exception as inner_e:
#                     logging.error(f"Fallback method also failed for {name}: {str(inner_e)}")
#         except Exception as e:
#             logging.error(f"Error fetching from {name}: {str(e)}")
    
#     if len(combined_df) > 0:
#         # Standardize column names
#         column_mapping = {
#             'Company': 'company_name',
#             'Description': 'description', 
#             'Website': 'website',
#             'company': 'company_name',
#             'url': 'website',
#             'description': 'description'
#         }
        
#         # Rename columns that exist in the DataFrame
#         for old_col, new_col in column_mapping.items():
#             if old_col in combined_df.columns:
#                 combined_df[new_col] = combined_df[old_col]
        
#         # Ensure required columns exist
#         required_cols = ['company_name', 'description', 'website', 'industry', 'location']
#         for col in required_cols:
#             if col not in combined_df.columns:
#                 combined_df[col] = None
        
#         # Filter for Bay Area mentions
#         bay_area_keywords = ['san francisco', 'sf', 'bay area', 'silicon valley', 'palo alto', 
#                             'san jose', 'menlo park', 'mountain view', 'santa clara', 'cupertino',
#                             'oakland', 'berkeley', 'san mateo', 'redwood city', 'south san francisco']
        
#         # Check location column if it exists and contains Bay Area references
#         if 'location' in combined_df.columns and not combined_df['location'].isna().all():
#             is_bay_area = combined_df['location'].astype(str).str.lower().apply(
#                 lambda x: any(keyword in x for keyword in bay_area_keywords)
#             )
#             combined_df = combined_df[is_bay_area]
        
#         # Add data source
#         combined_df['data_source'] = 'GitHub Repositories'
        
#         # Save to CSV
#         output_path = 'data/github_bay_area_companies.csv'
#         combined_df.to_csv(output_path, index=False)
#         logging.info(f"Saved {len(combined_df)} companies to {output_path}")
#         return combined_df
#     else:
#         logging.warning("No data retrieved from GitHub repositories")
#         # Create a minimal dataset with hard-coded Bay Area tech companies
#         minimal_df = pd.DataFrame({
#             'company_name': ['Airbnb', 'Stripe', 'Lyft', 'Uber', 'DoorDash', 'Coinbase', 'Discord', 'Figma', 'Slack', 'Square'],
#             'industry': ['Travel/Hospitality', 'Fintech', 'Transportation', 'Transportation', 'Food Delivery', 'Cryptocurrency', 'Communications', 'Design Software', 'Business Communications', 'Fintech'],
#             'location': ['San Francisco', 'San Francisco', 'San Francisco', 'San Francisco', 'San Francisco', 'San Francisco', 'San Francisco', 'San Francisco', 'San Francisco', 'San Francisco'],
#             'data_source': ['Fallback Data'] * 10
#         })
#         output_path = 'data/github_bay_area_companies.csv'
#         minimal_df.to_csv(output_path, index=False)
#         logging.info(f"Saved {len(minimal_df)} fallback companies to {output_path}")
#         return minimal_df

# def fetch_datasf_businesses():
#     """Fetch business data from DataSF with corrected column names."""
#     logging.info("Fetching business data from DataSF...")
    
#     # The DataSF Registered Business Locations dataset
#     url = "https://data.sfgov.org/resource/g8m3-pdis.json"
    
#     # Corrected query - use dba_start_date instead of business_start_date
#     # 10 years ago date
#     start_date = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')
    
#     params = {
#         "$limit": 50000,
#         # Filter for recently registered businesses using the correct column name
#         "$where": f"dba_start_date >= '{start_date}'",
#         "$order": "dba_start_date DESC",
#         # San Francisco tech-related businesses
#         "$q": "tech software app digital data platform"
#     }
    
#     try:
#         response = requests.get(url, params=params)
        
#         if response.status_code == 200:
#             data = response.json()
#             df = pd.DataFrame(data)
            
#             # Save raw data
#             raw_output_path = 'data/datasf_businesses_raw.csv'
#             df.to_csv(raw_output_path, index=False)
#             logging.info(f"Saved raw data to {raw_output_path} with {len(df)} records")
            
#             # Extract tech businesses
#             tech_keywords = ['tech', 'software', 'app', 'digital', 'ai', 'artificial intelligence',
#                             'machine learning', 'data', 'platform', 'saas', 'cloud', 'internet', 
#                             'web', 'mobile', 'robotics', 'automation', 'cyber', 'crypto']
            
#             # Extract tech businesses based on business name or NAICS descriptions
#             # Check each column that might contain relevant information
#             is_tech = pd.Series(False, index=df.index)
            
#             # Check business name (dba_name)
#             if 'dba_name' in df.columns:
#                 is_tech = is_tech | df['dba_name'].astype(str).str.lower().apply(
#                     lambda x: any(keyword in x for keyword in tech_keywords)
#                 )
            
#             # Check NAICS code descriptions
#             if 'naics_code_descriptions_list' in df.columns:
#                 is_tech = is_tech | df['naics_code_descriptions_list'].astype(str).str.lower().apply(
#                     lambda x: any(keyword in x for keyword in tech_keywords)
#                 )
            
#             tech_businesses = df[is_tech].copy()
            
#             # Standardize column names and add source
#             tech_businesses['company_name'] = tech_businesses['dba_name'] if 'dba_name' in tech_businesses.columns else ''
#             tech_businesses['location'] = 'San Francisco'
#             tech_businesses['data_source'] = 'DataSF Registered Businesses'
            
#             # Add founding year if possible
#             if 'dba_start_date' in tech_businesses.columns:
#                 tech_businesses['founding_year'] = pd.to_datetime(tech_businesses['dba_start_date'], errors='coerce').dt.year
            
#             # Save tech businesses
#             output_path = 'data/datasf_tech_businesses.csv'
#             tech_businesses.to_csv(output_path, index=False)
#             logging.info(f"Extracted {len(tech_businesses)} tech businesses and saved to {output_path}")
            
#             return tech_businesses
#         else:
#             # Log the error response
#             logging.error(f"Error accessing DataSF API: {response.status_code}")
#             logging.error(f"{response.text}")
            
#             # Create an empty DataFrame with expected columns
#             empty_df = pd.DataFrame(columns=['company_name', 'location', 'data_source'])
#             empty_df.to_csv('data/datasf_tech_businesses.csv', index=False)
#             return empty_df
#     except Exception as e:
#         logging.error(f"Exception while accessing DataSF API: {str(e)}")
#         # Create an empty DataFrame
#         empty_df = pd.DataFrame(columns=['company_name', 'location', 'data_source'])
#         empty_df.to_csv('data/datasf_tech_businesses.csv', index=False)
#         return empty_df

# def fetch_datasf_building_permits():
#     """Fetch building permit data from DataSF."""
#     logging.info("Fetching building permit data from DataSF...")
    
#     # Building Permits dataset
#     url = "https://data.sfgov.org/resource/i98e-djp9.json"
    
#     # 5 years ago date
#     start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
#     params = {
#         "$limit": 50000,
#         "$where": f"filed_date >= '{start_date}'",
#         # Filter for commercial permits
#         "$q": "commercial office tenant tech startup"
#     }
    
#     try:
#         response = requests.get(url, params=params)
        
#         if response.status_code == 200:
#             data = response.json()
#             df = pd.DataFrame(data)
            
#             # Save raw data
#             raw_output_path = 'data/datasf_building_permits_raw.csv'
#             df.to_csv(raw_output_path, index=False)
#             logging.info(f"Saved raw data to {raw_output_path} with {len(df)} records")
            
#             # Extract company names from permit descriptions
#             companies = []
            
#             if 'description' in df.columns:
#                 for idx, row in df.iterrows():
#                     description = str(row.get('description', '')).lower()
                    
#                     # Check for company indicators in permit descriptions
#                     company_indicators = ['inc.', 'llc', 'corporation', 'technologies', 'labs', ' ai', 'software']
#                     if any(indicator in description for indicator in company_indicators):
#                         # Extract words that might be company names (not perfect but a start)
#                         words = description.split()
#                         for i, word in enumerate(words):
#                             if word in company_indicators and i > 0:
#                                 potential_company = words[i-1]
#                                 if len(potential_company) > 3 and potential_company not in ['the', 'for', 'and', 'tenant']:
#                                     companies.append({
#                                         'company_name': potential_company.title(),
#                                         'full_description': description,
#                                         'location': 'San Francisco',
#                                         'data_source': 'DataSF Building Permits'
#                                     })
            
#             companies_df = pd.DataFrame(companies)
            
#             # Save extracted companies
#             output_path = 'data/datasf_permit_companies.csv'
#             if len(companies_df) > 0:
#                 companies_df.to_csv(output_path, index=False)
#                 logging.info(f"Extracted {len(companies_df)} companies from building permits and saved to {output_path}")
#             else:
#                 # Create empty DataFrame with expected columns
#                 pd.DataFrame(columns=['company_name', 'full_description', 'location', 'data_source']).to_csv(output_path, index=False)
#                 logging.info("No companies extracted from building permits")
            
#             return companies_df
#         else:
#             logging.error(f"Error accessing DataSF permits API: {response.status_code}")
#             # Create empty DataFrame
#             empty_df = pd.DataFrame(columns=['company_name', 'full_description', 'location', 'data_source'])
#             empty_df.to_csv('data/datasf_permit_companies.csv', index=False)
#             return empty_df
#     except Exception as e:
#         logging.error(f"Exception while accessing DataSF permits API: {str(e)}")
#         # Create empty DataFrame
#         empty_df = pd.DataFrame(columns=['company_name', 'full_description', 'location', 'data_source'])
#         empty_df.to_csv('data/datasf_permit_companies.csv', index=False)
#         return empty_df

# def fetch_bay_area_economic_data():
#     """Fetch data from Bay Area economic development sources."""
#     logging.info("Fetching data from Bay Area economic development sources...")
    
#     companies = []
    
#     # Bay Area counties and cities to check
#     locations = [
#         {
#             'name': 'San Mateo County',
#             'url': 'https://www.samceda.org/tech-companies'
#         },
#         {
#             'name': 'Santa Clara County',
#             'url': 'https://siliconvalleycentralchamber.org/membership-directory/'
#         }
#     ]
    
#     for location in locations:
#         try:
#             response = requests.get(location['url'], timeout=10)
            
#             if response.status_code == 200:
#                 soup = BeautifulSoup(response.text, 'html.parser')
                
#                 # Extract company names - this is a generic approach and may need customization
#                 # for each specific website
#                 company_elements = soup.select('.company, .business-name, .member-name')
                
#                 for element in company_elements:
#                     companies.append({
#                         'company_name': element.text.strip(),
#                         'location': location['name'],
#                         'data_source': f"{location['name']} Economic Data"
#                     })
                
#                 logging.info(f"Found {len(company_elements)} companies from {location['name']}")
#             else:
#                 logging.error(f"Error accessing {location['name']} API: {response.status_code}")
#         except Exception as e:
#             logging.error(f"Error processing {location['name']}: {str(e)}")
    
#     companies_df = pd.DataFrame(companies)
    
#     # Save companies
#     output_path = 'data/smc_tech_businesses.csv'
#     if len(companies_df) > 0:
#         companies_df.to_csv(output_path, index=False)
#         logging.info(f"Saved {len(companies_df)} companies from economic data to {output_path}")
#     else:
#         # Add some fallback data for Silicon Valley
#         fallback_df = pd.DataFrame({
#             'company_name': ['Meta', 'Google', 'Apple', 'Netflix', 'Adobe', 'Nvidia'],
#             'location': ['Menlo Park', 'Mountain View', 'Cupertino', 'Los Gatos', 'San Jose', 'Santa Clara'],
#             'data_source': ['Fallback Silicon Valley Data'] * 6
#         })
#         fallback_df.to_csv(output_path, index=False)
#         logging.info(f"Added fallback data for {len(fallback_df)} Silicon Valley companies")
#         return fallback_df
    
#     return companies_df

# def scrape_bay_area_vc_portfolios():
#     """Scrape Bay Area VC portfolios for startup companies."""
#     logging.info("Scraping Bay Area VC portfolios...")
    
#     # List of top VC firms in the Bay Area
#     vc_firms = [
#         {
#             'name': 'Andreessen Horowitz',
#             'url': 'https://a16z.com/portfolio/'
#         },
#         {
#             'name': 'Sequoia Capital',
#             'url': 'https://www.sequoiacap.com/companies/'
#         },
#         {
#             'name': 'Greylock',
#             'url': 'https://greylock.com/companies/'
#         },
#         {
#             'name': 'Khosla Ventures',
#             'url': 'https://www.khoslaventures.com/portfolio'
#         },
#         {
#             'name': 'Accel',
#             'url': 'https://www.accel.com/companies'
#         }
#     ]
    
#     all_companies = []
    
#     for vc in vc_firms:
#         logging.info(f"Checking portfolio for {vc['name']}...")
        
#         try:
#             # Allow some time between requests
#             time.sleep(2)
            
#             # Use a browser-like user agent
#             headers = {
#                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#             }
            
#             response = requests.get(vc['url'], headers=headers, timeout=10)
            
#             if response.status_code == 200:
#                 soup = BeautifulSoup(response.text, 'html.parser')
                
#                 # Different VCs have different HTML structures - this is a generic approach
#                 # Try various CSS selectors that might indicate company names
#                 company_elements = soup.select('.company-name, .portfolio-company, .company-card h3, .company-title')
                
#                 companies = []
#                 for element in company_elements:
#                     companies.append({
#                         'company_name': element.text.strip(),
#                         'investor': vc['name'],
#                         'data_source': 'VC Portfolio'
#                     })
                
#                 # If no companies found, try different approach
#                 if not companies:
#                     # Look for links or divs containing company references
#                     for link in soup.find_all('a'):
#                         text = link.text.strip()
#                         # Skip short texts or navigational elements
#                         if len(text) > 3 and text.lower() not in ['home', 'about', 'portfolio', 'contact', 'next', 'previous']:
#                             companies.append({
#                                 'company_name': text,
#                                 'investor': vc['name'],
#                                 'data_source': 'VC Portfolio (alternative method)'
#                             })
                
#                 logging.info(f"Found {len(companies)} potential companies from {vc['name']}")
#                 all_companies.extend(companies)
#             else:
#                 logging.warning(f"Could not access {vc['name']} portfolio page: {response.status_code}")
                
#                 # If we can't access the real data, add some known portfolio companies
#                 if vc['name'] == 'Andreessen Horowitz':
#                     all_companies.extend([
#                         {'company_name': 'Airbnb', 'investor': 'Andreessen Horowitz', 'data_source': 'Fallback VC Data'},
#                         {'company_name': 'Coinbase', 'investor': 'Andreessen Horowitz', 'data_source': 'Fallback VC Data'},
#                         {'company_name': 'Figma', 'investor': 'Andreessen Horowitz', 'data_source': 'Fallback VC Data'}
#                     ])
#                 elif vc['name'] == 'Sequoia Capital':
#                     all_companies.extend([
#                         {'company_name': 'Stripe', 'investor': 'Sequoia Capital', 'data_source': 'Fallback VC Data'},
#                         {'company_name': 'Zoom', 'investor': 'Sequoia Capital', 'data_source': 'Fallback VC Data'},
#                         {'company_name': 'DoorDash', 'investor': 'Sequoia Capital', 'data_source': 'Fallback VC Data'}
#                     ])
#         except Exception as e:
#             logging.error(f"Error processing {vc['name']} portfolio: {str(e)}")
            
#             # If we hit an error, add some known portfolio companies as fallback
#             if vc['name'] == 'Greylock':
#                 all_companies.extend([
#                     {'company_name': 'Discord', 'investor': 'Greylock', 'data_source': 'Fallback VC Data'},
#                     {'company_name': 'Roblox', 'investor': 'Greylock', 'data_source': 'Fallback VC Data'}
#                 ])
        
#         # Add some delay between requests
#         time.sleep(2)
    
#     if all_companies:
#         companies_df = pd.DataFrame(all_companies)
        
#         # Save to CSV
#         output_path = 'data/vc_portfolio_companies.csv'
#         companies_df.to_csv(output_path, index=False)
#         logging.info(f"Saved {len(companies_df)} VC portfolio companies to {output_path}")
#         return companies_df
#     else:
#         logging.warning("No VC portfolio companies found")
        
#         # Create fallback data
#         fallback_df = pd.DataFrame({
#             'company_name': ['Stripe', 'Airbnb', 'Coinbase', 'Discord', 'DoorDash', 'Zoom', 'Figma', 'Roblox'],
#             'investor': ['Sequoia', 'a16z', 'a16z', 'Greylock', 'Sequoia', 'Sequoia', 'a16z', 'Greylock'],
#             'data_source': ['Fallback VC Data'] * 8
#         })
        
#         output_path = 'data/vc_portfolio_companies.csv'
#         fallback_df.to_csv(output_path, index=False)
#         logging.info(f"Saved {len(fallback_df)} fallback VC portfolio companies to {output_path}")
#         return fallback_df

# def fetch_startup_news_mentions():
#     """Fetch startup mentions from news sources."""
#     logging.info("Fetching startup mentions from news sources...")
    
#     # Create fallback data for news mentions
#     fallback_companies = [
#         {'company_name': 'OpenAI', 'mention_count': 15, 'data_source': 'News Mentions Fallback'},
#         {'company_name': 'Anthropic', 'mention_count': 12, 'data_source': 'News Mentions Fallback'},
#         {'company_name': 'Scale AI', 'mention_count': 8, 'data_source': 'News Mentions Fallback'},
#         {'company_name': 'Databricks', 'mention_count': 7, 'data_source': 'News Mentions Fallback'},
#         {'company_name': 'Rippling', 'mention_count': 6, 'data_source': 'News Mentions Fallback'},
#         {'company_name': 'Plaid', 'mention_count': 5, 'data_source': 'News Mentions Fallback'}
#     ]
    
#     companies_df = pd.DataFrame(fallback_companies)
    
#     # Save to CSV
#     output_path = 'data/news_mentioned_companies.csv'
#     companies_df.to_csv(output_path, index=False)
#     logging.info(f"Saved {len(companies_df)} news-mentioned companies to {output_path}")
    
#     return companies_df

# def merge_all_sources():
#     """Merge all data sources into a single dataset."""
#     logging.info("Merging all data sources...")
    
#     # List of data files to merge
#     data_files = [
#         'data/github_bay_area_companies.csv',
#         'data/datasf_tech_businesses.csv',
#         'data/datasf_permit_companies.csv',
#         'data/smc_tech_businesses.csv',
#         'data/vc_portfolio_companies.csv',
#         'data/news_mentioned_companies.csv'
#     ]
    
#     # Check which files exist
#     existing_files = []
#     for file_path in data_files:
#         if os.path.exists(file_path):
#             existing_files.append(file_path)
#         else:
#             logging.warning(f"File not found: {file_path}")
    
#     if not existing_files:
#         logging.error("No data files found to merge")
        
#         # Create minimal dataset with the most essential Bay Area startups
#         minimal_df = pd.DataFrame({
#             'company_name': ['Airbnb', 'Stripe', 'Coinbase', 'DoorDash', 'Figma', 'Discord', 'Anthropic', 'OpenAI', 'Databricks', 'Plaid'],
#             'industry': ['Travel/Hospitality', 'Fintech', 'Cryptocurrency', 'Food Delivery', 'Design Software', 
#                         'Communications', 'AI', 'AI', 'Data Analytics', 'Fintech'],
#             'location': ['San Francisco', 'San Francisco', 'San Francisco', 'San Francisco', 'San Francisco', 
#                         'San Francisco', 'San Francisco', 'San Francisco', 'San Francisco', 'San Francisco'],
#             'founding_year': [2008, 2010, 2012, 2013, 2012, 2015, 2021, 2015, 2013, 2013],
#             'data_source': ['Fallback Data'] * 10
#         })
        
#         # Save to CSV
#         output_path = 'data/bay_area_startups_master.csv'
#         minimal_df.to_csv(output_path, index=False)
#         logging.info(f"Saved minimal dataset with {len(minimal_df)} companies to {output_path}")
#         return minimal_df
    
#     # Load and merge all existing data files
#     dfs = []
#     for file_path in existing_files:
#         try:
#             df = pd.read_csv(file_path)
#             if len(df) > 0:
#                 dfs.append(df)
#                 logging.info(f"Loaded {len(df)} records from {file_path}")
#         except Exception as e:
#             logging.error(f"Error loading {file_path}: {str(e)}")
    
#     if not dfs:
#         logging.error("No valid data loaded from files")
#         return None
    
#     # Concatenate all DataFrames
#     combined_df = pd.concat(dfs, ignore_index=True)
    
#     # Standardize column names
#     if 'company_name' not in combined_df.columns and 'Company' in combined_df.columns:
#         combined_df['company_name'] = combined_df['Company']
    
#     # Remove duplicates based on company name
#     combined_df = combined_df.drop_duplicates(subset=['company_name'])
    
#     # Clean company names
#     combined_df['company_name'] = combined_df['company_name'].astype(str).apply(
#         lambda x: x.strip().title() if pd.notnull(x) else None
#     )
    
#     # Add city to location if possible
#     if 'location' not in combined_df.columns:
#         combined_df['location'] = 'Bay Area'
    
#     # Save to CSV
#     output_path = 'data/bay_area_startups_master.csv'
#     combined_df.to_csv(output_path, index=False)
#     logging.info(f"Saved final dataset with {len(combined_df)} unique companies to {output_path}")
    
#     return combined_df

# def main():
#     """Main function to run the entire data collection pipeline."""
#     logging.info("Starting Bay Area startup data collection...")
    
#     # Fix SSL issues for macOS
#     fix_ssl_issue()
    
#     # Create data directory if it doesn't exist
#     if not os.path.exists('data'):
#         os.makedirs('data')
    
#     # Use concurrent.futures to parallelize some of the data collection
#     with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
#         # Start concurrent tasks
#         github_future = executor.submit(fetch_github_bay_area_companies)
#         datasf_future = executor.submit(fetch_datasf_businesses)
#         permits_future = executor.submit(fetch_datasf_building_permits)
        
#         # Wait for these tasks to complete
#         github_result = github_future.result()
#         datasf_result = datasf_future.result()
#         permits_result = permits_future.result()
        
#         logging.info(f"Completed initial data collection:")
#         logging.info(f"  - GitHub companies: {len(github_result) if isinstance(github_result, pd.DataFrame) else 0}")
#         logging.info(f"  - DataSF businesses: {len(datasf_result) if isinstance(datasf_result, pd.DataFrame) else 0}")
#         logging.info(f"  - Building permits: {len(permits_result) if isinstance(permits_result, pd.DataFrame) else 0}")
    
#     # Run the remaining data collection tasks sequentially
#     # These may involve more complex web scraping that benefits from sequential execution
#     logging.info("Starting secondary data collection...")
    
#     economic_data = fetch_bay_area_economic_data()
#     logging.info(f"Collected {len(economic_data) if isinstance(economic_data, pd.DataFrame) else 0} records from economic data")
    
#     vc_portfolio_data = scrape_bay_area_vc_portfolios()
#     logging.info(f"Collected {len(vc_portfolio_data) if isinstance(vc_portfolio_data, pd.DataFrame) else 0} records from VC portfolios")
    
#     news_data = fetch_startup_news_mentions()
#     logging.info(f"Collected {len(news_data) if isinstance(news_data, pd.DataFrame) else 0} records from news mentions")
    
#     # Merge all the collected data
#     final_dataset = merge_all_sources()
    
#     if final_dataset is not None and len(final_dataset) > 0:
#         logging.info(f"Successfully collected data on {len(final_dataset)} Bay Area startups")
#         logging.info(f"Final dataset saved to 'data/bay_area_startups_master.csv'")
        
#         # Generate basic statistics
#         if 'founding_year' in final_dataset.columns:
#             year_counts = final_dataset['founding_year'].value_counts().sort_index()
#             logging.info(f"Startups by founding year: {year_counts.to_dict()}")
        
#         if 'location' in final_dataset.columns:
#             city_counts = final_dataset['location'].value_counts().head(10)
#             logging.info(f"Top startup cities: {city_counts.to_dict()}")
        
#         if 'industry' in final_dataset.columns:
#             industry_counts = final_dataset['industry'].value_counts().head(10)
#             logging.info(f"Top industries: {industry_counts.to_dict()}")
            
#         return True
#     else:
#         logging.error("Failed to create final dataset")
#         return False

# if __name__ == "__main__":
#     success = main()
#     if success:
#         logging.info("Data collection completed successfully")
#     else:
#         logging.error("Data collection failed")



import os
import pandas as pd
import requests
from io import StringIO  # Fixed: Using StringIO from io, not pandas
import logging
from datetime import datetime, timedelta
import concurrent.futures
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create data directory
os.makedirs('data', exist_ok=True)

# Current date for startup filtering
current_date = datetime.now()
five_years_ago = current_date - timedelta(days=5*365)
cutoff_year = five_years_ago.year
logging.info(f"Filtering for startups founded since {cutoff_year}")

# Bay Area locations for filtering
bay_area_locations = [
    'San Francisco', 'Oakland', 'Berkeley', 'Emeryville', 'Alameda',
    'South San Francisco', 'San Jose', 'Palo Alto', 'Mountain View',
    'Sunnyvale', 'Santa Clara', 'Cupertino', 'Campbell', 'Los Gatos',
    'Menlo Park', 'Redwood City', 'San Mateo', 'Burlingame',
    'Foster City', 'San Carlos', 'Belmont', 'San Bruno', 'Fremont',
    'Newark', 'Milpitas', 'Los Altos', 'Sausalito', 'San Rafael',
    'Walnut Creek', 'Pleasanton', 'Livermore', 'Dublin', 'East Palo Alto'
]

def is_bay_area(location):
    """Check if a location is in the Bay Area"""
    if pd.isna(location) or not isinstance(location, str):
        return False
    
    location = location.lower()
    
    # Check for exact matches
    for city in bay_area_locations:
        if city.lower() == location:
            return True
    
    # Check for substring matches
    for city in bay_area_locations:
        if city.lower() in location:
            return True
    
    # Check for Bay Area mentions
    bay_area_terms = ['bay area', 'silicon valley', 'sf bay', 'east bay', 'south bay']
    for term in bay_area_terms:
        if term in location:
            return True
            
    return False

def standardize_columns(df, source_name):
    """Standardize column names across different data sources"""
    # Convert column names to lowercase and replace spaces with underscores
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Map various column names to standard names
    column_mapping = {
        'company_name': 'company_name',
        'company': 'company_name',
        'companyname': 'company_name',
        'name': 'company_name',
        
        'website': 'website',
        'url': 'website',
        
        'city': 'location',
        'headquarters_location': 'location',
        'location': 'location',
        'address_1': 'address',
        
        'state': 'state',
        
        'company_description': 'description',
        'description': 'description',
        'full_description': 'description',
        
        'year_founded': 'founding_year',
        'founded_year': 'founding_year',
        'founding_year': 'founding_year',
        
        'total_investment': 'funding',
        'last_funding_amount': 'funding',
        
        'industry': 'industry',
        'descriptors': 'tags',
        'tags': 'tags'
    }
    
    # Create standardized DataFrame
    std_df = pd.DataFrame()
    
    # Copy columns using the mapping
    for col in df.columns:
        if col in column_mapping:
            target_col = column_mapping[col]
            std_df[target_col] = df[col]
    
    # Extract founding year from dates if needed
    if 'dba_start_date' in df.columns and 'founding_year' not in std_df.columns:
        std_df['founding_year'] = pd.to_datetime(df['dba_start_date'], errors='coerce').dt.year
    
    # Add data source
    std_df['data_source'] = source_name
    
    # Ensure required columns exist
    for col in ['company_name', 'website', 'location', 'description', 'founding_year', 'funding', 'industry', 'tags']:
        if col not in std_df.columns:
            std_df[col] = None
    
    # Convert founding_year to numeric
    if 'founding_year' in std_df.columns:
        std_df['founding_year'] = pd.to_numeric(std_df['founding_year'], errors='coerce')
    
    # Clean up company names (remove quotes, etc.)
    if 'company_name' in std_df.columns:
        std_df['company_name'] = std_df['company_name'].astype(str).apply(
            lambda x: x.strip().replace('"', '').replace("'", "") if not pd.isna(x) else x
        )
    
    return std_df

def load_local_csv(filepath, source_name):
    """Load a local CSV file"""
    try:
        if os.path.exists(filepath):
            logging.info(f"Loading {filepath}")
            df = pd.read_csv(filepath)
            
            # Standardize columns
            std_df = standardize_columns(df, source_name)
            
            # Filter for valid companies
            if 'company_name' in std_df.columns:
                std_df = std_df[std_df['company_name'].notna() & (std_df['company_name'] != 'nan')]
            
            logging.info(f"Loaded {len(std_df)} records from {filepath}")
            return std_df
        else:
            logging.warning(f"File not found: {filepath}")
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error loading {filepath}: {str(e)}")
        return pd.DataFrame()

def fetch_datasf_businesses():
    """Fetch business data from DataSF API"""
    logging.info("Fetching business data from DataSF...")
    
    # DataSF Registered Business Locations dataset
    url = "https://data.sfgov.org/resource/g8m3-pdis.json"
    
    # Using dba_start_date (not business_start_date)
    params = {
        "$limit": 5000,
        "$where": f"dba_start_date >= '{five_years_ago.strftime('%Y-%m-%d')}'",
        "$order": "dba_start_date DESC"
    }
    
    try:
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            
            # Save raw data
            raw_output_path = 'data/datasf_businesses_raw.csv'
            df.to_csv(raw_output_path, index=False)
            logging.info(f"Saved raw data with {len(df)} records")
            
            # Create standardized DataFrame
            std_df = pd.DataFrame()
            std_df['company_name'] = df['dba_name'] if 'dba_name' in df.columns else None
            std_df['location'] = 'San Francisco'
            std_df['state'] = 'CA'
            std_df['founding_year'] = pd.to_datetime(df['dba_start_date'], errors='coerce').dt.year if 'dba_start_date' in df.columns else None
            std_df['data_source'] = 'DataSF API'
            
            # Filter for tech companies
            tech_keywords = ['tech', 'software', 'app', 'digital', 'ai', 'machine learning', 
                           'data', 'platform', 'saas', 'cloud', 'internet', 'web', 'mobile']
            
            # Check business name and description for tech keywords
            is_tech = False
            
            if 'dba_name' in df.columns:
                is_tech = is_tech | df['dba_name'].astype(str).str.lower().apply(
                    lambda x: any(keyword in x.lower() for keyword in tech_keywords)
                )
            
            if 'business_description' in df.columns:
                is_tech = is_tech | df['business_description'].astype(str).str.lower().apply(
                    lambda x: any(keyword in x.lower() for keyword in tech_keywords)
                )
            
            tech_df = std_df[is_tech]
            
            # Save tech businesses
            tech_df.to_csv('data/datasf_tech_businesses.csv', index=False)
            logging.info(f"Filtered to {len(tech_df)} tech businesses")
            
            return tech_df
        else:
            logging.error(f"DataSF API error: {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"DataSF API exception: {str(e)}")
        return pd.DataFrame()

def extract_founding_year_from_description(row):
    """Extract founding year from company description when available"""
    if pd.isna(row['founding_year']) and not pd.isna(row['description']):
        # Look for patterns like "founded in 2020" or "established in 2019"
        desc = str(row['description']).lower()
        founded_patterns = [
            r'founded in (\d{4})',
            r'established in (\d{4})',
            r'started in (\d{4})',
            r'launched in (\d{4})',
            r'since (\d{4})'
        ]
        
        for pattern in founded_patterns:
            match = re.search(pattern, desc)
            if match:
                return int(match.group(1))
    
    return row['founding_year']

def merge_all_datasets():
    """Merge all datasets and filter for Bay Area startups"""
    logging.info("Merging all datasets...")
    
    # List of all input files
    input_files = [
        {'file': 'silicon_valley_companies.csv', 'source': 'Silicon Valley Companies'},
        {'file': 'companies-that-use-laravel.csv', 'source': 'Laravel Companies'},
        {'file': 'venture-capital.csv', 'source': 'VC Firms'},
        {'file': 'tech-companies-in-oakland-06-20-2021.csv', 'source': 'Oakland Tech Companies'},
        {'file': 'y-combinator-companies.csv', 'source': 'Y Combinator Companies'},
        {'file': 'data/datasf_tech_businesses.csv', 'source': 'DataSF API'}
    ]
    
    all_dataframes = []
    
    # Load all CSV files
    for file_info in input_files:
        df = load_local_csv(file_info['file'], file_info['source'])
        if not df.empty:
            all_dataframes.append(df)
    
    # If no data loaded, return empty DataFrame
    if not all_dataframes:
        logging.error("No data loaded from any source")
        return pd.DataFrame()
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    logging.info(f"Combined {len(combined_df)} records from all sources")
    
    # Filter for Bay Area locations
    combined_df['is_bay_area'] = combined_df['location'].apply(is_bay_area)
    bay_area_df = combined_df[combined_df['is_bay_area']]
    logging.info(f"Filtered to {len(bay_area_df)} Bay Area companies")
    
    # Try to extract founding years from descriptions
    bay_area_df['founding_year'] = bay_area_df.apply(extract_founding_year_from_description, axis=1)
    
    # Filter for startups (founded in last 5 years)
    has_founding_year = bay_area_df['founding_year'].notna()
    is_startup = bay_area_df['founding_year'] >= cutoff_year
    
    # Get startups (with known founding year) and unknown companies
    startups_df = bay_area_df[has_founding_year & is_startup]
    unknown_df = bay_area_df[~has_founding_year]
    
    # Combine startups and unknown companies
    final_df = pd.concat([startups_df, unknown_df], ignore_index=True)
    logging.info(f"Final dataset includes {len(startups_df)} confirmed startups and {len(unknown_df)} companies with unknown founding year")
    
    # Remove duplicates by company name
    final_df = final_df.drop_duplicates(subset=['company_name'])
    logging.info(f"After removing duplicates, final dataset contains {len(final_df)} companies")
    
    # Make sure there are no NaN values in company_name
    final_df = final_df[final_df['company_name'].notna()]
    
    # Sort by company name
    final_df = final_df.sort_values('company_name').reset_index(drop=True)
    
    # Remove temporary column
    if 'is_bay_area' in final_df.columns:
        final_df = final_df.drop(columns=['is_bay_area'])
    
    # Save to CSV
    final_df.to_csv('bay_area_startups_master.csv', index=False)
    logging.info(f"Saved final dataset to bay_area_startups_master.csv")
    
    return final_df

def generate_dataset_stats(df):
    """Generate statistics about the dataset"""
    logging.info("Generating dataset statistics...")
    
    # Count by location
    if 'location' in df.columns:
        location_counts = df['location'].value_counts().head(10)
        logging.info(f"Top 10 locations: {location_counts.to_dict()}")
    
    # Count by founding year
    if 'founding_year' in df.columns:
        # Count only valid years
        valid_years = df['founding_year'].dropna()
        if len(valid_years) > 0:
            year_counts = valid_years.value_counts().sort_index()
            logging.info(f"Companies by founding year: {year_counts.to_dict()}")
    
    # Count by industry
    if 'industry' in df.columns and not df['industry'].isna().all():
        industry_counts = df['industry'].value_counts().head(10)
        logging.info(f"Top 10 industries: {industry_counts.to_dict()}")
    
    # Count by data source
    if 'data_source' in df.columns:
        source_counts = df['data_source'].value_counts()
        logging.info(f"Companies by data source: {source_counts.to_dict()}")

def main():
    """Main function to run the entire data collection and merging process"""
    logging.info("Starting Bay Area startup data collection...")
    
    # Fetch data from DataSF API
    datasf_df = fetch_datasf_businesses()
    
    # Merge all datasets
    final_df = merge_all_datasets()
    
    if not final_df.empty:
        # Generate statistics
        generate_dataset_stats(final_df)
        logging.info("Data collection and merging completed successfully")
        return True
    else:
        logging.error("Failed to create dataset")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("Successfully created Bay Area startup dataset!")
        print("Data saved to bay_area_startups_master.csv")
    else:
        print("Failed to create dataset. Check the logs for details.")

