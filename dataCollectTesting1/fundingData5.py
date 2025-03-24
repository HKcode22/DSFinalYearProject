import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Create directories for output
os.makedirs("data", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

print("=== Bay Area Startup Data Collection ===")

# 1. Collect business registration data from DataSF
def get_datasf_businesses():
    """Get startup data from DataSF API"""
    print("Fetching business data from DataSF API...")
    
    # Registered Business Locations dataset ID (from search result #14)
    dataset_id = "g8m3-pdis"
    url = f"https://data.sfgov.org/resource/{dataset_id}.json"
    
    # We're only interested in recent businesses (last 5 years = potential startups)
    five_years_ago = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%dT00:00:00')
    
    # Set parameters to get businesses started in last 5 years
    params = {
        "$where": f"location_start_date >= '{five_years_ago}'",
        "$limit": 10000
    }
    
    headers = {}
    # If you have an app token, add it here
    app_token = os.environ.get("DATASF_APP_TOKEN")
    if app_token:
        headers["X-App-Token"] = app_token
    
    try:
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            businesses = response.json()
            df = pd.DataFrame(businesses)
            print(f"Retrieved {len(df)} businesses from DataSF API")
            
            # Try to get NAICS codes from Business Register dataset
            register_dataset_id = "u6xu-nkq2"
            register_url = f"https://data.sfgov.org/resource/{register_dataset_id}.json"
            
            # Use a sample of business IDs to avoid timeout issues
            if 'ttxid' in df.columns:
                sample_ids = df['ttxid'].dropna().sample(min(1000, len(df))).tolist()
                
                # Break IDs into chunks to avoid URL length limits
                id_chunks = [sample_ids[i:i+50] for i in range(0, len(sample_ids), 50)]
                
                register_data = []
                for chunk in id_chunks:
                    id_filter = " OR ".join([f"ttxid='{id}'" for id in chunk])
                    register_params = {"$where": id_filter, "$limit": 5000}
                    
                    try:
                        register_response = requests.get(register_url, headers=headers, params=register_params)
                        if register_response.status_code == 200:
                            register_data.extend(register_response.json())
                        time.sleep(1)  # Respect API rate limits
                    except Exception as e:
                        print(f"Error fetching register data chunk: {e}")
                
                register_df = pd.DataFrame(register_data)
                print(f"Retrieved {len(register_df)} business register records")
                
                # Look for tech companies based on NAICS codes
                if len(register_df) > 0 and 'naic_code' in register_df.columns:
                    # Tech sector NAICS codes
                    tech_naics = ['5112', '5415', '5182', '5191', '5417', '5413', '5419']
                    register_df['is_tech'] = register_df['naic_code'].str.startswith(tuple(tech_naics), na=False)
                    
                    # Merge with main business data
                    if 'ttxid' in register_df.columns:
                        df = pd.merge(df, register_df[['ttxid', 'naic_code', 'naic_code_description', 'is_tech']], 
                                     on='ttxid', how='left')
                        
                        # Filter for tech companies
                        tech_businesses = df[df['is_tech'] == True].copy()
                        tech_businesses['data_source'] = 'DataSF'
                        print(f"Identified {len(tech_businesses)} tech startups from DataSF")
                        return tech_businesses
            
            # If we couldn't get NAICS codes, use keyword matching as fallback
            startup_keywords = ['tech', 'software', 'app', 'digital', 'cyber', 'ai', 'data', 
                              'analytics', 'cloud', 'platform', 'biotech', 'fintech', 'startup']
            
            if 'dba_name' in df.columns:
                df['is_tech_keyword'] = df['dba_name'].str.lower().str.contains(
                    '|'.join(startup_keywords), case=False, na=False)
                tech_businesses = df[df['is_tech_keyword'] == True].copy()
                tech_businesses['data_source'] = 'DataSF'
                print(f"Identified {len(tech_businesses)} potential tech startups by keywords")
                return tech_businesses
            
            return pd.DataFrame()  # Empty dataframe if we couldn't find any tech companies
        else:
            print(f"Error fetching DataSF data: {response.status_code}")
            return pd.DataFrame()
    
    except Exception as e:
        print(f"Error with DataSF API: {e}")
        return pd.DataFrame()

# 2. Get funding data from GitHub repositories
def get_github_funding_data():
    """Get startup funding data from GitHub repositories"""
    print("Fetching startup funding data from GitHub repositories...")
    
    # Based on search result #13, there's a funding.csv file in this repo
    url = "https://raw.githubusercontent.com/kyang01/startup-analysis/master/data/funding.csv"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Import the CSV data
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            print(f"Retrieved {len(df)} funding records from GitHub")
            
            # Filter for Bay Area companies
            bay_area_terms = ['San Francisco', 'SF', 'Palo Alto', 'Menlo Park', 'Mountain View',
                             'Sunnyvale', 'San Jose', 'Berkeley', 'Oakland', 'San Mateo',
                             'Redwood City', 'Fremont', 'South San Francisco']
            
            # Look for location columns
            location_cols = [col for col in df.columns if any(loc in col.lower() for loc in ['location', 'city', 'address'])]
            
            if location_cols:
                # Create a filter across all location columns
                bay_area_mask = pd.Series(False, index=df.index)
                for col in location_cols:
                    if df[col].dtype == object:
                        mask = df[col].str.contains('|'.join(bay_area_terms), case=False, na=False)
                        bay_area_mask = bay_area_mask | mask
                
                bay_area_df = df[bay_area_mask].copy()
                bay_area_df['data_source'] = 'GitHub Repository'
                print(f"Found {len(bay_area_df)} Bay Area startup funding records")
                return bay_area_df
            else:
                # If no location column, try another approach
                print("No location column found in GitHub data")
                
                # Try a second repository (search result #17)
                second_url = "https://raw.githubusercontent.com/nihalrai/tech-companies-bay-area/master/Bay-Area-Companies-List.csv"
                try:
                    second_response = requests.get(second_url)
                    if second_response.status_code == 200:
                        second_df = pd.read_csv(StringIO(second_response.text))
                        second_df['data_source'] = 'GitHub Bay Area Companies'
                        print(f"Retrieved {len(second_df)} Bay Area companies from second repository")
                        return second_df
                except Exception as e:
                    print(f"Error with second GitHub repository: {e}")
            
            return pd.DataFrame()
        else:
            print(f"Error fetching GitHub data: {response.status_code}")
            
            # Use fallback data if GitHub fetch fails
            print("Using fallback sample data")
            sample_data = [
                {"company_name": "Stripe", "funding_total_usd": "2200000000", "category": "FinTech", "city": "San Francisco"},
                {"company_name": "Airtable", "funding_total_usd": "735000000", "category": "Software", "city": "San Francisco"},
                {"company_name": "Notion", "funding_total_usd": "343000000", "category": "Productivity", "city": "San Francisco"},
                {"company_name": "Figma", "funding_total_usd": "330000000", "category": "Design", "city": "San Francisco"}
            ]
            sample_df = pd.DataFrame(sample_data)
            sample_df['data_source'] = 'Sample Data'
            return sample_df
    
    except Exception as e:
        print(f"Error fetching GitHub data: {e}")
        return pd.DataFrame()

# 3. Get Growth List startup data
def get_growth_list_data():
    """Get startup data from Growth List"""
    print("Fetching startup data from Growth List...")
    
    # Based on search result #2, Growth List offers San Francisco startup data
    url = "https://growthlist.co/san-francisco-startups/"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for tables with startup data
            tables = soup.find_all('table')
            
            if tables:
                startup_data = []
                
                # Extract table headers
                header_row = tables[0].find('tr')
                headers = [th.text.strip() for th in header_row.find_all('th')]
                
                # Extract table data
                for row in tables[0].find_all('tr')[1:]:  # Skip header row
                    cells = [td.text.strip() for td in row.find_all('td')]
                    if len(cells) == len(headers):
                        startup_data.append(dict(zip(headers, cells)))
                
                df = pd.DataFrame(startup_data)
                df['data_source'] = 'Growth List'
                print(f"Retrieved {len(df)} startups from Growth List")
                return df
            else:
                print("No tables found on Growth List page")
                
                # Fallback to recent funding data from search result #2
                print("Using fallback data from search results")
                sample_data = [
                    {"Company": "Mercor", "Funding Amount (USD)": "75000000", "Funding Type": "Series B", "Industry": "Artificial Intelligence"},
                    {"Company": "Created by Humans", "Funding Amount (USD)": "5500000", "Funding Type": "Seed", "Industry": "Artificial Intelligence"},
                    {"Company": "Manas AI", "Funding Amount (USD)": "24600000", "Funding Type": "Seed", "Industry": "Artificial Intelligence"},
                    {"Company": "BluWhale", "Funding Amount (USD)": "75000000", "Funding Type": "ICO", "Industry": "Artificial Intelligence"},
                    {"Company": "Savant Labs", "Funding Amount (USD)": "18500000", "Funding Type": "Series A", "Industry": "Artificial Intelligence"}
                ]
                sample_df = pd.DataFrame(sample_data)
                sample_df['data_source'] = 'Fundraise Insider'
                return sample_df
        else:
            print(f"Error fetching Growth List data: {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching Growth List data: {e}")
        return pd.DataFrame()

# 4. Get top-funded Bay Area unicorns from search results
def get_top_unicorns():
    """Get data on top-funded Bay Area unicorns mentioned in search results"""
    print("Creating dataset of top-funded Bay Area unicorns...")
    
    # Based on search results #8 and #10
    unicorn_data = [
        {
            "company_name": "OpenAI",
            "funding_total_usd": "11300000000",
            "category": "AI",
            "city": "San Francisco",
            "founded_year": 2015
        },
        {
            "company_name": "Anthropic",
            "funding_total_usd": "4100000000",
            "category": "AI",
            "city": "San Francisco",
            "founded_year": 2021
        },
        {
            "company_name": "Databricks",
            "funding_total_usd": "3500000000",
            "category": "Data & Analytics",
            "city": "San Francisco",
            "founded_year": 2013
        },
        {
            "company_name": "Stripe",
            "funding_total_usd": "2200000000",
            "category": "FinTech",
            "city": "San Francisco",
            "founded_year": 2010
        },
        {
            "company_name": "Plaid",
            "funding_total_usd": "734000000",
            "category": "FinTech",
            "city": "San Francisco",
            "founded_year": 2013
        }
    ]
    
    df = pd.DataFrame(unicorn_data)
    df['data_source'] = 'Top Unicorns'
    print(f"Created dataset with {len(df)} top-funded Bay Area unicorns")
    return df

# 5. Clean and standardize the data
def clean_data(df, data_type='funding'):
    """Clean and standardize the datasets for merging"""
    if df.empty:
        return df
    
    # Make a copy to avoid warnings
    cleaned_df = df.copy()
    
    # Standardize column names
    if data_type == 'funding':
        # Map variant column names to standard names
        column_mapping = {
            'company_name': 'name',
            'Company': 'name',
            'dba_name': 'name',
            'funding_total_usd': 'funding_amount',
            'Funding Amount (USD)': 'funding_amount',
            'Funding Type': 'funding_stage',
            'category': 'industry',
            'Industry': 'industry',
            'naic_code_description': 'industry',
            'city': 'location',
            'full_business_address': 'address'
        }
    else:
        column_mapping = {
            'company_name': 'name',
            'Company': 'name',
            'dba_name': 'name',
            'category': 'industry',
            'Industry': 'industry',
            'naic_code_description': 'industry',
            'city': 'location',
            'full_business_address': 'address'
        }
    
    # Rename columns if they exist
    for old_col, new_col in column_mapping.items():
        if old_col in cleaned_df.columns and new_col not in cleaned_df.columns:
            cleaned_df.rename(columns={old_col: new_col}, inplace=True)
    
    # Ensure data_source is present
    if 'data_source' not in cleaned_df.columns:
        cleaned_df['data_source'] = 'Unknown'
    
    # Clean funding amounts if present
    if 'funding_amount' in cleaned_df.columns:
        # Remove non-numeric characters
        if cleaned_df['funding_amount'].dtype == object:
            cleaned_df['funding_amount'] = cleaned_df['funding_amount'].str.replace(r'[$,]', '', regex=True)
            cleaned_df['funding_amount'] = pd.to_numeric(cleaned_df['funding_amount'], errors='coerce')
    
    # Add is_startup flag
    cleaned_df['is_startup'] = True
    
    # Determine startup stage if funding data is available
    if 'funding_amount' in cleaned_df.columns:
        conditions = [
            cleaned_df['funding_amount'].isna(),
            cleaned_df['funding_amount'] <= 1000000,
            cleaned_df['funding_amount'] <= 10000000,
            cleaned_df['funding_amount'] <= 50000000,
            cleaned_df['funding_amount'] <= 100000000
        ]
        choices = [
            'Unknown',
            'Seed Stage',
            'Early Stage',
            'Growth Stage',
            'Late Stage',
            'Unicorn'
        ]
        cleaned_df['funding_category'] = np.select(conditions, choices[:len(conditions)], choices[-1])
    
    return cleaned_df

# 6. Combine all data sources
def combine_datasets(datasets):
    """Combine all datasets into a single dataframe"""
    print("\nCombining all datasets...")
    
    # Filter out empty dataframes
    valid_datasets = [df for df in datasets if not df.empty]
    
    if not valid_datasets:
        print("No valid datasets to combine")
        return pd.DataFrame()
    
    combined_df = pd.concat(valid_datasets, ignore_index=True)
    
    # Remove duplicates based on name
    if 'name' in combined_df.columns:
        combined_df.drop_duplicates(subset=['name'], keep='first', inplace=True)
        print(f"Combined dataset has {len(combined_df)} unique startups")
    else:
        print(f"Combined dataset has {len(combined_df)} records (couldn't check for duplicates)")
    
    return combined_df

# 7. Create visualizations
def create_visualizations(df):
    """Create visualizations from the startup data"""
    if df.empty:
        print("No data available for visualizations")
        return
    
    print("\nCreating visualizations...")
    
    # 1. Data sources distribution
    plt.figure(figsize=(10, 6))
    source_counts = df['data_source'].value_counts()
    source_counts.plot(kind='bar', color='skyblue')
    plt.title('Bay Area Startup Data Sources')
    plt.xlabel('Source')
    plt.ylabel('Number of Startups')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/data_sources.png')
    print("Created data sources visualization")
    
    # 2. Industry distribution
    if 'industry' in df.columns:
        plt.figure(figsize=(12, 8))
        industry_counts = df['industry'].value_counts().head(10)
        industry_counts.plot(kind='barh', color='lightgreen')
        plt.title('Top 10 Industries Among Bay Area Startups')
        plt.xlabel('Number of Startups')
        plt.tight_layout()
        plt.savefig('visualizations/top_industries.png')
        print("Created industry distribution visualization")
    
    # 3. Funding distribution
    if 'funding_amount' in df.columns:
        funded_df = df[df['funding_amount'].notna()]
        if not funded_df.empty:
            plt.figure(figsize=(10, 6))
            plt.hist(funded_df['funding_amount'], bins=20, color='salmon')
            plt.title('Distribution of Funding Amounts for Bay Area Startups')
            plt.xlabel('Funding Amount (USD)')
            plt.ylabel('Number of Startups')
            plt.xscale('log')  # Use log scale for better visualization
            plt.tight_layout()
            plt.savefig('visualizations/funding_distribution.png')
            print("Created funding distribution visualization")
            
            # 4. Top funded startups
            plt.figure(figsize=(12, 8))
            top_funded = funded_df.sort_values('funding_amount', ascending=False).head(10)
            sns.barplot(x='funding_amount', y='name', data=top_funded)
            plt.title('Top 10 Funded Bay Area Startups')
            plt.xlabel('Funding Amount (USD)')
            plt.tight_layout()
            plt.savefig('visualizations/top_funded_startups.png')
            print("Created top funded startups visualization")
    
    # 5. Funding stages
    if 'funding_category' in df.columns:
        plt.figure(figsize=(10, 6))
        category_counts = df['funding_category'].value_counts()
        category_counts.plot(kind='bar', color='lightblue')
        plt.title('Bay Area Startups by Funding Stage')
        plt.xlabel('Funding Stage')
        plt.ylabel('Number of Startups')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('visualizations/funding_stages.png')
        print("Created funding stages visualization")

# Main execution function
def main():
    # Step A: Collect data from all sources
    datasf_data = get_datasf_businesses()
    github_data = get_github_funding_data()
    growth_list_data = get_growth_list_data()
    unicorn_data = get_top_unicorns()
    
    # Step B: Clean and standardize each dataset
    cleaned_datasf = clean_data(datasf_data, 'business')
    cleaned_github = clean_data(github_data, 'funding')
    cleaned_growth_list = clean_data(growth_list_data, 'funding')
    cleaned_unicorns = clean_data(unicorn_data, 'funding')
    
    # Step C: Save individual datasets
    if not cleaned_datasf.empty:
        cleaned_datasf.to_csv('data/datasf_startups.csv', index=False)
        print(f"Saved {len(cleaned_datasf)} DataSF startup records")
    
    if not cleaned_github.empty:
        cleaned_github.to_csv('data/github_startups.csv', index=False)
        print(f"Saved {len(cleaned_github)} GitHub startup records")
    
    if not cleaned_growth_list.empty:
        cleaned_growth_list.to_csv('data/growth_list_startups.csv', index=False)
        print(f"Saved {len(cleaned_growth_list)} Growth List startup records")
    
    if not cleaned_unicorns.empty:
        cleaned_unicorns.to_csv('data/unicorn_startups.csv', index=False)
        print(f"Saved {len(cleaned_unicorns)} unicorn startup records")
    
    # Step D: Combine all datasets
    all_datasets = [cleaned_datasf, cleaned_github, cleaned_growth_list, cleaned_unicorns]
    combined_data = combine_datasets(all_datasets)
    
    # Step E: Save the combined dataset
    if not combined_data.empty:
        combined_data.to_csv('data/bay_area_startups_combined.csv', index=False)
        print(f"Saved combined dataset with {len(combined_data)} records")
        
        # Step F: Create visualizations
        create_visualizations(combined_data)
    
    print("\n=== Bay Area Startup Data Collection Complete ===")
    print("All data files have been saved to the 'data' directory")
    print("Visualizations have been saved to the 'visualizations' directory")

# Run the script
if __name__ == "__main__":
    main()


# Run this code to examine the files
import pandas as pd

github_df = pd.read_csv('data/github_startups.csv')
growth_list_df = pd.read_csv('data/growth_list_startups.csv')
unicorn_df = pd.read_csv('data/unicorn_startups.csv')
combined_df = pd.read_csv('data/bay_area_startups_combined.csv')

print(f"GitHub: {len(github_df)} records, columns: {github_df.columns.tolist()}")
print(f"Growth List: {len(growth_list_df)} records, columns: {growth_list_df.columns.tolist()}")
print(f"Unicorns: {len(unicorn_df)} records, columns: {unicorn_df.columns.tolist()}")
print(f"Combined: {len(combined_df)} records, columns: {combined_df.columns.tolist()}")

# Check for name columns in each dataset
for df_name, df in [('GitHub', github_df), ('Growth List', growth_list_df), ('Unicorns', unicorn_df)]:
    if 'name' in df.columns:
        print(f"{df_name} - First 5 names: {df['name'].head().tolist()}")
    else:
        print(f"{df_name} - No 'name' column found. Available columns: {df.columns.tolist()}")
