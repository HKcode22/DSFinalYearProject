import pandas as pd
import requests
from io import StringIO
from bs4 import BeautifulSoup
import os
import re
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Create directories for output
os.makedirs("data", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

print("=== Bay Area Startup Funding Data Collection ===")

# Function to get funding data from GitHub public dataset
def get_funding_data_from_github():
    """Retrieve startup funding data from GitHub repository"""
    print("Fetching startup funding data from GitHub...")
    
    # URL of the raw funding CSV file
    url = "https://raw.githubusercontent.com/kyang01/startup-analysis/master/data/funding.csv"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Read the CSV content
        data = pd.read_csv(StringIO(response.text))
        print(f"Successfully retrieved data with {len(data)} records")
        
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return pd.DataFrame()

# Function to get AngelList/Wellfound startup data with improved anti-detection
def get_angellist_data():
    """Extract startup data from AngelList/Wellfound with improved methods"""
    print("Fetching startup data from AngelList/Wellfound...")
    
    # AngelList shows SF startups data according to search result #14
    url = "https://wellfound.com/job-collections/fastest-growing-startups-in-san-francisco-hiring-now"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Cache-Control': 'max-age=0',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1'
    }
    
    # Use a session to maintain cookies
    session = requests.Session()
    
    try:
        # First visit the homepage to get cookies
        session.get("https://wellfound.com/", headers=headers)
        
        # Add a delay to mimic human behavior
        time.sleep(random.uniform(2, 4))
        
        # Now visit the target page
        response = session.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract company data based on structure in search result #14
        companies = []
        
        # Look for company sections
        company_sections = soup.find_all('div', class_=lambda x: x and ('startup-card' in x or 'company-card' in x))
        
        if not company_sections:
            # Try alternative selectors if the above doesn't work
            company_sections = soup.select('div.styles_company__X2y86, div[data-test="startup-item"]')
        
        for section in company_sections:
            company = {}
            
            # Try to extract name
            name_element = section.select_one('h3, h4, [class*="name"], strong, b')
            if name_element:
                company['name'] = name_element.text.strip()
            
            # Try to extract stage info (if available)
            stage_element = section.select_one('[class*="stage"]')
            if stage_element:
                company['stage'] = stage_element.text.strip()
            
            # Try to extract funding info (search for patterns like $20M, $150k, etc.)
            for element in section.find_all(text=True):
                if '$' in element:
                    funding_match = re.search(r'\$\s*(\d+(?:\.\d+)?)\s*([kmbt])?', element, re.IGNORECASE)
                    if funding_match:
                        amount = float(funding_match.group(1))
                        unit = funding_match.group(2).lower() if funding_match.group(2) else ''
                        
                        # Convert to standard form
                        if unit == 'k':
                            amount *= 1_000
                        elif unit == 'm':
                            amount *= 1_000_000
                        elif unit == 'b':
                            amount *= 1_000_000_000
                        elif unit == 't':
                            amount *= 1_000_000_000_000
                            
                        company['funding_amount'] = amount
                        break
            
            # Look for industry/category
            for element in section.select('[class*="industry"], [class*="category"]'):
                company['industry'] = element.text.strip()
                break
            
            # Only add companies with a name
            if 'name' in company:
                company['source'] = 'AngelList/Wellfound'
                companies.append(company)
        
        df = pd.DataFrame(companies)
        print(f"Successfully extracted {len(df)} companies from AngelList/Wellfound")
        return df
        
    except Exception as e:
        print(f"Error extracting AngelList data: {e}")
        return pd.DataFrame()

# Function to get Growth List data
def get_growth_list_data():
    """Extract funding data from Growth List website"""
    print("Fetching funding data from Growth List...")
    
    # URL for San Francisco startups
    url = "https://growthlist.co/san-francisco-startups/"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract table data
        tables = soup.find_all('table')
        
        if not tables:
            print("No tables found on Growth List page")
            return pd.DataFrame()
        
        # Extract table data
        funding_data = []
        
        # Get headers from the first row
        header_row = tables[0].find('tr')
        headers = [th.text.strip() for th in header_row.find_all('th')]
        
        # Get data rows (skip header)
        for row in tables[0].find_all('tr')[1:]:
            cells = [td.text.strip() for td in row.find_all('td')]
            if len(cells) == len(headers):
                funding_data.append(dict(zip(headers, cells)))
        
        # Convert to DataFrame
        df = pd.DataFrame(funding_data)
        print(f"Successfully extracted {len(df)} funding records from Growth List")
        return df
    
    except Exception as e:
        print(f"Error extracting Growth List data: {e}")
        return pd.DataFrame()

# Function to filter for Bay Area startups
def filter_bay_area_startups(df):
    """Filter dataframe to only include Bay Area startups"""
    # Define Bay Area locations to search for
    bay_area_locations = [
        'San Francisco', 'SF', 'Palo Alto', 'Menlo Park', 'Mountain View',
        'Sunnyvale', 'Santa Clara', 'San Jose', 'Berkeley', 'Oakland',
        'San Mateo', 'Redwood City', 'South San Francisco', 'Fremont', 
        'Emeryville', 'San Rafael', 'Mill Valley', 'Burlingame', 'Bay Area'
    ]
    
    if df.empty:
        return df
    
    # Look for location columns
    location_cols = []
    for col in df.columns:
        if any(loc_term in col.lower() for loc_term in ['city', 'location', 'region', 'area', 'headquarters']):
            location_cols.append(col)
    
    # If we found location columns, use them for filtering
    if location_cols:
        # Start with empty mask
        bay_area_mask = pd.Series(False, index=df.index)
        
        # Update mask for each location column
        for col in location_cols:
            if df[col].dtype == object:  # Check if it's a string column
                col_mask = df[col].str.contains('|'.join(bay_area_locations), case=False, na=False)
                bay_area_mask = bay_area_mask | col_mask
        
        filtered_df = df[bay_area_mask]
        original_count = len(df)
        filtered_count = len(filtered_df)
        print(f"Filtered from {original_count} to {filtered_count} Bay Area startups")
        return filtered_df
    else:
        # If we don't have location columns but the data is supposed to be SF/Bay Area-specific
        # (like from Growth List SF or AngelList SF), we'll keep all rows
        print("No location column found for filtering, assuming all data is Bay Area")
        return df

# Function to clean funding amounts
def clean_funding_amounts(df):
    """Clean funding amount columns to consistent numeric format"""
    # Different column names that might contain funding information
    funding_columns = [
        'funding_total_usd', 'funding_amount', 'Funding Amount (USD)', 
        'Funding Amount', 'funding', 'amount', 'raised'
    ]
    
    for col in funding_columns:
        if col in df.columns:
            if df[col].dtype == object:  # If it's a string column
                # Fix the regex escape sequence issue by using a raw string
                df[col] = df[col].astype(str).replace(r'[$,]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
            break
    
    return df

# Collect data from multiple sources
github_data = get_funding_data_from_github()
angellist_data = get_angellist_data()
growth_list_data = get_growth_list_data()

# Filter and clean each dataset
if not github_data.empty:
    github_data = filter_bay_area_startups(github_data)
    github_data = clean_funding_amounts(github_data)
    github_data['source'] = 'GitHub Repository'
    github_data.to_csv('data/github_bay_area_startup_funding.csv', index=False)
    print("Saved GitHub Bay Area startup funding data")

if not angellist_data.empty:
    # AngelList data should already be Bay Area specific
    angellist_data = clean_funding_amounts(angellist_data)
    angellist_data.to_csv('data/angellist_bay_area_startup_funding.csv', index=False)
    print("Saved AngelList Bay Area startup funding data")

if not growth_list_data.empty:
    # Growth List data should already be Bay Area specific
    growth_list_data = clean_funding_amounts(growth_list_data)
    growth_list_data.to_csv('data/growth_list_bay_area_startup_funding.csv', index=False)
    print("Saved Growth List Bay Area startup funding data")

# Create a combined dataset
print("Creating combined dataset...")
dataframes_to_combine = []

if not github_data.empty:
    # Select relevant columns
    github_cols = ['name', 'funding_total_usd', 'market', 'funding_rounds', 'source']
    if all(col in github_data.columns for col in github_cols):
        github_subset = github_data[github_cols].copy()
        github_subset.rename(columns={'funding_total_usd': 'funding_amount', 'market': 'industry'}, inplace=True)
        dataframes_to_combine.append(github_subset)

if not angellist_data.empty:
    # Standardize column names
    angellist_subset = angellist_data.copy()
    if 'industry' not in angellist_subset.columns and 'category' in angellist_subset.columns:
        angellist_subset.rename(columns={'category': 'industry'}, inplace=True)
    dataframes_to_combine.append(angellist_subset)

if not growth_list_data.empty:
    # Standardize column names
    growth_list_subset = growth_list_data.copy()
    if 'Funding Amount (USD)' in growth_list_subset.columns:
        growth_list_subset.rename(columns={'Funding Amount (USD)': 'funding_amount'}, inplace=True)
    if 'Company' in growth_list_subset.columns:
        growth_list_subset.rename(columns={'Company': 'name'}, inplace=True)
    if 'Industry' in growth_list_subset.columns:
        growth_list_subset.rename(columns={'Industry': 'industry'}, inplace=True)
    growth_list_subset['source'] = 'Growth List'
    dataframes_to_combine.append(growth_list_subset)

# Combine all dataframes if we have any
if dataframes_to_combine:
    # Find common columns
    common_columns = set.intersection(*[set(df.columns) for df in dataframes_to_combine])
    
    # Ensure we have the essential columns
    essential_columns = ['name', 'funding_amount', 'industry', 'source']
    common_columns = list(common_columns.union(essential_columns))
    
    # Prepare dataframes with consistent columns
    standardized_dfs = []
    for df in dataframes_to_combine:
        std_df = pd.DataFrame()
        for col in common_columns:
            if col in df.columns:
                std_df[col] = df[col]
            else:
                std_df[col] = None
        standardized_dfs.append(std_df)
    
    # Combine all dataframes
    combined_df = pd.concat(standardized_dfs, ignore_index=True)
    
    # Remove duplicates based on company name
    combined_df = combined_df.drop_duplicates(subset=['name'])
    
    # Save the combined dataset
    combined_df.to_csv('data/combined_bay_area_startup_funding.csv', index=False)
    print(f"Saved combined dataset with {len(combined_df)} records")
    
    # Create visualizations
    print("Creating visualizations...")
    
    # 1. Funding distribution by source
    plt.figure(figsize=(10, 6))
    source_counts = combined_df['source'].value_counts()
    source_counts.plot(kind='bar')
    plt.title('Startup Data by Source')
    plt.xlabel('Data Source')
    plt.ylabel('Number of Startups')
    plt.tight_layout()
    plt.savefig('visualizations/startup_data_by_source.png')
    
    # 2. Industry distribution
    if 'industry' in combined_df.columns:
        plt.figure(figsize=(12, 8))
        industry_counts = combined_df['industry'].value_counts().head(10)
        industry_counts.plot(kind='barh')
        plt.title('Top 10 Industries Among Bay Area Startups')
        plt.xlabel('Number of Startups')
        plt.tight_layout()
        plt.savefig('visualizations/top_industries.png')
        print("Created industry visualization")
    
    # 3. Funding amount distribution
    if 'funding_amount' in combined_df.columns:
        # Remove NaN values for the histogram
        funding_data = combined_df['funding_amount'].dropna()
        
        if not funding_data.empty:
            plt.figure(figsize=(10, 6))
            # Plot with a logarithmic scale because funding amounts can vary widely
            plt.hist(funding_data, bins=30)
            plt.xscale('log')
            plt.title('Distribution of Funding Amounts (Log Scale)')
            plt.xlabel('Funding Amount (USD)')
            plt.ylabel('Number of Startups')
            plt.tight_layout()
            plt.savefig('visualizations/funding_distribution.png')
            print("Created funding distribution visualization")

print("\n=== Data Collection Complete ===")
print("All data files have been saved to the 'data' directory")
print("Visualizations have been saved to the 'visualizations' directory")
