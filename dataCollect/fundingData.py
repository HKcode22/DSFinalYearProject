# Import necessary libraries
import pandas as pd
import requests
from io import StringIO
from bs4 import BeautifulSoup
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
import re

# Create directories for output
os.makedirs("data", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

# First, let's collect GitHub data which is the most reliable free source
def get_funding_data_from_github():
    """Retrieve startup funding data from GitHub repository"""
    print("Fetching startup funding data from GitHub...")
    
    # URL of the raw funding CSV file found in search result #7
    url = "https://raw.githubusercontent.com/kyang01/startup-analysis/master/data/funding.csv"
    
    try:
        # Fetch the content
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Read the CSV content
        data = pd.read_csv(StringIO(response.text))
        print(f"Successfully retrieved data with {len(data)} records")
        
        # Print column names to debug
        print(f"Available columns: {data.columns.tolist()}")
        
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

# Function to filter for Bay Area startups
def filter_bay_area_startups(df):
    """Filter dataframe to only include Bay Area startups"""
    # Define Bay Area locations to search for
    bay_area_locations = [
        'San Francisco', 'SF', 'Palo Alto', 'Menlo Park', 'Mountain View',
        'Sunnyvale', 'Santa Clara', 'San Jose', 'Berkeley', 'Oakland',
        'San Mateo', 'Redwood City', 'South San Francisco', 'Fremont', 
        'Emeryville', 'San Rafael', 'Mill Valley', 'Burlingame'
    ]
    
    # Print a sample row to debug
    if len(df) > 0:
        print("Sample row for debugging:")
        print(df.iloc[0])
    
    # Check if there's a city or location column
    location_col = None
    for col in ['city', 'location', 'city_name', 'headquarters_location', 'state_code', 'country_code']:
        if col in df.columns:
            location_col = col
            print(f"Found location column: {col}")
            break
    
    # If we didn't find a standard column, let's look for any column that might contain location info
    if not location_col:
        for col in df.columns:
            if any(location in str(df[col].iloc[0]).lower() for location in ['california', 'ca', 'san']):
                location_col = col
                print(f"Found potential location column: {col}")
                break
    
    if location_col:
        # Create mask for Bay Area locations
        if df[location_col].dtype == object:  # Check if it's a string column
            location_filter = df[location_col].str.contains('|'.join(bay_area_locations), 
                                                         case=False, 
                                                         na=False)
            filtered_df = df[location_filter]
            print(f"Filtered to {len(filtered_df)} Bay Area startups")
            return filtered_df
        else:
            print(f"Column {location_col} is not a string type, skipping filtering")
            return df
    else:
        # Alternative: If we're specifically working with California data, just assume it's Bay Area
        # This is a fallback that might be too broad, but better than nothing
        print("No location column found for filtering, using all data as fallback")
        # Check if we at least know it's US data
        if 'country_code' in df.columns:
            us_filter = df['country_code'] == 'USA'
            us_df = df[us_filter]
            print(f"Filtered to {len(us_df)} US startups as fallback")
            return us_df
        return df

# Function to extract funding data from Growth List
def get_growth_list_data():
    """Extract funding data from Growth List website"""
    print("Fetching funding data from Growth List...")
    
    # URL from search result #1
    url = "https://growthlist.co/san-francisco-startups/"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Based on search result #1, we know there's a table with startup funding data
        tables = soup.find_all('table')
        
        if not tables:
            print("No tables found on Growth List page")
            return None
        
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
        return None

# Function to scrape Y Combinator data - FIXED VERSION
def get_yc_bay_area_data():
    """Extract data from Y Combinator directory pages"""
    print("Fetching startup data from Y Combinator...")
    
    # URLs from search results #3, #6, #10
    urls = [
        "https://www.ycombinator.com/companies/location/san-francisco-bay-area",  # All SF Bay Area
        "https://www.ycombinator.com/companies/industry/api/san-francisco-bay-area",  # API startups
        "https://www.ycombinator.com/companies/industry/operations/san-francisco-bay-area"  # Operations startups
    ]
    
    all_companies = []
    
    for url in urls:
        print(f"Processing: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        }
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Print first 500 characters to debug the page structure
            print(f"Page content preview: {response.text[:500]}...")
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try multiple potential selectors for company elements
            potential_selectors = [
                'div.company-card', 
                'div.company',
                'div[class*="company"]',
                'div.results_companies_item',
                'a.company-card',
                'div.grid-card',
                'div[class*="card"]'
            ]
            
            company_elements = []
            for selector in potential_selectors:
                elements = soup.select(selector)
                if elements:
                    print(f"Found {len(elements)} companies with selector: {selector}")
                    company_elements.extend(elements)
                    break  # Use the first successful selector
            
            if not company_elements:
                # Last resort: try to find any element that seems to contain company info
                print("Trying alternative approach to find company elements...")
                # Look for h3/h4 elements that might be company names
                name_elements = soup.select('h3, h4')
                for name_element in name_elements:
                    # Check if this looks like a company name section
                    if name_element.parent and not name_element.parent.select_one('h2'):  # Avoid headers
                        company_elements.append(name_element.parent)
            
            print(f"Found {len(company_elements)} potential company elements")
            
            for element in company_elements:
                company = {}
                
                # Try to extract company name
                name_element = element.select_one('h3, h4, [class*="name"], a')
                if name_element:
                    company['name'] = name_element.text.strip()
                
                # Try to extract description
                desc_element = element.select_one('p, [class*="description"]')
                if desc_element:
                    company['description'] = desc_element.text.strip()
                
                # Only add companies with a name
                if company.get('name'):
                    company['source_url'] = url
                    all_companies.append(company)
            
            # Be respectful with rate limits
            time.sleep(random.uniform(2, 4))
            
        except Exception as e:
            print(f"Error processing {url}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_companies)
    
    # Remove duplicates based on name
    if not df.empty:
        df = df.drop_duplicates(subset=['name'])
    
    print(f"Successfully extracted {len(df)} companies from Y Combinator")
    return df

# Function to clean and prepare the funding data
def clean_funding_data(github_df, growth_list_df, yc_df):
    """Clean and prepare the funding data for analysis"""
    
    # Process GitHub data
    if github_df is not None:
        # Filter for Bay Area
        github_bay_area = filter_bay_area_startups(github_df)
        github_bay_area['source'] = 'GitHub Repository'
        
        # Convert funding to numeric if available
        if 'funding_total_usd' in github_bay_area.columns:
            # Fixed regex escape sequence
            github_bay_area['funding_total_usd'] = pd.to_numeric(
                github_bay_area['funding_total_usd'].str.replace(r'[\$,]', '', regex=True),
                errors='coerce'
            )
        
        github_bay_area.to_csv('data/github_bay_area_startups.csv', index=False)
        print("Saved GitHub Bay Area startup data")
    else:
        github_bay_area = None
    
    # Process Growth List data
    if growth_list_df is not None:
        # Clean funding amount column
        if 'Funding Amount (USD)' in growth_list_df.columns:
            growth_list_df['Funding Amount (USD)'] = growth_list_df['Funding Amount (USD)'].str.replace(
                r'[^\d.]', '', regex=True)
            growth_list_df['Funding Amount (USD)'] = pd.to_numeric(
                growth_list_df['Funding Amount (USD)'], errors='coerce')
        
        growth_list_df['source'] = 'Growth List'
        growth_list_df.to_csv('data/growth_list_startups.csv', index=False)
        print("Saved Growth List startup data")
    
    # Process Y Combinator data
    if yc_df is not None and not yc_df.empty:
        yc_df['source'] = 'Y Combinator'
        yc_df.to_csv('data/yc_bay_area_startups.csv', index=False)
        print("Saved Y Combinator startup data")
    
    return {
        'github': github_bay_area if 'github_bay_area' in locals() and github_bay_area is not None else None,
        'growth_list': growth_list_df,
        'yc': yc_df
    }

# Function to analyze the funding data
def analyze_funding_data(data_sources):
    """Analyze the funding data and create visualizations"""
    
    # Create a directory for visualizations
    os.makedirs('visualizations', exist_ok=True)
    
    # Check if we have GitHub data with funding information
    if data_sources.get('github') is not None:
        df = data_sources['github']
        
        if 'funding_total_usd' in df.columns and 'market' in df.columns:
            # Funding by market/industry
            plt.figure(figsize=(12, 8))
            market_funding = df.groupby('market')['funding_total_usd'].sum().sort_values(ascending=False).head(10)
            market_funding.plot(kind='barh')
            plt.title('Top 10 Industries by Total Funding (Bay Area Startups)')
            plt.xlabel('Total Funding (USD)')
            plt.tight_layout()
            plt.savefig('visualizations/top_industries_by_funding.png')
            print("Created industry funding visualization")
            
            # Funding by round type
            if 'funding_round_type' in df.columns:
                plt.figure(figsize=(10, 6))
                round_funding = df.groupby('funding_round_type')['funding_total_usd'].sum().sort_values(ascending=False)
                round_funding.plot(kind='bar')
                plt.title('Funding by Round Type (Bay Area Startups)')
                plt.ylabel('Total Funding (USD)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig('visualizations/funding_by_round_type.png')
                print("Created funding round visualization")
    
    # Check if we have Growth List data with funding information
    if data_sources.get('growth_list') is not None:
        df = data_sources['growth_list']
        
        if 'Industry' in df.columns:
            # Industry distribution
            plt.figure(figsize=(12, 8))
            industry_counts = df['Industry'].value_counts().head(10)
            industry_counts.plot(kind='barh')
            plt.title('Top 10 Industries Among Recently Funded Bay Area Startups')
            plt.xlabel('Number of Startups')
            plt.tight_layout()
            plt.savefig('visualizations/top_industries_growth_list.png')
            print("Created Growth List industry visualization")
        
        if 'Funding Type' in df.columns:
            # Funding type distribution
            plt.figure(figsize=(10, 6))
            funding_type_counts = df['Funding Type'].value_counts()
            funding_type_counts.plot(kind='bar')
            plt.title('Distribution of Funding Types Among Bay Area Startups')
            plt.ylabel('Number of Startups')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('visualizations/funding_types_distribution.png')
            print("Created funding type visualization")

# Main function to orchestrate the entire process
def main():
    print("=== Bay Area Startup Funding Data Collection ===")
    
    # Step 1: Collect data from multiple sources
    github_data = get_funding_data_from_github()
    growth_list_data = get_growth_list_data()
    yc_data = get_yc_bay_area_data()
    
    # Step 2: Clean and prepare the data
    data_sources = clean_funding_data(github_data, growth_list_data, yc_data)
    
    # Step 3: Analyze the data
    analyze_funding_data(data_sources)
    
    print("\n=== Data Collection Complete ===")
    print("All data files have been saved to the 'data' directory")
    print("Visualizations have been saved to the 'visualizations' directory")
    
    return data_sources

# Run the script
if __name__ == "__main__":
    collected_data = main()
