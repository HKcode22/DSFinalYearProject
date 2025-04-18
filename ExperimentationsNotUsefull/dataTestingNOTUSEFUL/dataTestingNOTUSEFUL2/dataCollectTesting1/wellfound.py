import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_wellfound_startups(location="san-francisco-bay-area", pages=3):
    base_url = f"https://wellfound.com/startups/location/{location}"
    
    all_startups = []
    
    for page in range(1, pages+1):
        url = f"{base_url}?page={page}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"Failed to retrieve page {page}: {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find startup cards - adjust selector based on actual HTML
            startup_cards = soup.select('div.startup-card')  # Adjust selector as needed
            
            for card in startup_cards:
                startup = {}
                
                # Extract startup name
                name_element = card.select_one('h4.startup-name')
                if name_element:
                    startup['name'] = name_element.text.strip()
                
                # Extract other information
                # Adjust selectors based on actual HTML structure
                
                all_startups.append(startup)
                
            # Be respectful with rate limiting
            time.sleep(2)
            
        except Exception as e:
            print(f"Error on page {page}: {str(e)}")
    
    return pd.DataFrame(all_startups)

# Scrape Wellfound startups in Bay Area
df_wellfound = scrape_wellfound_startups(pages=5)
print(f"Retrieved {len(df_wellfound)} Wellfound startups")

# Save to CSV
df_wellfound.to_csv('wellfound_startups.csv', index=False)
print("Saved Wellfound startups to wellfound_startups.csv")

"""
Ok based on the previous steps, i have had given you 25 files which you have stored and recall, look at ur responses before, ive given you 25 files direct files csv look at the previouis responses, based on that

ok lets merge bay_area_startups_base.csv and Bay-Area-Companies-List.csv into one, recall its exact collumns and rows and merge it into one

then lets merge companies.csv, companies.csv (579 companies)
companies-that-use-laravel.csv (41 companies)
silicon-valley-companies.csv (200+ companies)
venture-capital.csv (60 VC firms)
y-combinator-companies.csv (65 YC companies)
→ Add a "source_category" column to indicate origin (Laravel users, YC companies, etc.)
into one

then lets merge crunchbase_alternative_data.csv (175+ rows, 21 columns)
crunchbase_companies.csv (250+ rows, 14 columns)
bay_area_startups_crunchbase_dataset.csv (100+ rows, 30 columns)
→ Keep all unique columns, mapping overlapping fields
into one

then lets merge, tech-companies-in-oakland-06-20-2021.csv (100+ rows)
san-francisco-tech-companies-06-30-2021.csv (50+ rows)
→ Add a "city" column to preserve location distinction
datasf_businesses_raw.csv (150+ rows)
sf_business_dataset.csv (150+ rows)
datasf_tech_businesses.csv (367 rows)
sf_tech_businesses1.csv (200+ rows)
→ Add a "business_type" column to distinguish tech vs. non-tech companies

then lets merge,
github_bay_area_startups.csv (unique funding round structure)
growth_list_startups.csv (2025 funding data)
high_potential_startups.csv (growth potential scoring)
business_register_sample.csv (Chinatown-specific business data)
recent_building_permits.csv (construction permit data)
bay_area_startups_combined_Links.csv - Extract just website URLs to merge with other company datasets
news_mentioned_companies.csv - Either keep separate (just 6 rows) or merge into a main company database
bay_area_startups_master.csv - If most fields are empty, extract only the populated fields to merge with other datasets
into one

please remember and recall all of the rows and columns for each files and correctly carefully merge them, and show me the python code
"""
