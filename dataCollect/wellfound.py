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

