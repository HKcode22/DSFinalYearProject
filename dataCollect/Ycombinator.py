# import requests
# from bs4 import BeautifulSoup
# import pandas as pd
# import time
# import re
# import time
# import random

# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# # Function to scrape YC companies
# def scrape_yc_companies(category="", location="san-francisco-bay-area"):
#     url = f"https://www.ycombinator.com/companies/{category}/{location}"
#     # headers = {
#     #     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#     # }

#     headers = {
#     'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
#     'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
#     'Accept-Language': 'en-US,en;q=0.9',
#     'Accept-Encoding': 'gzip, deflate, br',
#     'Connection': 'keep-alive',
#     'Upgrade-Insecure-Requests': '1',
#     'Sec-Fetch-Dest': 'document',
#     'Sec-Fetch-Mode': 'navigate',
#     'Sec-Fetch-Site': 'none',
#     'Sec-Fetch-User': '?1',
#     'Cache-Control': 'max-age=0'
#     } 

    
#     response = requests.get(url, headers=headers)
    
#     if response.status_code != 200:
#         print(f"Failed to retrieve page: {response.status_code}")
#         return pd.DataFrame()
    
#     soup = BeautifulSoup(response.content, 'html.parser')
    
#     # Find company cards - this selector will need to be adjusted based on YC's actual HTML structure
#     company_cards = soup.select('div.company-card')  # Adjust selector as needed
    
#     companies = []
#     for card in company_cards:
#         company = {}
        
#         # Extract company name - adjust selectors based on actual HTML
#         name_element = card.select_one('h4.company-name')
#         if name_element:
#             company['name'] = name_element.text.strip()
        
#         # Extract description
#         desc_element = card.select_one('p.company-description')
#         if desc_element:
#             company['description'] = desc_element.text.strip()
        
#         # Extract other available information (founding year, etc.)
#         # Add more extractors as needed
        
#         companies.append(company)


#         options = Options()
#         options.add_argument("--headless")
#         options.add_argument("--no-sandbox")
#         options.add_argument("--disable-dev-shm-usage")
#         options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")

#         driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
#         driver.get("https://www.ycombinator.com/companies/ai/san-francisco-bay-area")

#         proxies = {
#             'http': 'http://your-proxy-address:port',
#             'https': 'http://your-proxy-address:port'
#         }

#         response = requests.get(url, headers=headers, proxies=proxies)
#         time.sleep(random.uniform(3, 7))
        
#     return pd.DataFrame(companies)

# # Get AI startups in Bay Area
# df_yc_ai = scrape_yc_companies(category="ai")
# print(f"Retrieved {len(df_yc_ai)} YC AI companies")

# # Get Fintech startups in Bay Area
# df_yc_fintech = scrape_yc_companies(category="fintech")
# print(f"Retrieved {len(df_yc_fintech)} YC Fintech companies")

# # Combine datasets
# df_yc = pd.concat([df_yc_ai, df_yc_fintech], ignore_index=True)

# # Save to CSV
# df_yc.to_csv('yc_companies.csv', index=False)
# time.sleep(random.uniform(3, 7))






# # Now you can extract data from driver.page_source

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

def scrape_yc_companies():
    options = Options()
    
    # Critical: DO NOT use headless mode initially
    # options.add_argument("--headless")
    
    # Essential configurations
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    
    # Add realistic user agent
    options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
    
    # Important: Add these to avoid detection
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    # Execute JavaScript to mask WebDriver
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    # Navigate to YC companies page
    driver.get("https://www.ycombinator.com/companies/ai/san-francisco-bay-area")
    
    # Wait for page to load completely
    time.sleep(5)
    
    # Extract company data
    companies = []
    
    try:
        # Wait for company elements to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".company-card, .results_companies_item"))
        )
        
        # Adjust selector based on actual page structure
        # Use multiple potential selectors
        company_elements = driver.find_elements(By.CSS_SELECTOR, ".company-card, .results_companies_item, div[class*='company']")
        
        for element in company_elements:
            company = {}
            
            # Try different potential name selectors
            try:
                name_element = element.find_element(By.CSS_SELECTOR, "h4.company-name, h3, div[class*='name'], a[class*='name']")
                company['name'] = name_element.text.strip()
            except:
                try:
                    name_element = element.find_element(By.TAG_NAME, "h3")
                    company['name'] = name_element.text.strip()
                except:
                    company['name'] = "Name not found"
            
            # Try different potential description selectors
            try:
                desc_element = element.find_element(By.CSS_SELECTOR, "p.company-description, div[class*='description'], p")
                company['description'] = desc_element.text.strip()
            except:
                company['description'] = "Description not found"
            
            companies.append(company)
        
        print(f"Found {len(companies)} companies")
        
    except Exception as e:
        print(f"Error: {e}")
    
    driver.quit()
    
    return pd.DataFrame(companies)

# Get AI startups in Bay Area
df_yc_ai = scrape_yc_companies()

# Save to CSV
if not df_yc_ai.empty:
    df_yc_ai.to_csv('yc_companies.csv', index=False)
    print(f"Saved {len(df_yc_ai)} companies to CSV")
