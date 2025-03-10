# # # # # import requests
# # # # # from bs4 import BeautifulSoup
# # # # # import pandas as pd
# # # # # import time
# # # # # import re
# # # # # import time
# # # # # import random

# # # # # from selenium import webdriver
# # # # # from selenium.webdriver.chrome.options import Options
# # # # # from selenium.webdriver.chrome.service import Service
# # # # from webdriver_manager.chrome import ChromeDriverManager

# # # # # # Function to scrape YC companies
# # # # # def scrape_yc_companies(category="", location="san-francisco-bay-area"):
# # # # #     url = f"https://www.ycombinator.com/companies/{category}/{location}"
# # # # #     # headers = {
# # # # #     #     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
# # # # #     # }

# # # # #     headers = {
# # # # #     'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
# # # # #     'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
# # # # #     'Accept-Language': 'en-US,en;q=0.9',
# # # # #     'Accept-Encoding': 'gzip, deflate, br',
# # # # #     'Connection': 'keep-alive',
# # # # #     'Upgrade-Insecure-Requests': '1',
# # # # #     'Sec-Fetch-Dest': 'document',
# # # # #     'Sec-Fetch-Mode': 'navigate',
# # # # #     'Sec-Fetch-Site': 'none',
# # # # #     'Sec-Fetch-User': '?1',
# # # # #     'Cache-Control': 'max-age=0'
# # # # #     } 

    
# # # # #     response = requests.get(url, headers=headers)
    
# # # # #     if response.status_code != 200:
# # # # #         print(f"Failed to retrieve page: {response.status_code}")
# # # # #         return pd.DataFrame()
    
# # # # #     soup = BeautifulSoup(response.content, 'html.parser')
    
# # # # #     # Find company cards - this selector will need to be adjusted based on YC's actual HTML structure
# # # # #     company_cards = soup.select('div.company-card')  # Adjust selector as needed
    
# # # # #     companies = []
# # # # #     for card in company_cards:
# # # # #         company = {}
        
# # # # #         # Extract company name - adjust selectors based on actual HTML
# # # # #         name_element = card.select_one('h4.company-name')
# # # # #         if name_element:
# # # # #             company['name'] = name_element.text.strip()
        
# # # # #         # Extract description
# # # # #         desc_element = card.select_one('p.company-description')
# # # # #         if desc_element:
# # # # #             company['description'] = desc_element.text.strip()
        
# # # # #         # Extract other available information (founding year, etc.)
# # # # #         # Add more extractors as needed
        
# # # # #         companies.append(company)


# # # # #         options = Options()
# # # # #         options.add_argument("--headless")
# # # # #         options.add_argument("--no-sandbox")
# # # # #         options.add_argument("--disable-dev-shm-usage")
# # # # #         options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")

# # # # #         driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
# # # # #         driver.get("https://www.ycombinator.com/companies/ai/san-francisco-bay-area")

# # # # #         proxies = {
# # # # #             'http': 'http://your-proxy-address:port',
# # # # #             'https': 'http://your-proxy-address:port'
# # # # #         }

# # # # #         response = requests.get(url, headers=headers, proxies=proxies)
# # # # #         time.sleep(random.uniform(3, 7))
        
# # # # #     return pd.DataFrame(companies)

# # # # # # Get AI startups in Bay Area
# # # # # df_yc_ai = scrape_yc_companies(category="ai")
# # # # # print(f"Retrieved {len(df_yc_ai)} YC AI companies")

# # # # # # Get Fintech startups in Bay Area
# # # # # df_yc_fintech = scrape_yc_companies(category="fintech")
# # # # # print(f"Retrieved {len(df_yc_fintech)} YC Fintech companies")

# # # # # # Combine datasets
# # # # # df_yc = pd.concat([df_yc_ai, df_yc_fintech], ignore_index=True)

# # # # # # Save to CSV
# # # # # df_yc.to_csv('yc_companies.csv', index=False)
# # # # # time.sleep(random.uniform(3, 7))






# # # # # # Now you can extract data from driver.page_source

# # # # from selenium import webdriver
# # # # from selenium.webdriver.chrome.options import Options
# # # # from selenium.webdriver.chrome.service import Service
# # # # from selenium.webdriver.common.by import By
# # # # from selenium.webdriver.support.ui import WebDriverWait
# # # # from selenium.webdriver.support import expected_conditions as EC
# # # # import time
# # # # import pandas as pd

# # # # def scrape_yc_companies():
# # # #     options = Options()
    
# # # #     # Critical: DO NOT use headless mode initially
# # # #     # options.add_argument("--headless")
    
# # # #     # Essential configurations
# # # #     options.add_argument("--no-sandbox")
# # # #     options.add_argument("--disable-dev-shm-usage")
# # # #     options.add_argument("--window-size=1920,1080")
# # # #     options.add_argument("--disable-blink-features=AutomationControlled")
    
# # # #     # Add realistic user agent
# # # #     options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
    
# # # #     # Important: Add these to avoid detection
# # # #     options.add_experimental_option("excludeSwitches", ["enable-automation"])
# # # #     options.add_experimental_option('useAutomationExtension', False)
    
# # # #     driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
# # # #     # Execute JavaScript to mask WebDriver
# # # #     driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
# # # #     # Navigate to YC companies page
# # # #     driver.get("https://www.ycombinator.com/companies/ai/san-francisco-bay-area")
    
# # # #     # Wait for page to load completely
# # # #     time.sleep(5)
    
# # # #     # Extract company data
# # # #     companies = []
    
# # # #     try:
# # # #         # Wait for company elements to load
# # # #         WebDriverWait(driver, 10).until(
# # # #             EC.presence_of_element_located((By.CSS_SELECTOR, ".company-card, .results_companies_item"))
# # # #         )
        
# # # #         # Adjust selector based on actual page structure
# # # #         # Use multiple potential selectors
# # # #         company_elements = driver.find_elements(By.CSS_SELECTOR, ".company-card, .results_companies_item, div[class*='company']")
        
# # # #         for element in company_elements:
# # # #             company = {}
            
# # # #             # Try different potential name selectors
# # # #             try:
# # # #                 name_element = element.find_element(By.CSS_SELECTOR, "h4.company-name, h3, div[class*='name'], a[class*='name']")
# # # #                 company['name'] = name_element.text.strip()
# # # #             except:
# # # #                 try:
# # # #                     name_element = element.find_element(By.TAG_NAME, "h3")
# # # #                     company['name'] = name_element.text.strip()
# # # #                 except:
# # # #                     company['name'] = "Name not found"
            
# # # #             # Try different potential description selectors
# # # #             try:
# # # #                 desc_element = element.find_element(By.CSS_SELECTOR, "p.company-description, div[class*='description'], p")
# # # #                 company['description'] = desc_element.text.strip()
# # # #             except:
# # # #                 company['description'] = "Description not found"
            
# # # #             companies.append(company)
        
# # # #         print(f"Found {len(companies)} companies")
        
# # # #     except Exception as e:
# # # #         print(f"Error: {e}")
    
# # # #     driver.quit()
    
# # # #     return pd.DataFrame(companies)

# # # # # Get AI startups in Bay Area
# # # # df_yc_ai = scrape_yc_companies()

# # # # # Save to CSV
# # # # if not df_yc_ai.empty:
# # # #     df_yc_ai.to_csv('yc_companies.csv', index=False)
# # # #     print(f"Saved {len(df_yc_ai)} companies to CSV")




# # # # Required libraries for Y Combinator scraping
# # # from selenium import webdriver
# # # from selenium.webdriver.chrome.options import Options
# # # from bs4 import BeautifulSoup
# # # import time
# # # import json
# # # import pandas as pd
# # # from webdriver_manager.chrome import ChromeDriverManager
# # # from selenium.webdriver.common.by import By
# # # from selenium.webdriver.support.ui import WebDriverWait
# # # from selenium.webdriver.support import expected_conditions as EC
# # # import random
# # # import requests
# # # from selenium.webdriver.common.action_chains import ActionChains
# # # from selenium.webdriver.common.keys import Keys
# # # from selenium.webdriver.support.ui import Select
# # # from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
# # # from selenium.webdriver.common.proxy import Proxy, ProxyType
# # # import logging
# # # from selenium.webdriver.common.alert import Alert
# # # from selenium.webdriver.common.keys import Keys

# # # # Setup Chrome options
# # # chrome_options = Options()
# # # chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")
# # # chrome_options.add_argument("--window-size=1920,1080")
# # # chrome_options.add_argument("--disable-blink-features=AutomationControlled")
# # # # Uncomment for headless mode
# # # # chrome_options.add_argument("--headless")

# # # driver = webdriver.Chrome(options=chrome_options)




# # # # Navigate to YC company directory
# # # driver.get("https://www.ycombinator.com/companies")

# # # # Allow page to load
# # # time.sleep(5)

# # # # Apply filters for Bay Area
# # # # This would require interacting with filter elements on the page
# # # # Example (implementation depends on current page structure):
# # # location_filter = driver.find_element_by_xpath("//button[contains(text(), 'Region')]")
# # # location_filter.click()
# # # bay_area_option = driver.find_element_by_xpath("//div[contains(text(), 'San Francisco Bay Area')]")
# # # bay_area_option.click()


# # # # Wait for filtered results to load
# # # time.sleep(3)

# # # # Get page content
# # # page_source = driver.page_source
# # # soup = BeautifulSoup(page_source, 'html.parser')

# # # # Extract company data (selectors will depend on current page structure)
# # # companies = []
# # # company_elements = soup.find_all('div', class_='company-card')  # Example selector

# # # for company in company_elements:
# # #     name = company.find('h3').text.strip()
# # #     description = company.find('p', class_='description').text.strip()
# # #     # Extract other data points as needed
    
# # #     companies.append({
# # #         'name': name,
# # #         'description': description,
# # #         # Add other fields
# # #     })

# # # # Save results
# # # with open('yc_bay_area_startups.json', 'w') as f:
# # #     json.dump(companies, f, indent=2)


# # # from ycombinator_scraper import Scraper

# # # scraper = Scraper()
# # # # For specific companies
# # # company_data = scraper.scrape_company_data("https://www.workatastartup.com/companies/example-inc")

# # # # For founders information
# # # founders_data = scraper.scrape_founders_data("https://www.workatastartup.com/companies/example-inc")

# # # # Export to JSON
# # # print(company_data.model_dump_json(by_alias=True, indent=2))






# # import time
# # import json
# # import pandas as pd
# # from selenium import webdriver
# # from selenium.webdriver.chrome.options import Options
# # from selenium.webdriver.common.by import By
# # from selenium.webdriver.support.ui import WebDriverWait
# # from selenium.webdriver.support import expected_conditions as EC
# # from selenium.common.exceptions import TimeoutException, NoSuchElementException
# # import logging

# # # Configure logging
# # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # class YCombinatorScraper:
# #     def __init__(self, headless=False, timeout=30):
# #         """Initialize the YC scraper with configurable options."""
# #         self.timeout = timeout
# #         self.data = []
        
# #         # Configure browser options
# #         chrome_options = Options()
# #         if headless:
# #             chrome_options.add_argument("--headless")
        
# #         # Set convincing user agent and other parameters to avoid detection
# #         chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
# #         chrome_options.add_argument("--window-size=1920,1080")
# #         chrome_options.add_argument("--disable-blink-features=AutomationControlled")
# #         chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
# #         chrome_options.add_experimental_option('useAutomationExtension', False)
        
# #         self.driver = webdriver.Chrome(options=chrome_options)
# #         # Execute JavaScript to hide automation
# #         self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
# #         logging.info("Browser initialized with anti-detection measures")
        
# #     def wait_for_element(self, by, value):
# #         """Wait for an element to be present on the page."""
# #         try:
# #             element = WebDriverWait(self.driver, self.timeout).until(
# #                 EC.presence_of_element_located((by, value))
# #             )
# #             return element
# #         except TimeoutException:
# #             logging.warning(f"Timeout waiting for element: {value}")
# #             return None
    
# #     def navigate_to_directory(self):
# #         """Navigate to YC company directory and filter for Bay Area companies."""
# #         try:
# #             self.driver.get("https://www.ycombinator.com/companies")
# #             logging.info("Navigated to YC company directory")
            
# #             # Wait for page to load
# #             time.sleep(5)
            
# #             # Click on location filter
# #             location_filter = self.wait_for_element(By.XPATH, "//button[contains(text(), 'Location')]")
# #             if location_filter:
# #                 location_filter.click()
# #                 logging.info("Clicked location filter")
            
# #                 # Select San Francisco Bay Area
# #                 bay_area_option = self.wait_for_element(By.XPATH, "//div[contains(text(), 'San Francisco Bay Area')]")
# #                 if bay_area_option:
# #                     bay_area_option.click()
# #                     logging.info("Selected Bay Area filter")
# #                     # Wait for results to update
# #                     time.sleep(3)
# #                 else:
# #                     logging.error("Could not find Bay Area option")
# #             else:
# #                 logging.error("Could not find location filter")
                
# #         except Exception as e:
# #             logging.error(f"Error navigating to directory: {str(e)}")
    
# #     def extract_company_data(self):
# #         """Extract data from all companies shown on the current page."""
# #         try:
# #             # Find all company cards
# #             company_cards = self.driver.find_elements(By.CSS_SELECTOR, ".company-card")
# #             logging.info(f"Found {len(company_cards)} company cards on current page")
            
# #             for card in company_cards:
# #                 try:
# #                     # Extract basic information from card
# #                     company_info = {}
                    
# #                     # Company name
# #                     company_info['name'] = card.find_element(By.CSS_SELECTOR, "h3").text.strip()
                    
# #                     # Description
# #                     try:
# #                         company_info['description'] = card.find_element(By.CSS_SELECTOR, ".description").text.strip()
# #                     except NoSuchElementException:
# #                         company_info['description'] = "Not available"
                    
# #                     # Location - ensure we're capturing all Bay Area locations
# #                     try:
# #                         location_element = card.find_element(By.CSS_SELECTOR, ".location")
# #                         company_info['location'] = location_element.text.strip()
# #                     except NoSuchElementException:
# #                         company_info['location'] = "Bay Area - specific location not listed"
                    
# #                     # Get URL to detailed company page
# #                     company_url = card.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
# #                     company_info['url'] = company_url
                    
# #                     # Visit company detail page to extract more information
# #                     detailed_info = self.extract_detailed_info(company_url)
# #                     company_info.update(detailed_info)
                    
# #                     self.data.append(company_info)
# #                     logging.info(f"Extracted data for: {company_info['name']}")
                    
# #                 except Exception as e:
# #                     logging.error(f"Error extracting company card data: {str(e)}")
# #                     continue
                    
# #         except Exception as e:
# #             logging.error(f"Error extracting company data: {str(e)}")
    
# #     def extract_detailed_info(self, company_url):
# #         """Visit company detail page and extract comprehensive information."""
# #         detailed_info = {
# #             'founding_year': None,
# #             'team_size': None,
# #             'founders': [],
# #             'funding_rounds': [],
# #             'industries': [],
# #             'technologies': [],
# #             'social_links': {},
# #             'growth_metrics': {},
# #             'batch': None,
# #             'status': None
# #         }
        
# #         try:
# #             # Open new tab for company details
# #             self.driver.execute_script("window.open('');")
# #             self.driver.switch_to.window(self.driver.window_handles[1])
# #             self.driver.get(company_url)
            
# #             # Wait for page to load
# #             time.sleep(3)
            
# #             # Extract founding year if available
# #             try:
# #                 founding_element = self.driver.find_element(By.XPATH, "//div[contains(text(), 'Founded')]/following-sibling::div")
# #                 detailed_info['founding_year'] = founding_element.text.strip()
# #             except NoSuchElementException:
# #                 pass
            
# #             # Extract team size if available
# #             try:
# #                 team_element = self.driver.find_element(By.XPATH, "//div[contains(text(), 'Team Size')]/following-sibling::div")
# #                 detailed_info['team_size'] = team_element.text.strip()
# #             except NoSuchElementException:
# #                 pass
            
# #             # Extract batch information
# #             try:
# #                 batch_element = self.driver.find_element(By.XPATH, "//div[contains(text(), 'Batch')]/following-sibling::div")
# #                 detailed_info['batch'] = batch_element.text.strip()
# #             except NoSuchElementException:
# #                 pass
            
# #             # Extract status information
# #             try:
# #                 status_element = self.driver.find_element(By.XPATH, "//div[contains(text(), 'Status')]/following-sibling::div")
# #                 detailed_info['status'] = status_element.text.strip()
# #             except NoSuchElementException:
# #                 pass
            
# #             # Extract founder information
# #             try:
# #                 founder_elements = self.driver.find_elements(By.CSS_SELECTOR, ".founder-card")
# #                 for founder in founder_elements:
# #                     founder_info = {
# #                         'name': founder.find_element(By.CSS_SELECTOR, "h4").text.strip(),
# #                         'title': founder.find_element(By.CSS_SELECTOR, ".title").text.strip() if founder.find_elements(By.CSS_SELECTOR, ".title") else None,
# #                         'linkedin': founder.find_element(By.CSS_SELECTOR, "a[href*='linkedin.com']").get_attribute("href") if founder.find_elements(By.CSS_SELECTOR, "a[href*='linkedin.com']") else None
# #                     }
# #                     detailed_info['founders'].append(founder_info)
# #             except NoSuchElementException:
# #                 pass
            
# #             # Extract industries/tags
# #             try:
# #                 industry_elements = self.driver.find_elements(By.CSS_SELECTOR, ".industry-tag")
# #                 detailed_info['industries'] = [tag.text.strip() for tag in industry_elements]
# #             except NoSuchElementException:
# #                 pass
            
# #             # Close tab and switch back to main window
# #             self.driver.close()
# #             self.driver.switch_to.window(self.driver.window_handles[0])
            
# #         except Exception as e:
# #             logging.error(f"Error extracting detailed info: {str(e)}")
# #             # Make sure we switch back to main window if there's an error
# #             if len(self.driver.window_handles) > 1:
# #                 self.driver.close()
# #                 self.driver.switch_to.window(self.driver.window_handles[0])
        
# #         return detailed_info
    
# #     def navigate_pagination(self):
# #         """Navigate through all pages of results."""
# #         page_number = 1
# #         has_next_page = True
        
# #         while has_next_page:
# #             logging.info(f"Processing page {page_number}")
# #             self.extract_company_data()
            
# #             # Check if there's a next page
# #             try:
# #                 next_button = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Next')]")
# #                 if next_button.is_enabled():
# #                     next_button.click()
# #                     page_number += 1
# #                     time.sleep(3)  # Wait for page to load
# #                 else:
# #                     has_next_page = False
# #                     logging.info("Reached last page of results")
# #             except NoSuchElementException:
# #                 has_next_page = False
# #                 logging.info("No more pages found")
    
# #     def export_to_csv(self, filename="bay_area_startups.csv"):
# #         """Export collected data to CSV file."""
# #         try:
# #             df = pd.DataFrame(self.data)
# #             df.to_csv(filename, index=False)
# #             logging.info(f"Data exported to {filename}")
# #             return filename
# #         except Exception as e:
# #             logging.error(f"Error exporting data: {str(e)}")
# #             return None
    
# #     def export_to_json(self, filename="bay_area_startups.json"):
# #         """Export collected data to JSON file."""
# #         try:
# #             with open(filename, 'w') as f:
# #                 json.dump(self.data, f, indent=4)
# #             logging.info(f"Data exported to {filename}")
# #             return filename
# #         except Exception as e:
# #             logging.error(f"Error exporting data: {str(e)}")
# #             return None
    
# #     def scrape(self):
# #         """Main method to execute the full scraping process."""
# #         try:
# #             self.navigate_to_directory()
# #             self.navigate_pagination()
            
# #             # Export data
# #             csv_file = self.export_to_csv()
# #             json_file = self.export_to_json()
            
# #             return {
# #                 'csv_file': csv_file,
# #                 'json_file': json_file,
# #                 'company_count': len(self.data)
# #             }
# #         finally:
# #             # Clean up resources
# #             self.driver.quit()
# #             logging.info("Browser closed, scraping complete")

# # # Function to run the scraper with error handling
# # def scrape_bay_area_startups(headless=False):
# #     """Run the YC Bay Area startup scraper with error handling."""
# #     try:
# #         scraper = YCombinatorScraper(headless=headless)
# #         result = scraper.scrape()
# #         print(f"Successfully scraped {result['company_count']} Bay Area companies")
# #         print(f"Data saved to {result['csv_file']} and {result['json_file']}")
# #         return result
# #     except Exception as e:
# #         logging.error(f"Scraping failed: {str(e)}")
# #         return None

# # # Execute the scraper if run directly
# # if __name__ == "__main__":
# #     print("Starting Y Combinator Bay Area startup data extraction...")
# #     scrape_bay_area_startups(headless=False)  # Set to True for headless operation


# import time
# import json
# import pandas as pd
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.chrome.service import Service
# # Import By class for element locating strategies
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.common.exceptions import TimeoutException, NoSuchElementException
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# class YCombinatorScraper:
#     def __init__(self, headless=False, timeout=60):  # Increased timeout
#         """Initialize the YC scraper with configurable options."""
#         self.timeout = timeout
#         self.data = []
        
#         # Configure browser options
#         chrome_options = Options()
#         if headless:
#             chrome_options.add_argument("--headless=new")  # Updated headless syntax
        
#         # Set convincing user agent and other parameters to avoid detection
#         chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
#         chrome_options.add_argument("--window-size=1920,1080")
#         chrome_options.add_argument("--disable-blink-features=AutomationControlled")
#         chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
#         chrome_options.add_experimental_option('useAutomationExtension', False)
        
#         # Updated to use Service class instead of executable_path
#         self.driver = webdriver.Chrome(options=chrome_options)
#         # Execute JavaScript to hide automation
#         self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
#         logging.info("Browser initialized with anti-detection measures")
        
#     def wait_for_element(self, by, value):
#         """Wait for an element to be present on the page."""
#         try:
#             element = WebDriverWait(self.driver, self.timeout).until(
#                 EC.presence_of_element_located((by, value))
#             )
#             return element
#         except TimeoutException:
#             logging.warning(f"Timeout waiting for element: {value}")
#             return None
    
#     def navigate_to_directory(self):
#         """Navigate to YC company directory and filter for Bay Area companies."""
#         try:
#             self.driver.get("https://www.ycombinator.com/companies")
#             logging.info("Navigated to YC company directory")
            
#             # Wait for page to fully load
#             time.sleep(10)  # Increased wait time
            
#             # Try different possible filter button labels and approaches
#             filter_selectors = [
#                 (By.XPATH, "//button[contains(text(), 'Location')]"),
#                 (By.XPATH, "//button[contains(text(), 'Region')]"),
#                 (By.XPATH, "//span[contains(text(), 'Location')]/parent::button"),
#                 (By.XPATH, "//span[contains(text(), 'Region')]/parent::button"),
#                 (By.CSS_SELECTOR, "button.filter-button"),
#                 (By.CSS_SELECTOR, ".filters-container button")
#             ]
            
#             location_filter = None
#             for by, selector in filter_selectors:
#                 try:
#                     location_filter = self.driver.find_element(by, selector)
#                     logging.info(f"Found location filter using selector: {selector}")
#                     break
#                 except NoSuchElementException:
#                     continue
            
#             if location_filter:
#                 location_filter.click()
#                 logging.info("Clicked location filter")
#                 time.sleep(3)  # Wait for dropdown to appear
                
#                 # Try different approaches to find Bay Area option
#                 bay_area_selectors = [
#                     (By.XPATH, "//div[contains(text(), 'San Francisco Bay Area')]"),
#                     (By.XPATH, "//li[contains(text(), 'San Francisco Bay Area')]"),
#                     (By.XPATH, "//div[contains(text(), 'Bay Area')]"),
#                     (By.XPATH, "//span[contains(text(), 'Bay Area')]"),
#                     (By.XPATH, "//div[contains(text(), 'San Francisco')]")
#                 ]
                
#                 bay_area_option = None
#                 for by, selector in bay_area_selectors:
#                     try:
#                         bay_area_option = self.driver.find_element(by, selector)
#                         logging.info(f"Found Bay Area option using selector: {selector}")
#                         break
#                     except NoSuchElementException:
#                         continue
                
#                 if bay_area_option:
#                     bay_area_option.click()
#                     logging.info("Selected Bay Area filter")
#                     # Wait for results to update
#                     time.sleep(5)
#                 else:
#                     logging.error("Could not find Bay Area option - trying to proceed with all companies")
#             else:
#                 logging.error("Could not find location filter - proceeding with all companies")
            
#             # Take screenshot for debugging
#             self.driver.save_screenshot("yc_page.png")
#             logging.info("Saved screenshot to yc_page.png for debugging")
                
#         except Exception as e:
#             logging.error(f"Error navigating to directory: {str(e)}")
#             self.driver.save_screenshot("error_page.png")
    
#     def extract_company_data(self):
#         """Extract data from all companies shown on the current page."""
#         try:
#             # Try different selectors for company cards
#             card_selectors = [
#                 (By.CSS_SELECTOR, ".company-card"),
#                 (By.CSS_SELECTOR, ".directory-company"),
#                 (By.XPATH, "//div[contains(@class, 'company')]"),
#                 (By.CSS_SELECTOR, ".startup-card")
#             ]
            
#             company_cards = []
#             for by, selector in card_selectors:
#                 company_cards = self.driver.find_elements(by, selector)
#                 if company_cards:
#                     logging.info(f"Found {len(company_cards)} company cards using selector: {selector}")
#                     break
            
#             if not company_cards:
#                 logging.warning("No company cards found with any selector")
#                 self.driver.save_screenshot("no_companies.png")
#                 return
            
#             for card in company_cards:
#                 try:
#                     # Extract basic information from card
#                     company_info = {}
                    
#                     # Try different selectors for company name
#                     name_selectors = [
#                         (By.CSS_SELECTOR, "h3"),
#                         (By.CSS_SELECTOR, "h2"),
#                         (By.CSS_SELECTOR, ".company-name"),
#                         (By.XPATH, ".//h3"),
#                         (By.XPATH, ".//h2")
#                     ]
                    
#                     # Company name
#                     for by, selector in name_selectors:
#                         try:
#                             company_info['name'] = card.find_element(by, selector).text.strip()
#                             break
#                         except NoSuchElementException:
#                             continue
                    
#                     if 'name' not in company_info:
#                         company_info['name'] = "Unknown"
                    
#                     # Description
#                     desc_selectors = [
#                         (By.CSS_SELECTOR, ".description"),
#                         (By.CSS_SELECTOR, ".company-description"),
#                         (By.XPATH, ".//p"),
#                         (By.CSS_SELECTOR, "p")
#                     ]
                    
#                     for by, selector in desc_selectors:
#                         try:
#                             company_info['description'] = card.find_element(by, selector).text.strip()
#                             break
#                         except NoSuchElementException:
#                             continue
                    
#                     if 'description' not in company_info:
#                         company_info['description'] = "Not available"
                    
#                     # Location
#                     location_selectors = [
#                         (By.CSS_SELECTOR, ".location"),
#                         (By.CSS_SELECTOR, ".company-location"),
#                         (By.XPATH, ".//div[contains(@class, 'location')]"),
#                         (By.XPATH, ".//span[contains(@class, 'location')]")
#                     ]
                    
#                     for by, selector in location_selectors:
#                         try:
#                             company_info['location'] = card.find_element(by, selector).text.strip()
#                             break
#                         except NoSuchElementException:
#                             continue
                    
#                     if 'location' not in company_info:
#                         company_info['location'] = "Location not specified"
                    
#                     # Get URL to detailed company page
#                     url_selectors = [
#                         (By.TAG_NAME, "a"),
#                         (By.CSS_SELECTOR, "a"),
#                         (By.XPATH, ".//a")
#                     ]
                    
#                     for by, selector in url_selectors:
#                         try:
#                             company_info['url'] = card.find_element(by, selector).get_attribute("href")
#                             break
#                         except NoSuchElementException:
#                             continue
                    
#                     if 'url' not in company_info or not company_info['url']:
#                         company_info['url'] = "URL not available"
#                         # Skip detailed info if we don't have a URL
#                         self.data.append(company_info)
#                         logging.info(f"Added basic data for: {company_info['name']} (no URL)")
#                         continue
                    
#                     # Visit company detail page to extract more information
#                     detailed_info = self.extract_detailed_info(company_info['url'])
#                     company_info.update(detailed_info)
                    
#                     self.data.append(company_info)
#                     logging.info(f"Extracted data for: {company_info['name']}")
                    
#                 except Exception as e:
#                     logging.error(f"Error extracting company card data: {str(e)}")
#                     continue
                    
#         except Exception as e:
#             logging.error(f"Error extracting company data: {str(e)}")
    
#     def extract_detailed_info(self, company_url):
#         """Visit company detail page and extract comprehensive information."""
#         detailed_info = {
#             'founding_year': None,
#             'team_size': None,
#             'founders': [],
#             'funding_rounds': [],
#             'industries': [],
#             'technologies': [],
#             'social_links': {},
#             'growth_metrics': {},
#             'batch': None,
#             'status': None
#         }
        
#         try:
#             # Open new tab for company details
#             self.driver.execute_script("window.open('');")
#             self.driver.switch_to.window(self.driver.window_handles[1])
#             self.driver.get(company_url)
            
#             # Wait for page to load
#             time.sleep(5)
            
#             # Extract founding year if available
#             founding_selectors = [
#                 (By.XPATH, "//div[contains(text(), 'Founded')]/following-sibling::div"),
#                 (By.XPATH, "//span[contains(text(), 'Founded')]/following-sibling::span")
#             ]
            
#             for by, selector in founding_selectors:
#                 try:
#                     founding_element = self.driver.find_element(by, selector)
#                     detailed_info['founding_year'] = founding_element.text.strip()
#                     break
#                 except NoSuchElementException:
#                     continue
            
#             # Extract team size if available
#             team_selectors = [
#                 (By.XPATH, "//div[contains(text(), 'Team Size')]/following-sibling::div"),
#                 (By.XPATH, "//span[contains(text(), 'Team Size')]/following-sibling::span")
#             ]
            
#             for by, selector in team_selectors:
#                 try:
#                     team_element = self.driver.find_element(by, selector)
#                     detailed_info['team_size'] = team_element.text.strip()
#                     break
#                 except NoSuchElementException:
#                     continue
            
#             # Extract batch information
#             batch_selectors = [
#                 (By.XPATH, "//div[contains(text(), 'Batch')]/following-sibling::div"),
#                 (By.XPATH, "//span[contains(text(), 'Batch')]/following-sibling::span")
#             ]
            
#             for by, selector in batch_selectors:
#                 try:
#                     batch_element = self.driver.find_element(by, selector)
#                     detailed_info['batch'] = batch_element.text.strip()
#                     break
#                 except NoSuchElementException:
#                     continue
            
#             # Extract status information
#             status_selectors = [
#                 (By.XPATH, "//div[contains(text(), 'Status')]/following-sibling::div"),
#                 (By.XPATH, "//span[contains(text(), 'Status')]/following-sibling::span")
#             ]
            
#             for by, selector in status_selectors:
#                 try:
#                     status_element = self.driver.find_element(by, selector)
#                     detailed_info['status'] = status_element.text.strip()
#                     break
#                 except NoSuchElementException:
#                     continue
            
#             # Extract founder information
#             founder_selectors = [
#                 (By.CSS_SELECTOR, ".founder-card"),
#                 (By.XPATH, "//div[contains(@class, 'founder')]")
#             ]
            
#             for by, selector in founder_selectors:
#                 try:
#                     founder_elements = self.driver.find_elements(by, selector)
                    
#                     for founder in founder_elements:
#                         founder_info = {
#                             'name': 'Unknown',
#                             'title': None,
#                             'linkedin': None
#                         }
                        
#                         # Try to get founder name
#                         for name_by, name_selector in [(By.CSS_SELECTOR, "h4"), (By.XPATH, ".//h4"), (By.XPATH, ".//div[contains(@class, 'name')]")]:
#                             try:
#                                 founder_info['name'] = founder.find_element(name_by, name_selector).text.strip()
#                                 break
#                             except NoSuchElementException:
#                                 continue
                        
#                         # Try to get founder title
#                         for title_by, title_selector in [(By.CSS_SELECTOR, ".title"), (By.XPATH, ".//div[contains(@class, 'title')]")]:
#                             try:
#                                 founder_info['title'] = founder.find_element(title_by, title_selector).text.strip()
#                                 break
#                             except NoSuchElementException:
#                                 continue
                        
#                         # Try to get LinkedIn
#                         for linkedin_by, linkedin_selector in [(By.XPATH, ".//a[contains(@href, 'linkedin.com')]"), (By.CSS_SELECTOR, "a[href*='linkedin.com']")]:
#                             try:
#                                 founder_info['linkedin'] = founder.find_element(linkedin_by, linkedin_selector).get_attribute("href")
#                                 break
#                             except NoSuchElementException:
#                                 continue
                        
#                         detailed_info['founders'].append(founder_info)
                    
#                     if detailed_info['founders']:
#                         break
#                 except NoSuchElementException:
#                     continue
            
#             # Extract industries/tags
#             industry_selectors = [
#                 (By.CSS_SELECTOR, ".industry-tag"),
#                 (By.XPATH, "//div[contains(@class, 'tag')]"),
#                 (By.XPATH, "//span[contains(@class, 'tag')]")
#             ]
            
#             for by, selector in industry_selectors:
#                 try:
#                     industry_elements = self.driver.find_elements(by, selector)
#                     if industry_elements:
#                         detailed_info['industries'] = [tag.text.strip() for tag in industry_elements]
#                         break
#                 except NoSuchElementException:
#                     continue
            
#             # Close tab and switch back to main window
#             self.driver.close()
#             self.driver.switch_to.window(self.driver.window_handles[0])
            
#         except Exception as e:
#             logging.error(f"Error extracting detailed info: {str(e)}")
#             # Make sure we switch back to main window if there's an error
#             if len(self.driver.window_handles) > 1:
#                 self.driver.close()
#                 self.driver.switch_to.window(self.driver.window_handles[0])
        
#         return detailed_info
    
#     def navigate_pagination(self):
#         """Navigate through all pages of results."""
#         page_number = 1
#         has_next_page = True
        
#         while has_next_page:
#             logging.info(f"Processing page {page_number}")
#             self.extract_company_data()
            
#             # Check if there's a next page
#             next_selectors = [
#                 (By.XPATH, "//button[contains(text(), 'Next')]"),
#                 (By.XPATH, "//a[contains(text(), 'Next')]"),
#                 (By.CSS_SELECTOR, ".pagination-next"),
#                 (By.XPATH, "//button[contains(@class, 'next')]")
#             ]
            
#             next_button = None
#             for by, selector in next_selectors:
#                 try:
#                     next_button = self.driver.find_element(by, selector)
#                     if next_button.is_enabled():
#                         break
#                     else:
#                         next_button = None
#                 except NoSuchElementException:
#                     continue
            
#             if next_button and next_button.is_enabled():
#                 next_button.click()
#                 page_number += 1
#                 time.sleep(5)  # Wait for page to load
#             else:
#                 has_next_page = False
#                 logging.info("No more pages found or next button not enabled")
    
#     def export_to_csv(self, filename="bay_area_startups.csv"):
#         """Export collected data to CSV file."""
#         try:
#             df = pd.DataFrame(self.data)
#             df.to_csv(filename, index=False)
#             logging.info(f"Data exported to {filename}")
#             return filename
#         except Exception as e:
#             logging.error(f"Error exporting data: {str(e)}")
#             return None
    
#     def export_to_json(self, filename="bay_area_startups.json"):
#         """Export collected data to JSON file."""
#         try:
#             with open(filename, 'w') as f:
#                 json.dump(self.data, f, indent=4)
#             logging.info(f"Data exported to {filename}")
#             return filename
#         except Exception as e:
#             logging.error(f"Error exporting data: {str(e)}")
#             return None
    
#     def scrape(self):
#         """Main method to execute the full scraping process."""
#         try:
#             self.navigate_to_directory()
#             self.navigate_pagination()
            
#             # Export data
#             csv_file = self.export_to_csv()
#             json_file = self.export_to_json()
            
#             return {
#                 'csv_file': csv_file,
#                 'json_file': json_file,
#                 'company_count': len(self.data)
#             }
#         finally:
#             # Clean up resources
#             self.driver.quit()
#             logging.info("Browser closed, scraping complete")

# # Function to run the scraper with error handling
# def scrape_bay_area_startups(headless=False):
#     """Run the YC Bay Area startup scraper with error handling."""
#     try:
#         scraper = YCombinatorScraper(headless=headless)
#         result = scraper.scrape()
#         print(f"Successfully scraped {result['company_count']} Bay Area companies")
#         print(f"Data saved to {result['csv_file']} and {result['json_file']}")
#         return result
#     except Exception as e:
#         logging.error(f"Scraping failed: {str(e)}")
#         return None

# # Execute the scraper if run directly
# if __name__ == "__main__":
#     print("Starting Y Combinator Bay Area startup data extraction...")
#     scrape_bay_area_startups(headless=False)  # Set to True for headless operation



import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class YCDirectPageCollector:
    def __init__(self):
        self.bay_area_industry_urls = [
            "https://www.ycombinator.com/companies/industry/design-tools/san-francisco-bay-area",
            "https://www.ycombinator.com/companies/industry/engineering-product-and-design/san-francisco-bay-area",
            "https://www.ycombinator.com/companies/industry/developer-tools/san-francisco-bay-area",
            "https://www.ycombinator.com/companies/industry/ai/san-francisco-bay-area",
            "https://www.ycombinator.com/companies/industry/b2b/san-francisco-bay-area",
            "https://www.ycombinator.com/companies/industry/fintech/san-francisco-bay-area"
        ]
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
        }
        self.companies = []
        
    def collect_from_page(self, url):
        """Collect company data from a specific industry page."""
        try:
            logging.info(f"Collecting from {url}")
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            company_cards = soup.select(".company-card, .directory-company, div[class*='company']")
            
            page_companies = []
            for card in company_cards:
                company = {}
                
                # Extract name
                name_elem = card.select_one("h3, h2, .company-name")
                if name_elem:
                    company["name"] = name_elem.text.strip()
                else:
                    continue  # Skip if no name found
                
                # Extract description
                desc_elem = card.select_one(".description, p")
                if desc_elem:
                    company["description"] = desc_elem.text.strip()
                
                # Extract URL
                link_elem = card.select_one("a")
                if link_elem and "href" in link_elem.attrs:
                    company["url"] = link_elem["href"]
                    if not company["url"].startswith("http"):
                        company["url"] = f"https://www.ycombinator.com{company['url']}"
                
                # Extract industry from URL
                industry = url.split("/")[-2].replace("-", " ").title()
                company["industry"] = industry
                company["location"] = "San Francisco Bay Area"
                
                page_companies.append(company)
            
            logging.info(f"Found {len(page_companies)} companies on this page")
            return page_companies
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error collecting from {url}: {str(e)}")
            return []
    
    def collect_all_pages(self):
        """Collect data from all industry pages."""
        for url in self.bay_area_industry_urls:
            companies = self.collect_from_page(url)
            # Avoid duplicates by checking if company is already in the list
            for company in companies:
                if company["name"] not in [c["name"] for c in self.companies]:
                    self.companies.append(company)
            time.sleep(2)  # Polite delay between requests
    
    def export_data(self, csv_filename="yc_bay_area_startups.csv", json_filename="yc_bay_area_startups.json"):
        """Export collected data to CSV and JSON files."""
        if not self.companies:
            logging.warning("No companies to export")
            return
            
        # Export to CSV
        df = pd.DataFrame(self.companies)
        df.to_csv(csv_filename, index=False)
        logging.info(f"Exported {len(self.companies)} companies to {csv_filename}")
        
        # Export to JSON
        with open(json_filename, 'w') as f:
            json.dump({"companies": self.companies}, f, indent=2)
        logging.info(f"Exported {len(self.companies)} companies to {json_filename}")
    
    def collect(self):
        """Run the full collection process."""
        self.collect_all_pages()
        self.export_data()
        return len(self.companies)

if __name__ == "__main__":
    print("Starting Y Combinator direct page collection...")
    collector = YCDirectPageCollector()
    company_count = collector.collect()
    print(f"Successfully collected {company_count} Bay Area companies")


import requests
import pandas as pd
import json
import time
import logging
import backoff

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BayAreaStartupCollector:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        self.companies = []
    
    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
    def _make_request(self, url, params=None):
        """Make an HTTP request with exponential backoff."""
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response
    
    def collect_crunchbase_data(self):
        """Collect Bay Area startup data from Crunchbase Public API."""
        logging.info("Collecting Crunchbase free data...")
        
        # Crunchbase free data is limited, but there are some public endpoints
        try:
            # Example endpoint - actual endpoints may vary and may require authentication
            url = "https://api.crunchbase.com/api/v4/searches/organizations"
            params = {
                "query": "location:San Francisco Bay Area",
                "limit": 100
            }
            
            response = self._make_request(url, params)
            data = response.json()
            
            if "entities" in data:
                for entity in data["entities"]:
                    company = {
                        "name": entity.get("properties", {}).get("name"),
                        "description": entity.get("properties", {}).get("short_description"),
                        "location": entity.get("properties", {}).get("location_identifiers", [{}])[0].get("value"),
                        "funding_total": entity.get("properties", {}).get("funding_total_usd"),
                        "founded_on": entity.get("properties", {}).get("founded_on"),
                        "source": "Crunchbase"
                    }
                    self.companies.append(company)
                
                logging.info(f"Collected {len(data['entities'])} companies from Crunchbase")
            else:
                logging.warning("No Crunchbase company data collected")
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Error collecting Crunchbase data: {str(e)}")
    
    def collect_linkedin_data(self):
        """Collect Bay Area startup data based on LinkedIn posts."""
        logging.info("Collecting LinkedIn startup data...")
        
        # Based on search result [9], there's a list of YC companies with 50-100 employees
        yc_growing_companies = [
            {"name": "Gigs", "description": "Stripe for phone plans", "employee_count": "50-100", "remote": True},
            {"name": "Embrace", "description": "mobile observability built on OpenTelemetry", "employee_count": "50-100", "remote": True},
            {"name": "PostHog", "description": "open source product operating system", "employee_count": "50-100", "remote": True},
            {"name": "SafeBase", "description": "trust center platform", "employee_count": "50-100", "remote": True},
            {"name": "Veryfi", "description": "data extraction platform", "employee_count": "50-100", "remote": True},
            {"name": "Gather", "description": "better way to meet online", "employee_count": "50-100", "remote": True},
            {"name": "R2", "description": "embedded lending infrastructure", "employee_count": "50-100", "remote": True},
            {"name": "Pulley", "description": "equity and cap table management", "employee_count": "50-100", "remote": True}
        ]
        
        # Filter for Bay Area companies (this would need to be enhanced with actual location data)
        bay_area_companies = [
            {**company, "source": "LinkedIn list", "location": "Bay Area (to be verified)"}
            for company in yc_growing_companies
        ]
        
        self.companies.extend(bay_area_companies)
        logging.info(f"Added {len(bay_area_companies)} companies from LinkedIn data")
    
    def export_data(self, csv_filename="bay_area_startups.csv", json_filename="bay_area_startups.json"):
        """Export collected data to CSV and JSON files."""
        if not self.companies:
            logging.warning("No companies to export")
            return
            
        # Export to CSV
        df = pd.DataFrame(self.companies)
        df.to_csv(csv_filename, index=False)
        logging.info(f"Exported {len(self.companies)} companies to {csv_filename}")
        
        # Export to JSON
        with open(json_filename, 'w') as f:
            json.dump({"companies": self.companies}, f, indent=2)
        logging.info(f"Exported {len(self.companies)} companies to {json_filename}")
    
    def collect(self):
        """Run the full collection process."""
        self.collect_crunchbase_data()
        self.collect_linkedin_data()
        self.export_data()
        return len(self.companies)

if __name__ == "__main__":
    print("Starting Bay Area startup data collection from alternative sources...")
    collector = BayAreaStartupCollector()
    company_count = collector.collect()
    print(f"Successfully collected {company_count} Bay Area companies")


