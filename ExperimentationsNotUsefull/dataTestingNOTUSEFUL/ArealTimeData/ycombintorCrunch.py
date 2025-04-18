# import requests
# from bs4 import BeautifulSoup
# import json
# import time

# def scrape_yc_companies(batch="W24"):
#     """
#     Scrape YCombinator companies from a specific batch.
    
#     Args:
#         batch (str): The batch code (e.g., "W24" for Winter 2024)
    
#     Returns:
#         list: A list of dictionaries containing company data
#     """
#     url = f"https://www.ycombinator.com/companies?batch={batch}"
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
#     }
    
#     response = requests.get(url, headers=headers)
    
#     if response.status_code != 200:
#         print(f"Failed to fetch data: {response.status_code}")
#         return []
    
#     soup = BeautifulSoup(response.text, 'html.parser')
#     companies = []
    
#     # Find company cards - adjust the selector based on actual HTML structure
#     company_cards = soup.select('div.CompanyCard_root__hK85u')
    
#     for card in company_cards:
#         name_element = card.select_one('h3.CompanyCard_name__eAkro')
#         desc_element = card.select_one('div.CompanyCard_tagline__MZrn7')
#         url_element = card.select_one('a.CompanyCard_companyLink__L3crw')
        
#         name = name_element.text.strip() if name_element else "Unknown"
#         description = desc_element.text.strip() if desc_element else "No description"
#         company_url = url_element['href'] if url_element and 'href' in url_element.attrs else None
        
#         company_data = {
#             'name': name,
#             'description': description,
#             'url': f"https://www.ycombinator.com{company_url}" if company_url else None,
#             'batch': batch
#         }
        
#         companies.append(company_data)
        
#     return companies

# if __name__ == "__main__":
#     # Scrape multiple batches
#     all_companies = []
#     batches = ["W24", "S23", "W23"]
    
#     for batch in batches:
#         print(f"Scraping batch {batch}...")
#         companies = scrape_yc_companies(batch)
#         all_companies.extend(companies)
#         print(f"Found {len(companies)} companies in batch {batch}")
#         time.sleep(2)  # Be respectful with rate limiting
    
#     # Save to JSON file
#     with open('yc_companies.json', 'w') as f:
#         json.dump(all_companies, f, indent=2)
    
#     print(f"Total companies scraped: {len(all_companies)}")
#     print("Data saved to yc_companies.json")


# # Install the package
# # pip install ycombinator-scraper

# from ycombinator_scraper import YComboScraper

# # Initialize scraper
# scraper = YComboScraper()

# # Get jobs from specific companies
# jobs = scraper.get_jobs(companies=["openai", "anthropic"])

# # Export to CSV
# scraper.export_to_csv(jobs, "yc_jobs.csv")


# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from webdriver_manager.chrome import ChromeDriverManager
# import time
# import json
# import random

# def scrape_crunchbase_company(company_slug):
#     """
#     Scrape company data from Crunchbase using Selenium.
    
#     Args:
#         company_slug (str): The company identifier in Crunchbase URL
        
#     Returns:
#         dict: Company information
#     """
#     # Set up Chrome options
#     chrome_options = Options()
#     # Uncomment if you want to run in headless mode
#     # chrome_options.add_argument("--headless")
#     chrome_options.add_argument("--window-size=1920,1080")
#     chrome_options.add_argument("--no-sandbox")
#     chrome_options.add_argument("--disable-dev-shm-usage")
#     # Use a realistic user agent
#     chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")
    
#     # Initialize the WebDriver
#     driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    
#     company_data = {}
    
#     try:
#         # Navigate to the company page
#         url = f"https://www.crunchbase.com/organization/{company_slug}"
#         driver.get(url)
        
#         # Handle possible Cloudflare or other challenges
#         # Sometimes just waiting helps bypass simple checks
#         time.sleep(random.uniform(5, 10))
        
#         # Wait for the content to load
#         WebDriverWait(driver, 30).until(
#             EC.presence_of_element_located((By.CSS_SELECTOR, "h1.profile-name"))
#         )
        
#         # Extract basic company information
#         company_data['name'] = driver.find_element(By.CSS_SELECTOR, "h1.profile-name").text.strip()
        
#         # Extract description
#         try:
#             description = driver.find_element(By.CSS_SELECTOR, "div.description").text.strip()
#             company_data['description'] = description
#         except:
#             company_data['description'] = "Not available"
        
#         # Extract website
#         try:
#             website = driver.find_element(By.CSS_SELECTOR, "a.website-link").get_attribute("href")
#             company_data['website'] = website
#         except:
#             company_data['website'] = "Not available"
        
#         # Extract funding information
#         try:
#             funding_elements = driver.find_elements(By.CSS_SELECTOR, "div.funding-rounds")
#             funding_data = []
#             for element in funding_elements:
#                 funding_data.append(element.text.strip())
#             company_data['funding'] = funding_data
#         except:
#             company_data['funding'] = []
        
#         # Extract team members
#         try:
#             team_elements = driver.find_elements(By.CSS_SELECTOR, "div.team-members")
#             team_data = []
#             for element in team_elements:
#                 team_data.append(element.text.strip())
#             company_data['team'] = team_data
#         except:
#             company_data['team'] = []
            
#         return company_data
    
#     except Exception as e:
#         print(f"Error scraping {company_slug}: {str(e)}")
#         return {"error": str(e)}
    
#     finally:
#         driver.quit()

# def scrape_multiple_companies(company_slugs):
#     """
#     Scrape multiple companies with delay between requests.
    
#     Args:
#         company_slugs (list): List of company slugs to scrape
        
#     Returns:
#         dict: Dictionary with company data
#     """
#     results = {}
    
#     for slug in company_slugs:
#         print(f"Scraping {slug}...")
#         results[slug] = scrape_crunchbase_company(slug)
        
#         # Random delay to avoid being blocked
#         delay = random.uniform(30, 60)
#         print(f"Waiting {delay:.2f} seconds before next request...")
#         time.sleep(delay)
    
#     return results

# if __name__ == "__main__":
#     # Example company slugs
#     companies = ["openai", "anthropic", "stripe"]
    
#     results = scrape_multiple_companies(companies)
    
#     # Save results to JSON
#     with open('crunchbase_data.json', 'w') as f:
#         json.dump(results, f, indent=2)
    
#     print("Scraping completed. Data saved to crunchbase_data.json")


# # Install playwright
# # pip install playwright
# # playwright install

# from playwright.sync_api import sync_playwright
# import time
# import json
# import random

# def scrape_with_playwright(url, company_name):
#     """
#     Scrape website using Playwright for better JS handling and bot detection evasion.
    
#     Args:
#         url (str): URL to scrape
#         company_name (str): Name of company for logging
        
#     Returns:
#         dict: Scraped data
#     """
#     with sync_playwright() as p:
#         # Use Chromium browser
#         browser = p.chromium.launch(headless=False)  # Set to True for headless mode
        
#         # Create a new context with specific viewport and user agent
#         context = browser.new_context(
#             viewport={"width": 1920, "height": 1080},
#             user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
#         )
        
#         # Create a new page
#         page = context.new_page()
        
#         # Navigate to the URL
#         page.goto(url, wait_until="networkidle")
        
#         # Wait for content to load
#         page.wait_for_selector("body", timeout=30000)
        
#         # Add random human-like behavior
#         page.mouse.move(random.randint(100, 500), random.randint(100, 500))
#         page.wait_for_timeout(random.uniform(1000, 3000))
        
#         # Extract data based on the website structure
#         data = {}
        
#         # Example: extract text from specific elements
#         if "crunchbase.com" in url:
#             # Crunchbase specific extraction
#             try:
#                 data["name"] = page.text_content("h1.profile-name").strip()
#                 data["description"] = page.text_content("div.description").strip()
#                 # Add more selectors as needed
#             except Exception as e:
#                 print(f"Error extracting Crunchbase data: {e}")
        
#         elif "ycombinator.com" in url:
#             # YCombinator specific extraction
#             try:
#                 data["companies"] = []
#                 company_cards = page.query_selector_all("div.CompanyCard_root__hK85u")
                
#                 for card in company_cards:
#                     company_info = {}
#                     try:
#                         company_info["name"] = card.query_selector("h3.CompanyCard_name__eAkro").text_content().strip()
#                         company_info["description"] = card.query_selector("div.CompanyCard_tagline__MZrn7").text_content().strip()
#                         # Add more fields as needed
#                         data["companies"].append(company_info)
#                     except:
#                         continue
#             except Exception as e:
#                 print(f"Error extracting YCombinator data: {e}")
        
#         # Take a screenshot for debugging
#         page.screenshot(path=f"{company_name}_screenshot.png")
        
#         # Close browser
#         browser.close()
        
#         return data

# if __name__ == "__main__":
#     # Example usage
#     yc_data = scrape_with_playwright("https://www.ycombinator.com/companies?batch=W24", "ycombinator")
    
#     with open('yc_playwright_data.json', 'w') as f:
#         json.dump(yc_data, f, indent=2)
    
#     # Add a significant delay before the next request
#     time.sleep(random.uniform(60, 120))
    
#     crunchbase_data = scrape_with_playwright("https://www.crunchbase.com/organization/openai", "openai")
    
#     with open('crunchbase_playwright_data.json', 'w') as f:
#         json.dump(crunchbase_data, f, indent=2)
    
#     print("Scraping completed!")



# """""""""


import time
import csv
import json
import logging
import os
import re
from dataclasses import dataclass, asdict
import random
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException, StaleElementReferenceException, WebDriverException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager

# Create organized directory structure
BASE_DIR = Path("fundraiseinsider_data")
CSV_DIR = BASE_DIR / "csv"
JSON_DIR = BASE_DIR / "json"
LOGS_DIR = BASE_DIR / "logs"
SCREENSHOT_DIR = BASE_DIR / "screenshots"

# Create all directories
for dir_path in [BASE_DIR, CSV_DIR, JSON_DIR, LOGS_DIR, SCREENSHOT_DIR]:
    dir_path.mkdir(exist_ok=True)

# Update logging configuration to use new directory
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "fundraiseinsider_scraping.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Startup:
    """Data class to store startup information"""
    company: str = ""
    total_employees: int = 0
    industry: str = ""
    website: str = ""
    funding_date: str = ""
    funding_type: str = ""
    funding_amount_usd: str = ""
    headquarters: str = ""
    funding_amount_normalized: float = None
    
    def __post_init__(self):
        # Convert total_employees to int if possible
        if isinstance(self.total_employees, str) and self.total_employees.isdigit():
            self.total_employees = int(self.total_employees)
        
        # Normalize funding amount
        if self.funding_amount_usd:
            try:
                # Remove non-numeric characters except decimal point
                amount_str = re.sub(r'[^\d.]', '', self.funding_amount_usd)
                if amount_str:
                    self.funding_amount_normalized = float(amount_str)
            except:
                pass

class FundraiseInsiderScraper:
    """Class to scrape startup data from fundraiseinsider.com"""
    BASE_URL = "https://fundraiseinsider.com/blog/recently-funded-startups-san-francisco/"
    
    def __init__(self, headless: bool = False):
        """Initialize with optimized Chrome options"""
        logger.info("Setting up Chrome WebDriver with optimizations")
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless=new")
        
        # Performance optimizations
        chrome_options.add_argument("--blink-settings=imagesEnabled=false")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-popup-blocking")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--window-size=1920,1080")
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.execute_script("window.performance.setResourceTimingBufferSize(500);")
        self.startups: List[Startup] = []
        logger.info("WebDriver setup complete")

    def log_heartbeat(self, message="Still running"):
        """Log a heartbeat message to show the script is still alive"""
        logger.info(f"HEARTBEAT: {message} - {len(self.startups)} startups collected so far")

    def take_debug_screenshot(self, name="debug"):
        """Take a screenshot for debugging purposes"""
        if SCREENSHOT_DIR:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.driver.save_screenshot(f"{SCREENSHOT_DIR}/{name}_{timestamp}.png")

    def analyze_page_structure(self):
        """Analyze the page structure to adapt scraping strategy"""
        try:
            table_exists = bool(self.driver.find_elements(By.TAG_NAME, "table"))
            cards_exist = bool(self.driver.find_elements(By.CSS_SELECTOR, ".startup-card, .company-card"))
            
            if table_exists:
                logger.info("Detected table-based layout")
                return "table"
            elif cards_exist:
                logger.info("Detected card-based layout")
                return "cards"
            else:
                logger.info("Unknown layout, using fallback extraction")
                return "unknown"
        except:
            return "unknown"

    def navigate_to_page(self, page_url: str = None):
        """Navigate to a specific page URL or the base URL"""
        url = page_url or self.BASE_URL
        logger.info(f"Navigating to: {url}")
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 5).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            logger.info("Page loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load page: {str(e)}")
            return False

    def wait_for_page_load(self, timeout=3):
        """Wait for the page to be fully loaded"""
        try:
            WebDriverWait(self.driver, timeout).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            time.sleep(2)  # Wait for JavaScript frameworks
            return True
        except TimeoutException:
            logger.error(f"Page did not load completely after {timeout} seconds")
            return False

    def handle_popup(self):
        """Enhanced popup handling with multiple strategies"""
        try:
            popup_exists = self.driver.execute_script("""
                return Boolean(
                    document.querySelector('div[role="dialog"], [class*="popup"], [class*="modal"], .overlay')
                );
            """)
            
            if not popup_exists:
                return False
                
            self.driver.execute_script("""
                ['div[role="dialog"]', '[class*="popup"]', '[class*="modal"]', '.overlay', '.lightbox']
                    .forEach(selector => {
                        document.querySelectorAll(selector).forEach(el => el.remove());
                    });
                document.body.style.overflow = 'auto';
                document.documentElement.style.overflow = 'auto';
            """)
            
            ActionChains(self.driver).send_keys(Keys.ESCAPE).perform()
            logger.info("Popup handled via JavaScript removal")
            return True
            
        except Exception as e:
            logger.warning(f"Popup handling warning (non-critical): {str(e)}")
            return False

    def ensure_browser_active(self):
        """Ensure browser is still active, restart if needed"""
        try:
            self.driver.current_url
            return True
        except WebDriverException:
            logger.warning("Browser appears to be closed, restarting...")
            try:
                self.cleanup()
            except:
                pass
                
            chrome_options = Options()
            chrome_options.add_argument("--blink-settings=imagesEnabled=false")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-popup-blocking")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--no-sandbox")
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            return False

    def human_scroll(self, scroll_amount=None):
        """Perform human-like scrolling with natural pauses"""
        try:
            # Random scroll amount if not specified
            if not scroll_amount:
                scroll_amount = random.randint(300, 700)
            
            # Smooth scroll with random speed
            scroll_time = random.uniform(0.5, 1.5)
            scroll_steps = int(scroll_amount / random.randint(20, 40))
            
            for step in range(0, scroll_amount, scroll_steps):
                self.driver.execute_script(f"window.scrollBy(0, {scroll_steps});")
                time.sleep(scroll_time / (scroll_amount / scroll_steps))
            
            # Random pause after scrolling
            time.sleep(random.uniform(0.5, 1.2))
            return True
        except Exception as e:
            logger.error(f"Scrolling error: {str(e)}")
            return False

    def scroll_with_timeout(self, max_scroll_time=60):
        """Improved scroll detection"""
        start_time = time.time()
        last_height = self.driver.execute_script("return document.documentElement.scrollHeight")
        last_count = len(self.startups)
        
        while time.time() - start_time < max_scroll_time:
            self.human_scroll()
            
            if random.random() < 0.3:
                time.sleep(random.uniform(1, 2))
            
            new_height = self.driver.execute_script("return document.documentElement.scrollHeight")
            current_count = len(self.startups)
            
            if new_height == last_height and current_count == last_count:
                end_markers = self.driver.find_elements(
                    By.CSS_SELECTOR, ".no-more-results, .end-of-content, .pagination-end, footer"
                )
                if end_markers:
                    logger.info("Found end of content marker")
                    break
                time.sleep(random.uniform(0.8, 1.5))
                if new_height == self.driver.execute_script("return document.documentElement.scrollHeight"):
                    break
            
            last_height = new_height
            last_count = current_count

    def extract_table_data(self):
        """Extract data with stale element handling"""
        try:
            table_container = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.table-container, table"))
            )
            return self.parse_table(table_container)
        except StaleElementReferenceException:
            logger.warning("Stale element detected, reinitializing...")
            return self.extract_table_data()
        except Exception as e:
            logger.error(f"Table extraction error: {str(e)}")
            return False

    def parse_table(self, container):
        """Parse table with live logging"""
        rows = container.find_elements(By.TAG_NAME, "tr")[1:]  # Skip header
        for idx, row in enumerate(rows, 1):
            try:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) < 7:
                    continue
                
                company = cells[0].text.strip()
                if not company or company in {s.company for s in self.startups}:
                    continue
                    
                website = cells[3].find_element(By.TAG_NAME, "a").get_attribute("href") if cells[3].find_elements(By.TAG_NAME, "a") else cells[3].text.strip()
                
                startup = Startup(
                    company=company,
                    total_employees=cells[1].text.strip(),
                    industry=cells[2].text.strip(),
                    website=website,
                    funding_date=cells[4].text.strip(),
                    funding_type=cells[5].text.strip(),
                    funding_amount_usd=cells[6].text.strip(),
                    headquarters=cells[7].text.strip() if len(cells) > 7 else ""
                )
                
                logger.info(f"Extracted Startup {idx}: {startup.company} | "
                           f"Funding: {startup.funding_amount_usd} | "
                           f"Industry: {startup.industry}")
                self.startups.append(startup)
                
            except StaleElementReferenceException:
                continue
            except Exception as e:
                logger.error(f"Error parsing row {idx}: {str(e)}")
                continue
        return True

    def click_next_page(self):
        """Human-like page navigation"""
        try:
            next_btn = WebDriverWait(self.driver, 3).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "a.pagination-next, [rel='next']"))
            )
            
            # Move mouse naturally to button
            actions = ActionChains(self.driver)
            actions.move_to_element_with_offset(next_btn, 
                random.randint(5, 20), random.randint(5, 15))
            actions.pause(random.uniform(0.1, 0.3))
            actions.move_to_element(next_btn)
            actions.pause(random.uniform(0.2, 0.4))
            actions.click()
            actions.perform()
            
            # Natural waiting time for page load
            time.sleep(random.uniform(1.5, 2.5))
            
            # Scroll slightly on new page (like a human)
            self.human_scroll(random.randint(100, 300))
            return True
            
        except Exception as e:
            logger.error(f"Navigation error: {str(e)}")
            return False

    def safe_extract(self, max_retries=3, timeout=300):
        """Error-resistant extraction with timeout protection"""
        start_time = time.time()
        
        for attempt in range(max_retries):
            if time.time() - start_time > timeout:
                logger.error(f"Extraction timed out after {timeout} seconds")
                return False
                
            try:
                if not self.navigate_to_page():
                    continue
                    
                self.handle_popup()
                self.driver.set_page_load_timeout(30)
                
                scroll_start = time.time()
                self.scroll_with_timeout(max_scroll_time=60)
                
                if self.extract_table_data():
                    return True
                    
            except WebDriverException as e:
                if "target window already closed" in str(e):
                    logger.error("Browser window was closed unexpectedly")
                    return False
                logger.error(f"WebDriver error: {str(e)}")
                if not self.ensure_browser_active():
                    break
            except Exception as e:
                logger.error(f"Extraction error: {str(e)}")
                break
        return False

    def extract_data(self, max_pages=None, max_runtime=600):
        """Main extraction flow with timeout protection"""
        start_time = time.time()
        
        if not self.safe_extract():
            return
            
        current_page = 1
        while True:
            if time.time() - start_time > max_runtime:
                logger.warning(f"Extraction stopped after reaching max runtime of {max_runtime} seconds")
                break
                
            if not self.ensure_browser_active():
                if not self.safe_extract():
                    break
            
            if max_pages and current_page >= max_pages:
                break
                
            if SCREENSHOT_DIR:
                self.driver.save_screenshot(f"{SCREENSHOT_DIR}/page_{current_page}.png")
                
            if not self.click_next_page():
                logger.info("No more pages to navigate")
                break
                
            if not self.safe_extract():
                break
                
            current_page += 1
            
            if current_page % 3 == 0 and self.startups:
                self.save_results(intermediate=True)
                
            self.log_heartbeat(f"Page {current_page} processed")
                
        if self.startups:
            self.save_results()
            logger.info(f"Extracted {len(self.startups)} startups from {current_page} pages")

    def save_results(self, intermediate=False):
        """Save results to organized directories"""
        if not self.startups:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{'interim_results' if intermediate else 'final_results'}_{timestamp}"
        
        # Save CSV
        csv_path = CSV_DIR / f"fundraiseinsider_startups_{base_name}.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Company", "Total Employees", "Industry", "Website", 
                           "Funding Date", "Funding Type", "Funding Amount (USD)", 
                           "Headquarters"])
            for startup in self.startups:
                writer.writerow([startup.company, startup.total_employees, 
                               startup.industry, startup.website, 
                               startup.funding_date, startup.funding_type,
                               startup.funding_amount_usd, startup.headquarters])
        
        # Save JSON
        json_path = JSON_DIR / f"fundraiseinsider_startups_{base_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(startup) for startup in self.startups], f, indent=4)
        
        if not intermediate:
            self._generate_analytics()
        
        logger.info(f"Saved {'interim' if intermediate else 'final'} results: {len(self.startups)} startups")

    def _generate_analytics(self):
        """Generate analytics about the startups"""
        logger.info("Generating analytics...")
        industry_counts = {}
        funding_types = {}
        funding_amounts = []
        for startup in self.startups:
            industry = startup.industry.strip()
            if industry:
                industry_counts[industry] = industry_counts.get(industry, 0) + 1
            funding_type = startup.funding_type.strip()
            if funding_type:
                funding_types[funding_type] = funding_types.get(funding_type, 0) + 1
            if startup.funding_amount_normalized:
                funding_amounts.append({"company": startup.company, "amount": startup.funding_amount_normalized})
        funding_amounts.sort(key=lambda x: x["amount"], reverse=True)
        analytics = {
            "total_startups": len(self.startups),
            "industries": dict(sorted(industry_counts.items(), key=lambda x: x[1], reverse=True)),
            "funding_types": dict(sorted(funding_types.items(), key=lambda x: x[1], reverse=True)),
            "top_funded": funding_amounts[:10] if funding_amounts else [],
            "average_funding": sum(item["amount"] for item in funding_amounts) / len(funding_amounts) if funding_amounts else 0,
            "total_funding": sum(item["amount"] for item in funding_amounts) if funding_amounts else 0,
            "extraction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        analytics_path = JSON_DIR / "fundraiseinsider_analytics.json"
        with open(analytics_path, 'w', encoding='utf-8') as f:
            json.dump(analytics, f, indent=4)
        logger.info(f"Saved analytics to: {analytics_path}")

    def cleanup(self):
        """Close the WebDriver and perform cleanup"""
        try:
            self.driver.quit()
            logger.info("WebDriver closed successfully")
        except Exception as e:
            logger.warning(f"Error closing WebDriver: {str(e)}")

def scheduled_execution():
    """Run the scraper on a schedule"""
    import schedule
    
    logger.info("Setting up scheduled execution")
    
    def job():
        """Job to run on schedule"""
        logger.info("Running scheduled job")
        scraper = FundraiseInsiderScraper(headless=True)
        
        try:
            scraper.extract_data()
        except Exception as e:
            logger.error(f"Error in scheduled job: {str(e)}")
        finally:
            scraper.cleanup()
        
        logger.info("Scheduled job completed")
        logger.info(f"Next run will be in 24 hours at {datetime.now() + timedelta(days=1)}")
    
    # Schedule the job to run daily at 2 AM
    schedule.every().day.at("02:00").do(job)
    logger.info("Job scheduled to run daily at 02:00")
    
    # Run the scheduler loop
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

def main():
    """Main function to run the scraper"""
    logger.info("Starting Fundraise Insider scraper")
    scraper = FundraiseInsiderScraper(headless=False)  # Set to True for production
    
    try:
        # Extract data from all pages 
        scraper.extract_data(max_pages=None)  # None means all pages
        
        # Show summary
        if scraper.startups:
            logger.info(f"Extraction completed successfully. Extracted {len(scraper.startups)} startups.")
    except Exception as e:
        logger.error(f"An error occurred during scraping: {str(e)}")
    finally:
        # Ensure proper cleanup
        scraper.cleanup()

if __name__ == "__main__":
    import argparse
    from datetime import timedelta
    
    parser = argparse.ArgumentParser(description="Fundraise Insider Scraper")
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run the scraper on a schedule (daily at 2 AM)"
    )
    
    args = parser.parse_args()
    
    if args.schedule:
        scheduled_execution()
    else:
        main()
