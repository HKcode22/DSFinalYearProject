import os
import time
import random
import json
import requests
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium_stealth import stealth
import undetected_chromedriver as uc
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("success_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WorkingScraper")

class BayAreaScraper:
    def __init__(self):
        self.output_dir = "startup_data"
        self._create_output_dir()
        self.driver = self._init_stealth_browser()

    def _create_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def _init_stealth_browser(self):
        """Initialize undetectable browser with advanced fingerprinting"""
        options = uc.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-web-security")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        # Random window size to avoid fingerprinting
        options.add_argument(f"--window-size={random.randint(1000,1400)},{random.randint(800,1200)}")
        
        service = Service(ChromeDriverManager().install())
        driver = uc.Chrome(service=service, options=options)
        
        stealth(driver,
                languages=["en-US", "en"],
                vendor="Google Inc.",
                platform="Win32",
                webgl_vendor="Intel Inc.",
                renderer="Intel Iris OpenGL Engine",
                fix_hairline=True)
        
        return driver

    def _save_data(self, data, filename):
        """Save data with validation"""
        if not data:
            logger.warning(f"No data to save for {filename}")
            return
            
        path = os.path.join(self.output_dir, filename)
        try:
            pd.DataFrame(data).to_csv(path, index=False)
            logger.info(f"Saved {len(data)} records to {filename}")
        except Exception as e:
            logger.error(f"Error saving {filename}: {str(e)}")

    def scrape_ycombinator(self):
        """Reverse-engineered YC API call from search results [1][7]"""
        logger.info("Starting Y Combinator scrape")
        try:
            # Direct API call discovered in search results
            response = requests.get(
                "https://api.ycombinator.com/companies/v1",
                params={
                    "location": "san-francisco-bay-area",
                    "limit": 100,
                    "fields": "name,website,description,batch"
                },
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
                    "Referer": "https://www.ycombinator.com/companies"
                }
            )
            
            companies = []
            for company in response.json()["companies"]:
                companies.append({
                    "name": company["name"],
                    "website": company.get("website", ""),
                    "description": company.get("description", ""),
                    "batch": company.get("batch", ""),
                    "source": "Y Combinator"
                })
            
            self._save_data(companies, "ycombinator.csv")
            return companies
            
        except Exception as e:
            logger.error(f"YC API error: {str(e)}")
            return []

    def scrape_growth_list(self):
        """Robust Growth List scraper with table parsing"""
        logger.info("Starting Growth List scrape")
        self.driver.get("https://growthlist.co/san-francisco-startups/")
        companies = []
        
        try:
            # Wait for table with explicit timeout
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table"))
            )
            
            # Extract table data using JavaScript execution
            companies = self.driver.execute_script('''
                return Array.from(document.querySelectorAll('table tr:not(:first-child)')).slice(0,100).map(row => {
                    const cols = row.querySelectorAll('td');
                    return {
                        name: cols[0]?.innerText?.trim(),
                        website: cols[1]?.innerText?.trim(),
                        category: cols[2]?.innerText?.trim(),
                        funding: cols[4]?.innerText?.trim(),
                        round: cols[5]?.innerText?.trim(),
                        date: cols[6]?.innerText?.trim(),
                        source: 'Growth List'
                    };
                }).filter(item => item.name);
            ''')
            
            self._save_data(companies, "growth_list.csv")
            return companies
            
        except Exception as e:
            logger.error(f"Growth List error: {str(e)}")
            return []

    def scrape_crunchbase(self):
        """Crunchbase JSON-LD extraction from search results [3][9]"""
        logger.info("Starting Crunchbase scrape")
        self.driver.get("https://www.crunchbase.com/search/organizations/field/organization_locations/san-francisco-bay-area")
        companies = []
        
        try:
            # Wait for JSON-LD data with extended timeout
            WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.XPATH, '//script[@type="application/ld+json"]'))
            )
            
            # Extract and parse JSON-LD data
            script = self.driver.find_element(By.XPATH, '//script[@type="application/ld+json"]')
            data = json.loads(script.get_attribute("innerHTML"))
            
            companies = [{
                "name": item["item"]["name"],
                "description": item["item"].get("description", ""),
                "website": item["item"].get("url", ""),
                "funding": item["item"].get("fundingTotal", {}).get("value", ""),
                "source": "Crunchbase"
            } for item in data["itemListElement"][:100]]  # Limit to 100 entries
            
            self._save_data(companies, "crunchbase.csv")
            return companies
            
        except Exception as e:
            logger.error(f"Crunchbase error: {str(e)}")
            return []

    def run(self):
        """Execute all scrapers with optimized flow"""
        logger.info("Starting reliable data collection...")
        
        results = {}
        try:
            # YC first - API based
            results["yc"] = self.scrape_ycombinator()
            time.sleep(random.uniform(3, 5))
            
            # Growth List with browser
            results["growth_list"] = self.scrape_growth_list()
            time.sleep(random.uniform(3, 5))
            
            # Crunchbase with browser
            results["crunchbase"] = self.scrape_crunchbase()
            
        finally:
            self.driver.quit()
        
        # Print summary
        print("\nFinal Results:")
        print(f"Y Combinator: {len(results.get('yc', []))} companies")
        print(f"Growth List: {len(results.get('growth_list', []))} companies")
        print(f"Crunchbase: {len(results.get('crunchbase', []))} companies")
        return results

if __name__ == "__main__":
    scraper = BayAreaScraper()
    final_data = scraper.run()
