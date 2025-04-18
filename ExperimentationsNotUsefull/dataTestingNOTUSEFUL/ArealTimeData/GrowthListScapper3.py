import os
import time
import json
import random
import pandas as pd
import pytz
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from loguru import logger

class GrowthListScraper:
    def __init__(self):
        self.driver = self._init_driver()
        self.data = []
        self.pacific = pytz.timezone('US/Pacific')
        self.BAY_AREA_KEYWORDS = [
            'san francisco', 'sf', 'bay area', 'silicon valley',
            'oakland', 'san jose', 'palo alto', 'mountain view'
        ]

    def _init_driver(self):
        """Initialize stealth browser with anti-detection measures"""
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        options.add_argument("--window-size=1920,1080")
        
        # Disable logging for cleaner output
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=options)

    def _accept_cookies(self):
        """Handle cookie consent popup"""
        try:
            WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.ID, "CybotCookiebotDialogBodyButtonAccept"))
            ).click()
            logger.info("Accepted cookies")
            time.sleep(1)
        except Exception as e:
            logger.debug("No cookie consent found")

    def _scrape_page(self, page):
        """Scrape individual page with multiple fallbacks"""
        try:
            url = "https://growthlist.co/san-francisco-startups/" + (f"?page={page}" if page > 1 else "")
            self.driver.get(url)
            
            # Save page for debugging
            with open(f"debug_page_{page}.html", "w") as f:
                f.write(self.driver.page_source)
            
            # Wait for core content with multiple strategies
            try:
                WebDriverWait(self.driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.overflow-x-auto table.w-full"))
                )
            except:
                WebDriverWait(self.driver, 20).until(
                    EC.presence_of_element_located((By.TAG_NAME, "table"))
                )

            self._accept_cookies()
            return self._extract_table_data()
            
        except Exception as e:
            logger.error(f"Page {page} failed: {str(e)}")
            self.driver.save_screenshot(f"error_page_{page}.png")
            return []

    def _extract_table_data(self):
        """Extract and validate table data"""
        data = []
        table = self.driver.find_element(By.CSS_SELECTOR, "div.overflow-x-auto table.w-full")
        rows = table.find_elements(By.CSS_SELECTOR, "tbody tr:not(:first-child)")
        
        for row in rows:
            try:
                cols = row.find_elements(By.TAG_NAME, "td")
                if len(cols) != 7:
                    logger.warning(f"Skipping invalid row with {len(cols)} columns")
                    continue
                
                startup = {
                    'name': cols[0].text.strip(),
                    'website': cols[1].text.strip(),
                    'industry': cols[2].text.strip(),
                    'location': cols[3].text.strip().lower(),
                    'funding': self._parse_funding(cols[4].text),
                    'stage': cols[5].text.strip(),
                    'last_funding': cols[6].text.strip(),
                    'scraped_at': datetime.now(self.pacific).isoformat()
                }
                
                if any(kw in startup['location'] for kw in self.BAY_AREA_KEYWORDS):
                    data.append(startup)
                    logger.debug(f"Added: {startup['name']}")
                    
            except Exception as e:
                logger.error(f"Row error: {str(e)}")
        return data

    def _parse_funding(self, text):
        """Handle all funding formats from search results"""
        clean = text.replace('$', '').replace(',', '').strip()
        return float(clean) if clean.replace('.', '', 1).isdigit() else 0.0

    def _has_next_page(self):
        """Check pagination controls"""
        try:
            return "disabled" not in self.driver.find_element(
                By.CSS_SELECTOR, "a[rel='next']"
            ).get_attribute("class")
        except:
            return False

    def _save_data(self):
        """Create versioned dataset"""
        if not self.data:
            logger.error("No valid data collected")
            return
            
        timestamp = datetime.now(self.pacific).strftime("%Y%m%d_%H%M%S")
        output_dir = f"GrowthData_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.DataFrame(self.data)
        file_path = f"{output_dir}/bay_area_startups.csv"
        df.to_csv(file_path, index=False)
        logger.success(f"Successfully saved {len(df)} startups to {file_path}")

    def scrape_all(self):
        """Complete scraping workflow"""
        logger.info("Starting full scrape...")
        self.data = []
        page = 1
        
        try:
            while True:
                page_data = self._scrape_page(page)
                self.data.extend(page_data)
                logger.info(f"Page {page}: Collected {len(page_data)} valid startups")
                
                if not self._has_next_page() or page >= 10:  # Safety limit
                    break
                    
                page += 1
                time.sleep(random.uniform(2, 5))  # Human-like delay
                
            self._save_data()
            
        except Exception as e:
            logger.critical(f"Fatal error: {str(e)}")
            raise
        finally:
            self.driver.quit()

if __name__ == "__main__":
    scraper = GrowthListScraper()
    scraper.scrape_all()
