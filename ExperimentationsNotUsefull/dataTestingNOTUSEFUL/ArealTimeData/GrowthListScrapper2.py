import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import logging
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('growth_scraper.log'), logging.StreamHandler()]
)

class GrowthListScraper:
    def __init__(self):
        self.base_url = "https://growthlist.co/san-francisco-startups/"
        self.data_dir = "growth_data"
        self.total_collected = 0
        self.driver = self._init_browser()
        os.makedirs(self.data_dir, exist_ok=True)

    def _init_browser(self):
        """Headless browser with anti-detection features"""
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--window-size=1920,1080")
        return webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )

    def _handle_infinite_scroll(self):
        """Scroll to bottom and trigger dynamic loading"""
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    def _parse_table(self):
        """Extract data from both static and dynamic tables"""
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        
        # Handle main table
        main_table = soup.find('table')
        if main_table:
            return self._parse_table_rows(main_table)
            
        # Handle dynamic cards (fallback)
        return self._parse_card_layout(soup)

    def _parse_table_rows(self, table):
        """Parse traditional HTML table"""
        rows = table.find_all('tr')[1:]  # Skip header
        return [self._parse_row(row) for row in rows]

    def _parse_card_layout(self, soup):
        """Handle alternative card-based layout"""
        cards = soup.find_all('div', class_='startup-card')
        return [self._parse_card(card) for card in cards]

    def _parse_row(self, row):
        """Standard row parsing with error handling"""
        try:
            cols = row.find_all('td')
            return {
                'name': cols[0].text.strip(),
                'website': cols[1].text.strip(),
                'industry': cols[2].text.strip(),
                'country': cols[3].text.strip(),
                'funding': self._parse_funding(cols[4].text),
                'funding_type': cols[5].text.strip(),
                'last_funding': pd.to_datetime(cols[6].text.strip()),
                'scraped_at': datetime.now().isoformat()
            }
        except Exception as e:
            logging.error(f"Row error: {str(e)}")
            return None

    def _parse_funding(self, text):
        """Convert multiple funding formats"""
        conversions = {'M': 1e6, 'B': 1e9, 'k': 1e3}
        clean_text = text.replace('$', '').replace(',', '').strip()
        
        for suffix, mult in conversions.items():
            if suffix in clean_text:
                return float(clean_text.replace(suffix, '')) * mult
        try:
            return float(clean_text)
        except:
            return None

    def _save_data(self, data):
        """Save with version control"""
        df = pd.DataFrame([item for item in data if item])
        master_path = os.path.join(self.data_dir, 'master.csv')
        
        if os.path.exists(master_path):
            existing = pd.read_csv(master_path)
            df = pd.concat([existing, df]).drop_duplicates(subset=['website'])
            
        df.to_csv(master_path, index=False)
        logging.info(f"Saved {len(df)} entries")

    def scrape_all_pages(self):
        """Hybrid scraping strategy"""
        self.driver.get(self.base_url)
        time.sleep(3)
        
        # Handle infinite scroll first
        self._handle_infinite_scroll()
        
        # Pagination fallback
        page = 1
        while True:
            data = self._parse_table()
            valid_data = [item for item in data if item]
            
            if valid_data:
                self.total_collected += len(valid_data)
                self._save_data(valid_data)
                logging.info(f"Page {page}: Collected {len(valid_data)} (Total: {self.total_collected})")
                
                # Check next page
                try:
                    next_btn = self.driver.find_element(By.CSS_SELECTOR, 'a[aria-label="Next"]')
                    next_btn.click()
                    time.sleep(random.uniform(2, 5))
                    page += 1
                except:
                    break
            else:
                break

        self.driver.quit()
        logging.info(f"Final collection: {self.total_collected} startups")

if __name__ == "__main__":
    scraper = GrowthListScraper()
    scraper.scrape_all_pages()
