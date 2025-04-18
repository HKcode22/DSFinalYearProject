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
from apscheduler.schedulers.blocking import BlockingScheduler
from loguru import logger

class GrowthListScraper:
    BASE_URL = "https://growthlist.co/funded-startups/"
    OUTPUT_DIR = "GrowthList_Data"
    
    def __init__(self):
        self.driver = None
        self.pacific = pytz.timezone('US/Pacific')
        
    def _init_driver(self):
        """Initialize browser with stealth settings"""
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=options)

    def _get_update_time(self):
        """Extract the last update timestamp from the page"""
        try:
            elements = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Last updated')]")
            if elements:
                text = elements[0].text
                date_str = text.split("Last updated: ")[1].split(" ")[0]
                return datetime.strptime(date_str, "%Y-%m-%d").astimezone(self.pacific)
            
            # Fallback: use funding date from first entry
            rows = self.driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
            if rows:
                date_cell = rows[0].find_elements(By.TAG_NAME, "td")[6]
                return datetime.strptime(f"2025-{date_cell.text}", "%Y-%b %Y").astimezone(self.pacific)
                
            return datetime.now(self.pacific)
        except Exception as e:
            logger.error(f"Update time extraction failed: {str(e)}")
            return datetime.now(self.pacific)

    def _scrape_table(self):
        """Extract startup data from the current page"""
        try:
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
            
            rows = self.driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
            logger.info(f"Found {len(rows)} rows in table")
            
            data = []
            for row in rows:
                try:
                    cols = row.find_elements(By.TAG_NAME, "td")
                    if len(cols) >= 7:
                        startup = {
                            'name': cols[0].text.strip(),
                            'website': cols[1].text.strip(),
                            'industry': cols[2].text.strip(),
                            'country': cols[3].text.strip(),
                            'funding': cols[4].text.replace('$', '').replace(',', '').strip(),
                            'stage': cols[5].text.strip(),
                            'last_funding': cols[6].text.strip(),
                            'scraped_at': datetime.now(self.pacific).isoformat()
                        }
                        data.append(startup)
                except Exception as e:
                    logger.warning(f"Row error: {str(e)}")
                    
            return data
        except Exception as e:
            logger.error(f"Table extraction failed: {str(e)}")
            self.driver.save_screenshot("table_error.png")
            return []

    def _check_for_updates(self):
        """Check if there are new updates on GrowthList"""
        try:
            # Check if any previous data exists
            os.makedirs(self.OUTPUT_DIR, exist_ok=True)
            files = sorted([f for f in os.listdir(self.OUTPUT_DIR) if f.endswith('.json')])
            
            if not files:
                logger.info("No previous data found, proceeding with scrape")
                return True
                
            # Get the latest file and its date
            latest_file = files[-1]
            with open(os.path.join(self.OUTPUT_DIR, latest_file), 'r') as f:
                previous_data = json.load(f)
            
            if not previous_data:
                return True
                
            # Initialize driver and check current update time
            self.driver = self._init_driver()
            self.driver.get(self.BASE_URL)
            current_update = self._get_update_time()
            
            # Get date from previous data
            prev_date_str = previous_data[0].get('scraped_at', '').split('T')[0]
            prev_date = datetime.strptime(prev_date_str, "%Y-%m-%d").astimezone(self.pacific)
            
            # Compare dates
            if current_update.date() > prev_date.date():
                logger.info(f"New update found: {current_update.date()} vs previous {prev_date.date()}")
                return True
            
            logger.info("No new updates since last scrape")
            return False
            
        except Exception as e:
            logger.error(f"Update check failed: {str(e)}")
            return True
        finally:
            if self.driver:
                self.driver.quit()

    def _save_data(self, data):
        """Save data to CSV and JSON"""
        if not data:
            logger.error("No data to save")
            return False
            
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now(self.pacific).strftime("%Y%m%d_%H%M%S")
        
        # CSV
        df = pd.DataFrame(data)
        csv_path = os.path.join(self.OUTPUT_DIR, f"growthlist_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        
        # JSON 
        json_path = os.path.join(self.OUTPUT_DIR, f"growthlist_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.success(f"Saved {len(data)} startups to {csv_path}")
        return True

    def _detect_total_pages(self):
        """Smart detection of total pages using multiple methods"""
        try:
            # Method 1: Count pagination buttons
            pagination_items = self.driver.find_elements(By.CSS_SELECTOR, "ul.pagination li:not(.previous):not(.next)")
            if pagination_items:
                return max([int(item.text) for item in pagination_items if item.text.isdigit()])
            
            # Method 2: Check URL parameters in links
            page_links = self.driver.find_elements(By.CSS_SELECTOR, "a.page-link[href*='?page=']")
            if page_links:
                return max([int(link.get_attribute("href").split('=')[-1]) for link in page_links])
            
            # Method 3: Progressive discovery
            max_page = 1
            for _ in range(20):  # Safety limit
                if not self._has_next_page():
                    break
                self._click_next_button()
                max_page += 1
                time.sleep(1)
            return max_page
            
        except Exception as e:
            logger.warning(f"Page detection failed: {str(e)}, using conservative estimate")
            return 5

    def _has_next_page(self):
        """Check if next page exists"""
        try:
            next_btn = self.driver.find_element(By.CSS_SELECTOR, "a[rel='next'], li.next a")
            return "disabled" not in next_btn.get_attribute("class")
        except:
            return False

    def _click_next_button(self):
        """Click next page button with error handling"""
        try:
            next_btn = self.driver.find_element(By.CSS_SELECTOR, "a[rel='next'], li.next a")
            self.driver.execute_script("arguments[0].scrollIntoView(true);", next_btn)
            self.driver.execute_script("arguments[0].click();", next_btn)
            time.sleep(2)
            return True
        except Exception as e:
            logger.warning(f"Next button click failed: {str(e)}")
            return False

    def scrape(self):
        """Dynamic pagination handling with smart navigation"""
        logger.info("Starting GrowthList scrape...")
        self.driver = self._init_driver()
        all_data = []
        current_page = 1
        
        try:
            # Initial page load and page count detection
            self.driver.get(self.BASE_URL)
            time.sleep(3)
            total_pages = self._detect_total_pages()
            logger.info(f"Detected {total_pages} pages to scrape")

            while True:
                logger.info(f"Processing page {current_page}/{total_pages}")
                
                # Scrape current page
                page_data = self._scrape_table()
                all_data.extend(page_data)
                logger.info(f"Collected {len(page_data)} startups from page {current_page}")
                
                # Exit conditions
                if current_page >= total_pages or not self._has_next_page():
                    break
                    
                # Navigate to next page
                current_page += 1
                if current_page <= total_pages:
                    self.driver.get(f"{self.BASE_URL}?page={current_page}")
                else:
                    if not self._click_next_button():
                        break
                
                time.sleep(random.uniform(1, 3))

            self._save_data(all_data)
            logger.info(f"Total startups collected: {len(all_data)}")
            return len(all_data) > 0
            
        except Exception as e:
            logger.critical(f"Scraping failed: {str(e)}")
            return False
        finally:
            if self.driver:
                self.driver.quit()

    def schedule(self):
        """Scheduler with improved reliability"""
        scheduler = BlockingScheduler(timezone='US/Pacific')
        scheduler.add_job(
            self._scheduled_job,
            'cron',
            day_of_week="sun",
            hour=20,
            minute=5
        )
        
        logger.info("Performing initial scrape...")
        self.scrape()
        
        try:
            logger.info("Next scheduled run: Sundays at 8:05 PM PT")
            scheduler.start()
        except KeyboardInterrupt:
            logger.info("Scheduler stopped")
        except Exception as e:
            logger.error(f"Scheduler error: {str(e)}")

    def _scheduled_job(self):
        """Automated update check and scrape"""
        if self._check_for_updates():
            logger.info("Updates detected, starting scrape...")
            self.scrape()
        else:
            logger.info("No updates found, skipping scrape")

if __name__ == "__main__":
    scraper = GrowthListScraper()
    scraper.schedule()
