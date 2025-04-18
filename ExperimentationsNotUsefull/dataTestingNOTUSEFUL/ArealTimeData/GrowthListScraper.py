import os
import time
import random
import pandas as pd
import pytz
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from apscheduler.schedulers.blocking import BlockingScheduler
from loguru import logger
from faker import Faker

class GrowthListScraper:
    def __init__(self):
        self.fake = Faker()
        self.driver = self._init_driver()
        self.data = []
        self.scraped_urls = set()
        self.pacific = pytz.timezone('US/Pacific')
        self.last_update = None  # Initialize with default value
        
    def _check_data_freshness(self):
        """Verify weekly updates using GrowthList's timestamp"""
        try:
            self.driver.get("https://growthlist.co/san-francisco-startups/")
            update_element = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.XPATH, "//*[contains(., 'Last updated')]"))
            )
            date_str = update_element.text.split("Last updated: ")[1].split(" ")[0]
            self.last_update = datetime.strptime(date_str, "%Y-%m-%d").astimezone(self.pacific)
            
            # Calculate expected Sunday update window
            last_sunday = datetime.now(self.pacific) - timedelta(
                (datetime.now(self.pacific).weekday() + 1) % 7
            )
            return self.last_update.date() >= last_sunday.date()
            
        except Exception as e:
            logger.error("Freshness check failed: {}", str(e))
            self.last_update = datetime.now(self.pacific) - timedelta(days=365)  # Default to old date
            return False
    def _scrape_page(self, url):
        """Scrape individual page with error recovery"""
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table.w-full"))
            )
            
            # Human-like interaction pattern
            for _ in range(3):
                self.driver.execute_script("window.scrollBy(0, 500)")
                time.sleep(random.uniform(0.5, 1.5))
                
            rows = self.driver.find_elements(By.CSS_SELECTOR, "table.w-full tr:not(:first-child)")
            for row in rows:
                try:
                    cols = row.find_elements(By.TAG_NAME, "td")
                    if len(cols) >= 7:
                        startup = {
                            'name': cols[0].text.strip(),
                            'website': cols[1].text.strip(),
                            'industry': cols[2].text.strip(),
                            'location': cols[3].text.strip(),
                            'funding': self._clean_funding(cols[4].text),
                            'stage': cols[5].text.strip(),
                            'last_funding': self._parse_date(cols[6].text.strip()),
                            'scraped_at': datetime.now(self.pacific).strftime('%Y-%m-%d %H:%M:%S')
                        }
                        if self._is_valid_startup(startup):
                            self.data.append(startup)
                            
                except Exception as e:
                    logger.error("Row error: {}", str(e))
                    continue

        except Exception as e:
            logger.error("Page error: {}", str(e))
            self.driver.save_screenshot(f"error_{datetime.now().timestamp()}.png")

    def _clean_funding(self, text):
        """Convert funding text to numeric USD"""
        return float(text.replace('$', '').replace(',', '').strip() or 0)

    def _parse_date(self, date_str):
        """Convert GrowthList date formats to datetime"""
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
        except:
            return pd.NaT

    def _is_valid_startup(self, startup):
        """Validate Bay Area startups with funding"""
        location_terms = ['san francisco', 'sf', 'bay area', 'silicon valley']
        return (
            any(term in startup['location'].lower() for term in location_terms) and
            startup['funding'] > 0 and
            pd.notna(startup['last_funding'])
        )

    def scrape(self):
        """Main scraping workflow"""
        if not self._check_data_freshness():
            logger.info("No new data available. Last update: {}", self.last_update)
            return
            
        logger.info("New weekly data detected! Starting scrape...")
        
        # Scrape paginated results
        for page in range(1, 11):  # 10 pages × 50 startups = 500
            url = f"https://growthlist.co/san-francisco-startups/?page={page}"
            if url not in self.scraped_urls:
                self._scrape_page(url)
                self.scraped_urls.add(url)
                time.sleep(random.uniform(2, 5))  # Rate limiting
                
        logger.success("Scraped {} valid startups", len(self.data))

    def save_data(self):
        """Merge new data with historical records"""
        if not self.data:
            return
            
        df = pd.DataFrame(self.data)
        df['last_funding'] = pd.to_datetime(df['last_funding'])
        
        historical_path = "AMergedCsvFiles2/growthlist_full.csv"
        if os.path.exists(historical_path):
            historical = pd.read_csv(historical_path, parse_dates=['last_funding'])
            merged = pd.concat([historical, df]).drop_duplicates(['website', 'last_funding'])
            merged.to_csv(historical_path, index=False)
        else:
            df.to_csv(historical_path, index=False)
            
        logger.success("Saved {} total startups to {}", len(df), historical_path)

    def run(self):
        try:
            self.scrape()
            self.save_data()
        finally:
            self.driver.quit()

def scheduled_task():
    logger.info("=== Starting scheduled scrape ===")
    scraper = GrowthListScraper()
    scraper.run()
    logger.info("=== Completed scheduled scrape ===")


if __name__ == "__main__":
    # Configure scheduler for Monday 00:05 PT
    scheduler = BlockingScheduler(timezone='US/Pacific')
    scheduler.add_job(
        scheduled_task,
        'cron',
        day_of_week='mon',
        hour=0,
        minute=5,
        misfire_grace_time=3600
    )
    
    # Calculate next run manually
    pacific = pytz.timezone('US/Pacific')
    now = datetime.now(pacific)
    days_until_monday = (7 - now.weekday()) % 7  # Monday is weekday 0
    next_run_time = (now + timedelta(days=days_until_monday))\
        .replace(hour=0, minute=5, second=0, microsecond=0)
    
    logger.info("Scheduler initialized. Next run: {}", next_run_time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    
    # TESTING: Force immediate execution
    logger.info("=== Starting manual test run ===")
    scheduled_task()  # This will create the CSV immediately
    logger.info("=== Completed manual test run ===")
    
    try:
        scheduler.start()  # Keep this for scheduled runs
    except KeyboardInterrupt:
        logger.info("Scheduler shutdown requested")
 