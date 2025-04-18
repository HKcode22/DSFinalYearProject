import time
import csv
import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException, StaleElementReferenceException
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

# Configure logging
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
    """Startup data model"""
    company: str
    total_employees: str
    industry: str
    website: str
    funding_date: str
    funding_type: str
    funding_amount_usd: str
    headquarters: str

class FundraiseInsiderScraper:
    """Scraper for FundraiseInsider with improved pagination"""
    BASE_URL = "https://fundraiseinsider.com/blog/recently-funded-startups-san-francisco/"
    
    def __init__(self, headless: bool = False):
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless=new")
        
        # Essential options for stability
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.startups = []
        logger.info("WebDriver setup complete")

    def navigate_to_page(self):
        """Navigate to page with error handling"""
        try:
            self.driver.get(self.BASE_URL)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            return True
        except Exception as e:
            logger.error(f"Navigation error: {str(e)}")
            return False

    def handle_popup(self):
        """Reliable popup handling"""
        try:
            self.driver.execute_script("""
                const selectors = [
                    '.modal', '.popup', '.modal-backdrop',
                    '[class*="modal"]', '[class*="popup"]'
                ];
                selectors.forEach(s => {
                    document.querySelectorAll(s).forEach(el => el.remove());
                });
                document.body.style.overflow = 'auto';
                document.documentElement.style.overflow = 'auto';
            """)
            return True
        except Exception as e:
            logger.error(f"Popup handling error: {str(e)}")
            return False

    def extract_table_data(self):
        """Extract data with specific table structure handling"""
        try:
            # Wait for table with specific structure
            table = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table.wp-block-table table, .tablepress"))
            )
            
            # Get current startup count for verification
            startups_before = len(self.startups)
            
            # Process rows with column verification
            rows = table.find_elements(By.TAG_NAME, "tr")
            header = rows[0] if rows else None
            
            if header:
                # Verify column structure
                headers = [col.text.strip().lower() for col in header.find_elements(By.TAG_NAME, "th")]
                expected_columns = ['company', 'employees', 'industry', 'website', 'funding date', 'type', 'amount']
                
                if not any(col in ' '.join(headers) for col in expected_columns):
                    logger.warning("Table structure doesn't match expected format")
                    return False
                
                # Process data rows
                for row in rows[1:]:
                    try:
                        cells = row.find_elements(By.TAG_NAME, "td")
                        if len(cells) < 7:
                            continue
                        
                        company = cells[0].text.strip()
                        if not company or company in {s.company for s in self.startups}:
                            continue
                        
                        startup = Startup(
                            company=company,
                            total_employees=cells[1].text.strip(),
                            industry=cells[2].text.strip(),
                            website=cells[3].find_element(By.TAG_NAME, "a").get_attribute("href") 
                                   if cells[3].find_elements(By.TAG_NAME, "a") else cells[3].text.strip(),
                            funding_date=cells[4].text.strip(),
                            funding_type=cells[5].text.strip(),
                            funding_amount_usd=cells[6].text.strip(),
                            headquarters=cells[7].text.strip() if len(cells) > 7 else ""
                        )
                        
                        self.startups.append(startup)
                        logger.info(f"Added: {startup.company}")
                        
                    except Exception as e:
                        logger.error(f"Row processing error: {str(e)}")
                        continue
            
            # Return True if we found new startups
            return len(self.startups) > startups_before
            
        except Exception as e:
            logger.error(f"Table extraction error: {str(e)}")
            return False

    def handle_infinite_scroll(self):
        """Handle infinite scroll with AJAX content loading"""
        try:
            initial_height = self.driver.execute_script("return document.documentElement.scrollHeight")
            initial_count = len(self.startups)
            scroll_attempts = 0
            max_attempts = 3

            while scroll_attempts < max_attempts:
                # Scroll in smaller increments
                current_height = self.driver.execute_script("return window.pageYOffset")
                target_height = current_height + 500
                
                # Smooth scroll
                self.driver.execute_script(f"""
                    window.scrollTo({{
                        top: {target_height},
                        behavior: 'smooth'
                    }});
                """)
                time.sleep(1)

                # Check for new content
                new_height = self.driver.execute_script("return document.documentElement.scrollHeight")
                if new_height > initial_height:
                    logger.info("Detected new content after scroll")
                    self.extract_table_data()
                    if len(self.startups) > initial_count:
                        logger.info(f"Found {len(self.startups) - initial_count} new items")
                        return True
                
                scroll_attempts += 1
            
            logger.info("No new content found after scrolling")
            return False
            
        except Exception as e:
            logger.error(f"Scroll error: {str(e)}")
            return False

    def wait_for_table_load(self):
        """Wait for dynamic table loading"""
        try:
            initial_height = self.driver.execute_script("return document.documentElement.scrollHeight")
            initial_content = self.driver.page_source
            retry_count = 0
            
            while retry_count < 3:
                # Scroll to different positions to trigger loading
                scroll_positions = [300, 500, 800, 1000]
                for position in scroll_positions:
                    self.driver.execute_script(f"window.scrollTo(0, {position})")
                    time.sleep(0.5)
                
                # Check for changes
                new_height = self.driver.execute_script("return document.documentElement.scrollHeight")
                new_content = self.driver.page_source
                
                if new_height > initial_height or new_content != initial_content:
                    logger.info("Detected dynamic content loading")
                    return True
                    
                retry_count += 1
                time.sleep(1)
            
            return False
            
        except Exception as e:
            logger.error(f"Error waiting for table load: {str(e)}")
            return False

    def try_url_patterns(self, current_page):
        """Try different URL patterns for pagination"""
        base_url = self.BASE_URL.rstrip('/')
        patterns = [
            f"{base_url}/page/{current_page}",
            f"{base_url}?page={current_page}",
            f"{base_url}?paged={current_page}",
            f"{base_url}&page={current_page}"
        ]
        
        initial_count = len(self.startups)
        for url in patterns:
            try:
                logger.info(f"Trying URL pattern: {url}")
                self.driver.get(url)
                
                # Wait for table and check if it's valid
                WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.TAG_NAME, "table"))
                )
                
                if "404" not in self.driver.title:
                    self.extract_table_data()
                    if len(self.startups) > initial_count:
                        logger.info(f"Successfully loaded page via URL: {url}")
                        return True
            except:
                continue
        return False

    def click_next_page(self):
        """Hybrid pagination with real-time DOM inspection"""
        try:
            initial_count = len(self.startups)
            
            # Method 1: Check for infinite scroll/AJAX loading
            last_height = self.driver.execute_script("return document.documentElement.scrollHeight")
            self.driver.execute_script(f"window.scrollTo(0, {last_height})")
            time.sleep(2)
            
            # Verify if new content loaded
            self.extract_table_data()
            if len(self.startups) > initial_count:
                logger.info("Successfully loaded more content via scroll")
                return True
            
            # Method 2: Look for and click view more/load more button
            more_button = self.driver.execute_script("""
                return Array.from(document.querySelectorAll('*')).find(el => {
                    const style = window.getComputedStyle(el);
                    const text = (el.textContent || '').toLowerCase();
                    return style.display !== 'none' && 
                           style.visibility !== 'hidden' && 
                           (text.includes('view more') || 
                            text.includes('load more') || 
                            text.includes('show more'));
                });
            """)
            
            if more_button:
                logger.info("Found load more button")
                self.driver.execute_script("arguments[0].click();", more_button)
                time.sleep(2)
                self.extract_table_data()
                if len(self.startups) > initial_count:
                    return True
            
            # Method 3: Try direct URL modification
            current_url = self.driver.current_url
            next_page = 2  # Default to page 2 if no page number found
            
            page_match = re.search(r'/page/(\d+)', current_url)
            if page_match:
                next_page = int(page_match.group(1)) + 1
            
            next_url = f"{self.BASE_URL.rstrip('/')}/page/{next_page}/"
            logger.info(f"Attempting to load: {next_url}")
            
            # Store current page source for comparison
            current_source = self.driver.page_source
            
            self.driver.get(next_url)
            
            # Wait for and verify new content
            try:
                WebDriverWait(self.driver, 5).until(
                    lambda d: d.page_source != current_source and
                    len(d.find_elements(By.TAG_NAME, "table")) > 0
                )
                self.extract_table_data()
                if len(self.startups) > initial_count:
                    logger.info(f"Successfully loaded page {next_page}")
                    return True
            except:
                logger.info("No more pages available")
                return False
            
            return False
            
        except Exception as e:
            logger.error(f"Navigation error: {str(e)}")
            return False

    def save_results(self):
        """Save results to CSV and JSON"""
        if not self.startups:
            return

        # Save CSV
        csv_path = CSV_DIR / "fundraiseinsider_latest.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Company", "Employees", "Industry", "Website", 
                           "Funding Date", "Funding Type", "Amount", "HQ"])
            for s in self.startups:
                writer.writerow([s.company, s.total_employees, s.industry,
                               s.website, s.funding_date, s.funding_type,
                               s.funding_amount_usd, s.headquarters])

        # Save JSON
        json_path = JSON_DIR / "fundraiseinsider_latest.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(s) for s in self.startups], f, indent=2)

        logger.info(f"Saved {len(self.startups)} startups to output files")

    def cleanup(self):
        """Clean shutdown"""
        try:
            if self.driver:
                self.driver.quit()
                logger.info("Browser closed successfully")
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")
        finally:
            self.driver = None

    def run(self):
        """Enhanced execution with better progress tracking"""
        try:
            if not self.navigate_to_page():
                return
            
            page = 1
            consecutive_empty = 0
            max_empty = 2  # Stop after 2 consecutive empty pages
            
            while True:
                logger.info(f"Processing page {page}")
                initial_count = len(self.startups)
                
                # Handle any popups
                self.handle_popup()
                
                # Extract data
                self.extract_table_data()
                
                if len(self.startups) == initial_count:
                    consecutive_empty += 1
                    logger.warning(f"No new data found (attempt {consecutive_empty})")
                    if consecutive_empty >= max_empty:
                        logger.info("Stopping: No new data in consecutive attempts")
                        break
                else:
                    consecutive_empty = 0
                    logger.info(f"Found {len(self.startups) - initial_count} new startups")
                
                # Try next page
                if not self.click_next_page():
                    logger.info("No more pages available")
                    break
                
                page += 1
                if page % 2 == 0:  # Save progress every 2 pages
                    self.save_results()
            
            if self.startups:
                self.save_results()
                logger.info(f"Completed scraping with {len(self.startups)} total startups")
                
        except Exception as e:
            logger.error(f"Error during execution: {str(e)}")
        finally:
            self.cleanup()

def main():
    """Main entry point"""
    logger.info("Starting Fundraise Insider scraper")
    scraper = FundraiseInsiderScraper(headless=False)
    scraper.run()

if __name__ == "__main__":
    main()
