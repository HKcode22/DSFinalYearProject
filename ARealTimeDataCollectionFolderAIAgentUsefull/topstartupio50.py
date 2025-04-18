import time
import json
import csv
import random
import logging
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import schedule

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, 
    NoSuchElementException, 
    StaleElementReferenceException,
    ElementClickInterceptedException,
    WebDriverException
)

from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd

# Create directory for storing data
DATA_DIR = Path("topstartiorealtimedata")
DATA_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(DATA_DIR / "topstartups_scraping.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Startup:
    """Data class to store startup information"""
    name: str = ""
    description: str = ""
    headquarters: str = ""
    funding: str = ""
    website: str = ""
    has_reviews: bool = False
    category: str = ""
    employees: str = ""
    founding_year: str = ""
    other_details: Dict[str, str] = None
    
    def __post_init__(self):
        if self.other_details is None:
            self.other_details = {}

class TopStartupsScraperConfig:
    """Configuration for the TopStartups scraper"""
    BASE_URL = "https://topstartups.io/"
    DEFAULT_QUERY = "?hq_location=San+Francisco+Bay+Area&sort=funding"
    
    # CSS Selectors (improved for better extraction)
    CARD_SELECTORS = [
        "div[class*='card']",
        "article",
        "div[class*='company']",
        "div[class*='startup']"
    ]
    
    NAME_SELECTORS = [
        "h2", 
        "h3",
        "div > h2",
        ".company-name",
        "div[class*='name']",
        "div[class*='title']"
    ]
    
    DESCRIPTION_SELECTORS = [
        "div:contains('What they do') + div",
        "div:contains('What they do')",
        "p[class*='description']",
        "div[class*='description']"
    ]
    
    HEADQUARTERS_SELECTORS = [
        "div:contains('HQ:') span",
        "div:contains('📍')",
        "div[class*='location']",
        "div[class*='quick-facts']"
    ]
    
    FUNDING_SELECTORS = [
        "div:contains('Funding:') + div",
        "div[class*='funding']",
        "div:contains('Funding:')"
    ]
    
    WEBSITE_LINK_SELECTORS = [
        "a:contains('Check company site')",
        "a:contains('company site')",
        "a[target='_blank']",
        "div:contains('Take action:') a",
        "a[href*='http']"
    ]
    
    # Scrolling parameters - EXTREMELY FAST
    MIN_SCROLL_DELAY = 0.01  # Reduced to near-zero
    MAX_SCROLL_DELAY = 0.03  # Reduced to near-zero
    MIN_SCROLL_AMOUNT = 500  # Increased for faster scrolling
    MAX_SCROLL_AMOUNT = 1000  # Increased for faster scrolling
    SCROLL_BEHAVIOR_VARIANCE = 0.1  # Minimal variance for consistent speed
    
    # Waiting parameters - EXTREMELY FAST
    DEFAULT_WAIT_TIME = 5  # Reduced for faster element loading
    LOAD_MORE_WAIT_TIME = 1  # Reduced for faster "Load More" interactions
    PAGE_LOAD_WAIT_TIME = 5  # Reduced for faster page loading
    
    # Retry parameters
    MAX_RETRIES = 2  # Reduced from 3
    RETRY_DELAY = 1  # Reduced from 2
    
    # Output file paths - UPDATED FOR NEW DIRECTORY
    CSV_OUTPUT_PATH = str(DATA_DIR / "topstartups_data.csv")
    JSON_OUTPUT_PATH = str(DATA_DIR / "topstartups_data.json")

class HumanLikeBehavior:
    """Class to simulate human-like browsing behavior - EXTREMELY FAST"""
    
    @staticmethod
    def random_delay(min_seconds=0.01, max_seconds=0.05):
        """Wait for a very short random amount of time"""
        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)
        return delay
    
    @staticmethod
    def variable_scroll(driver, config):
        """Scroll with minimal delay and large increments"""
        # Always use fast, chunky scrolling
        scroll_amount = random.randint(
            config.MIN_SCROLL_AMOUNT * 2,  # Double the minimum scroll amount
            config.MAX_SCROLL_AMOUNT * 2  # Double the maximum scroll amount
        )
        
        # Perform the scroll
        driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
        
        # Minimal delay after scrolling
        HumanLikeBehavior.random_delay(0.01, 0.03)
        
        return scroll_amount

class TopStartupsScraper:
    """Class to scrape startup data from topstartups.io - OPTIMIZED VERSION"""
    
    def __init__(self, headless: bool = False):
        """Initialize the scraper with WebDriver setup"""
        self.config = TopStartupsScraperConfig()
        self.driver = self._setup_driver(headless)
        self.wait = WebDriverWait(self.driver, self.config.DEFAULT_WAIT_TIME)
        self.startups: List[Startup] = []
        self.extracted_count = 0
        
    def _setup_driver(self, headless: bool) -> webdriver.Chrome:
        """Set up and configure the Chrome WebDriver - PERFORMANCE OPTIMIZED"""
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        
        # Add common options for stability and performance
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Performance optimizations
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--blink-settings=imagesEnabled=false")  # Disable images
        
        # Add user agent
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36")
        
        # Install and use the WebDriver
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=chrome_options)
    
    def navigate_to_page(self, query: str = None) -> bool:
        """Navigate to the topstartups.io page with the specified query"""
        url = self.config.BASE_URL + (query or self.config.DEFAULT_QUERY)
        logger.info(f"Navigating to: {url}")
        
        try:
            self.driver.get(url)
            
            # Wait for the page to load
            WebDriverWait(self.driver, self.config.PAGE_LOAD_WAIT_TIME).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            
            # Shorter delay
            HumanLikeBehavior.random_delay(0.5, 1.5)
            
            logger.info("Page loaded successfully")
            return True
            
        except (TimeoutException, WebDriverException) as e:
            logger.error(f"Failed to load page: {str(e)}")
            return False

    def find_startup_cards(self) -> List:
        """Find all startup cards with backup strategies - IMPROVED VERSION"""
        for selector in self.config.CARD_SELECTORS:
            try:
                cards = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if cards and len(cards) > 0:
                    return cards
            except:
                continue
        
        # Enhanced XPATH strategy
        try:
            # Look for divs containing elements that match startup card patterns
            cards = self.driver.find_elements(
                By.XPATH, 
                "//*[contains(., 'What they do') or contains(., 'Quick facts') or contains(., 'Funding:')]"
            )
            if cards and len(cards) > 0:
                return cards
        except:
            pass
        
        # JavaScript fallback for complex cases
        try:
            cards = self.driver.execute_script("""
                return Array.from(document.querySelectorAll('*')).filter(el => {
                    const text = el.textContent || '';
                    return (
                        (text.includes('What they do') || text.includes('Quick facts') || text.includes('Funding:')) &&
                        el.querySelectorAll('a').length > 0 &&
                        el.offsetHeight > 100
                    );
                });
            """)
            if cards and len(cards) > 0:
                return cards
        except:
            pass
        
        return []

    def extract_startup_info(self, card) -> Optional[Startup]:
        """Extract information with enhanced error handling - COMPLETE EXTRACTION"""
        startup = Startup()
        
        try:
            # Skip if not visible
            if not card.is_displayed():
                return None
            
            # CRITICAL FIX: Extract all fields properly
            
            # 1. Extract company name
            for selector in self.config.NAME_SELECTORS:
                try:
                    elements = card.find_elements(By.CSS_SELECTOR, selector)
                    for element in elements:
                        text = element.text.strip()
                        if text and len(text) > 1 and ":" not in text and "HQ" not in text and "What they do" not in text:
                            startup.name = text
                            break
                    if startup.name:
                        break
                except:
                    continue
            
            # 2. Extract description (What they do)
            try:
                # Better method using text patterns
                what_they_do_div = None
                
                # Try to find the "What they do:" label
                what_they_do_elements = card.find_elements(By.XPATH, ".//*[contains(text(), 'What they do')]")
                if what_they_do_elements:
                    label_element = what_they_do_elements[0]
                    # Get parent div
                    parent = label_element.find_element(By.XPATH, "./..")
                    
                    # Get text content and remove label
                    text = parent.text
                    if "What they do:" in text:
                        desc = text.split("What they do:")[1].strip()
                        if desc:
                            startup.description = desc
                    
                    # If no text directly, try next sibling
                    if not startup.description:
                        siblings = self.driver.execute_script("""
                            var el = arguments[0];
                            var parent = el.parentElement;
                            var children = Array.from(parent.children);
                            var index = children.indexOf(el);
                            return index < children.length - 1 ? children[index + 1].textContent : '';
                        """, label_element)
                        
                        if siblings:
                            startup.description = siblings.strip()
            except:
                pass
            
            # 3. Extract headquarters location
            try:
                # Look for text pattern with 📍 or HQ:
                hq_elements = card.find_elements(By.XPATH, ".//*[contains(text(), 'HQ:') or contains(text(), '📍')]")
                if hq_elements:
                    hq_text = hq_elements[0].text
                    # Clean up the text
                    hq_clean = hq_text.replace("📍", "").replace("HQ:", "").strip()
                    if hq_clean:
                        startup.headquarters = hq_clean
                    
                    # If not found directly, try parent element
                    if not startup.headquarters:
                        parent = hq_elements[0].find_element(By.XPATH, "./..")
                        parent_text = parent.text
                        if "HQ:" in parent_text:
                            parts = parent_text.split("HQ:")
                            if len(parts) > 1:
                                startup.headquarters = parts[1].strip().split("\n")[0]
            except:
                pass
            
            # 4. Extract employee count and founding year
            try:
                # Look for employee count pattern
                emp_elements = card.find_elements(By.XPATH, ".//*[contains(text(), 'employees')]")
                if emp_elements:
                    startup.employees = emp_elements[0].text.strip()
                
                # Look for founded year pattern
                found_elements = card.find_elements(By.XPATH, ".//*[contains(text(), 'Founded:')]")
                if found_elements:
                    founded_text = found_elements[0].text
                    if "Founded:" in founded_text:
                        startup.founding_year = founded_text.replace("Founded:", "").strip()
            except:
                pass
            
            # 5. Extract funding information
            try:
                # Look for "Funding:" section
                funding_elements = card.find_elements(By.XPATH, ".//*[contains(text(), 'Funding:')]")
                if funding_elements:
                    funding_section = funding_elements[0].find_element(By.XPATH, "./..")
                    funding_text = funding_section.text
                    
                    # Clean up the funding text
                    if "Funding:" in funding_text:
                        startup.funding = funding_text.replace("Funding:", "").strip()
                        
                    # If not found directly, try children
                    if not startup.funding:
                        funding_details = funding_section.find_elements(By.XPATH, ".//div")
                        for detail in funding_details:
                            detail_text = detail.text.strip()
                            if "$" in detail_text:
                                startup.funding = detail_text
                                break
            except:
                pass
            
            # 6. Extract website link
            try:
                # Look for "Check company site" link
                site_links = card.find_elements(By.XPATH, ".//a[contains(text(), 'Check company site') or contains(text(), 'company site')]")
                if site_links:
                    startup.website = site_links[0].get_attribute("href")
                
                # If not found, look for any external link
                if not startup.website:
                    all_links = card.find_elements(By.TAG_NAME, "a")
                    for link in all_links:
                        href = link.get_attribute("href")
                        if href and not href.startswith(self.config.BASE_URL) and "://" in href:
                            startup.website = href
                            break
            except:
                pass
            
            # 7. Check if reviews are available
            try:
                reviews_element = card.find_elements(By.XPATH, ".//a[contains(text(), 'Read reviews') or contains(text(), '⭐')]")
                startup.has_reviews = len(reviews_element) > 0
            except:
                startup.has_reviews = False
            
            # Return startup if we at least found a name
            if startup.name:
                return startup
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error extracting startup data: {str(e)}")
            return None

    def debug_card_structure(self, card, page_number: int):
        """Analyze card structure for debugging purposes"""
        try:
            logger.debug(f"=== Card Analysis on Page {page_number} ===")
            html = card.get_attribute('outerHTML')
            if html:
                logger.debug(f"Card HTML (first 300 chars): {html[:300]}...")
            
            # Try to find company name with multiple methods
            selectors = [
                ('h2', 'CSS'), ('h3', 'CSS'), ('.company-name', 'CSS'),
                ('.startup-name', 'CSS'), ('div[class*="name"]', 'CSS'),
                ('//*[contains(@class, "name")]', 'XPATH')
            ]
            
            for selector, method in selectors:
                try:
                    if method == 'CSS':
                        elements = card.find_elements(By.CSS_SELECTOR, selector)
                    else:
                        elements = card.find_elements(By.XPATH, selector)
                    
                    if elements:
                        for i, el in enumerate(elements[:3]):  # First 3 matches
                            logger.debug(f"Match {i+1} with {selector}: '{el.text}'")
                except:
                    pass
            
            # Extract all text to see what's available
            logger.debug(f"Card text: {card.text[:200]}...")
            logger.debug("=== End Card Analysis ===")
        except Exception as e:
            logger.debug(f"Error analyzing card: {e}")

    def scrape_startups(self, max_startups: int = None, max_pages: int = 100) -> List[Startup]:
        """Main method to scrape startups with optimized speed"""
        if not self.navigate_to_page():
            logger.error("Failed to navigate to the target page. Aborting.")
            return []
        
        logger.info("Beginning startup extraction")
        
        seen_startups = set()
        page_number = 1
        consecutive_empty_pages = 0
        max_consecutive_empty_pages = 1  # Stop after 1 empty page for speed
        
        while (not max_startups or len(self.startups) < max_startups) and page_number <= max_pages:
            cards = self.find_startup_cards()
            logger.info(f"Found {len(cards)} startup cards on current view")
            
            startups_before = len(self.startups)
            
            for card in cards:
                try:
                    self.driver.execute_script(
                        "arguments[0].scrollIntoView({block: 'center'});", 
                        card
                    )
                    HumanLikeBehavior.random_delay(0.01, 0.03)  # Minimal delay
                    
                    startup = self.extract_startup_info(card)
                    
                    if startup and startup.name:
                        if startup.name in seen_startups:
                            continue
                        seen_startups.add(startup.name)
                        self.startups.append(startup)
                        self.extracted_count += 1
                        logger.info(f"Extracted startup #{self.extracted_count}: {startup.name}")
                        
                        if max_startups and len(self.startups) >= max_startups:
                            return self.startups
                except Exception as e:
                    logger.warning(f"Error processing card: {str(e)}")
                    continue
            
            startups_extracted = len(self.startups) - startups_before
            logger.info(f"Extracted {startups_extracted} startups from current page")
            
            if startups_extracted == 0:
                consecutive_empty_pages += 1
                if consecutive_empty_pages >= max_consecutive_empty_pages:
                    logger.info("Stopping due to consecutive empty pages")
                    break
            else:
                consecutive_empty_pages = 0
            
            page_number += 1
            next_page_url = self._construct_pagination_url(page_number)
            logger.info(f"Navigating to page {page_number}: {next_page_url}")
            
            try:
                self.driver.get(next_page_url)
                WebDriverWait(self.driver, self.config.PAGE_LOAD_WAIT_TIME).until(
                    lambda d: d.execute_script("return document.readyState") == "complete"
                )
            except Exception as e:
                logger.error(f"Error navigating to page {page_number}: {str(e)}")
                break
        
        logger.info(f"Extraction complete. Total startups extracted: {len(self.startups)}")
        return self.startups

    def _construct_pagination_url(self, page_number: int) -> str:
        """Build pagination URL based on current URL or default query"""
        try:
            current_url = self.driver.current_url
            if "page=" in current_url:
                # Replace existing page parameter
                return re.sub(r'page=\d+', f'page={page_number}', current_url)
            elif "?" in current_url:
                # Add page parameter to existing query
                return f"{current_url}&page={page_number}"
            else:
                # Add as first parameter
                return f"{current_url}?page={page_number}"
        except:
            # Fallback to default query with page number
            query = self.config.DEFAULT_QUERY
            if "?" in query:
                return f"{self.config.BASE_URL}{query}&page={page_number}"
            else:
                return f"{self.config.BASE_URL}{query}?page={page_number}"

    def save_results(self) -> Tuple[str, str]:
        """Save the extracted startup data to CSV and JSON files"""
        if not self.startups:
            logger.warning("No startups to save")
            return None, None
        
        try:
            # Save to CSV
            with open(self.config.CSV_OUTPUT_PATH, 'w', newline='', encoding='utf-8') as csv_file:
                fieldnames = ['name', 'description', 'headquarters', 'funding', 'website', 'has_reviews', 
                             'employees', 'founding_year']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                
                for startup in self.startups:
                    # Convert dataclass to dict and write to CSV
                    startup_dict = {k: v for k, v in asdict(startup).items() if k in fieldnames}
                    writer.writerow(startup_dict)
            
            logger.info(f"Saved data to CSV: {self.config.CSV_OUTPUT_PATH}")
            
            # Save to JSON
            startup_list = [asdict(startup) for startup in self.startups]
            with open(self.config.JSON_OUTPUT_PATH, 'w', encoding='utf-8') as json_file:
                json.dump(startup_list, json_file, indent=4, ensure_ascii=False)
            
            logger.info(f"Saved data to JSON: {self.config.JSON_OUTPUT_PATH}")
            
            return self.config.CSV_OUTPUT_PATH, self.config.JSON_OUTPUT_PATH
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return None, None

    def cleanup(self):
        """Close the WebDriver and perform cleanup"""
        try:
            self.driver.quit()
            logger.info("WebDriver closed successfully")
        except:
            logger.warning("Error closing WebDriver")

def update_latest_data_links(current_date):
    """Update symbolic links to latest data files"""
    latest_csv = DATA_DIR / "latest_data.csv"
    latest_json = DATA_DIR / "latest_data.json"
    
    # Source files
    source_csv = DATA_DIR / current_date / "topstartups_data.csv"
    source_json = DATA_DIR / current_date / "topstartups_data.json"
    
    # Update symlinks or copy files
    if os.path.exists(latest_csv):
        os.remove(latest_csv)
    if os.path.exists(latest_json):
        os.remove(latest_json)
        
    try:
        os.symlink(source_csv, latest_csv)
        os.symlink(source_json, latest_json)
    except:
        import shutil
        shutil.copy2(source_csv, latest_csv)
        shutil.copy2(source_json, latest_json)
    
    logger.info(f"Updated latest data links to {current_date} version")

def check_for_changes(current_date):
    """Compare current data with previous data to detect changes"""
    date_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir() and d.name != current_date])
    
    if not date_dirs:
        logger.info("No previous data found for comparison")
        return
    
    prev_date_dir = date_dirs[-1]
    prev_json = prev_date_dir / "topstartups_data.json"
    current_json = DATA_DIR / current_date / "topstartups_data.json"
    
    if not prev_json.exists() or not current_json.exists():
        logger.warning("Cannot compare data: missing files")
        return
    
    with open(prev_json, 'r') as f:
        prev_data = json.load(f)
    with open(current_json, 'r') as f:
        current_data = json.load(f)
    
    prev_map = {s['name']: s for s in prev_data}
    current_map = {s['name']: s for s in current_data}
    
    new_startups = [s for s in current_data if s['name'] not in prev_map]
    updated_startups = []
    for startup in current_data:
        name = startup['name']
        if name in prev_map:
            prev = prev_map[name]
            for field in ['funding', 'employees', 'website']:
                if startup.get(field) != prev.get(field) and startup.get(field) and prev.get(field):
                    updated_startups.append({
                        'name': name,
                        'field': field,
                        'old_value': prev.get(field),
                        'new_value': startup.get(field)
                    })
    
    changes_file = DATA_DIR / current_date / "changes_report.json"
    changes_report = {
        'date': current_date,
        'previous_date': prev_date_dir.name,
        'new_startups_count': len(new_startups),
        'updated_startups_count': len(updated_startups),
        'new_startups': new_startups,
        'updated_startups': updated_startups
    }
    
    with open(changes_file, 'w') as f:
        json.dump(changes_report, f, indent=4)
    
    logger.info(f"Change detection completed. Found {len(new_startups)} new and {len(updated_startups)} updated startups.")

def scheduled_data_collection():
    """Run the scraper on a schedule and manage data versioning"""
    def job():
        current_date = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Starting scheduled data collection for {current_date}")
        
        version_dir = DATA_DIR / current_date
        version_dir.mkdir(exist_ok=True)
        
        TopStartupsScraperConfig.CSV_OUTPUT_PATH = str(version_dir / "topstartups_data.csv")
        TopStartupsScraperConfig.JSON_OUTPUT_PATH = str(version_dir / "topstartups_data.json")
        
        scraper = TopStartupsScraper(headless=True)
        
        try:
            scraper.scrape_startups(max_startups=None)
            scraper.save_results()
            update_latest_data_links(current_date)
            check_for_changes(current_date)
            logger.info(f"Scheduled data collection completed for {current_date}")
        except Exception as e:
            logger.error(f"Scheduled job failed: {str(e)}")
        finally:
            scraper.cleanup()
    
    schedule.every().day.at("03:00").do(job)
    logger.info("Scheduler started. Data collection will run daily at 03:00")
    
    while True:
        schedule.run_pending()
        time.sleep(60)

def main():
    """Main function to run the scraper with scheduling"""
    logger.info("Initializing TopStartups.io scraper")
    
    while True:
        scraper = TopStartupsScraper(headless=False)  # Set to True for headless mode
        
        try:
            # Start the scraping process
            start_time = datetime.now()
            logger.info(f"Starting new data collection cycle at {start_time}")
            
            # Run with date versioning
            current_date = datetime.now().strftime("%Y-%m-%d")
            version_dir = DATA_DIR / current_date
            version_dir.mkdir(exist_ok=True)
            
            # Configure versioned output paths
            TopStartupsScraperConfig.CSV_OUTPUT_PATH = str(version_dir / "topstartups_data.csv")
            TopStartupsScraperConfig.JSON_OUTPUT_PATH = str(version_dir / "topstartups_data.json")
            
            # Perform the scrape
            scraper.scrape_startups(max_startups=None)
            scraper.save_results()
            
            # Update latest symlinks
            update_latest_data_links(current_date)
            
            # Calculate next run time
            end_time = datetime.now()
            duration = end_time - start_time
            next_run = end_time + timedelta(hours=24)
            logger.info(f"Collection cycle completed in {duration}. Next run at {next_run}")
            
        except Exception as e:
            logger.error(f"Error during scraping cycle: {str(e)}")
            
        finally:
            # Ensure proper cleanup
            scraper.cleanup()
            
            # Calculate sleep time (24 hours from start time)
            sleep_time = (24 * 3600) - duration.total_seconds()
            if sleep_time < 0:
                sleep_time = 0
                
            logger.info(f"Sleeping for {sleep_time/3600:.2f} hours until next cycle")
            time.sleep(sleep_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TopStartups.io Scraper')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    args = parser.parse_args()
    
    if args.once:
        # Single run mode
        scraper = TopStartupsScraper(headless=False)
        try:
            scraper.scrape_startups(max_startups=None)
            scraper.save_results()
        finally:
            scraper.cleanup()
    else:
        # Start continuous collection
        logger.info("Starting continuous data collection service")
        logger.info("Data will be collected every 24 hours")
        main()
