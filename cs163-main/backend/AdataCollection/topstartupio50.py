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
import shutil

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

# Define paths for data storage
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_FOLDER = os.path.join(PROJECT_ROOT, "JSONFolder")
if not os.path.exists(JSON_FOLDER):
    os.makedirs(JSON_FOLDER, exist_ok=True)

# Keep original data dir for logs and local data
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
    
    # Output file paths - UPDATED FOR JSONFOLDER
    CSV_OUTPUT_PATH = str(os.path.join(JSON_FOLDER, "topstartupio50.csv"))
    JSON_OUTPUT_PATH = str(os.path.join(JSON_FOLDER, "topstartupio50.json"))

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
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_experimental_option("detach", True)
            chrome_options.add_argument("--disable-backgrounding-occluded-windows")
            chrome_options.add_argument("--disable-renderer-backgrounding")
        
        # Common options for stability and performance
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        # Performance optimizations (ensure these are kept or added if missing)
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--blink-settings=imagesEnabled=false")  # Disable images
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36")
        
        # Install and use the WebDriver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        time.sleep(1) # Added delay after driver instantiation
        return driver
    
    def navigate_to_page(self, query: str = None) -> bool:
        """Navigate to the topstartups.io page with the specified query"""
        url = self.config.BASE_URL + (query or self.config.DEFAULT_QUERY)
        logger.info(f"Navigating to: {url}")
        
        time.sleep(0.5) # Small delay before self.driver.get()
        try:
            self.driver.get(url)
            time.sleep(1.5) # Ensured delay is 1.5s
            
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

    def save_results(self):
        """Save results to JSONFolder and historical copies"""
        if not self.startups:
            return None, None

        # Save CSV (overwrite)
        csv_path = self.config.CSV_OUTPUT_PATH
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            # Include all fields including category and others
            writer = csv.DictWriter(f, fieldnames=['name', 'description', 'headquarters', 'funding', 
                                               'website', 'has_reviews', 'category', 'employees', 
                                               'founding_year', 'other_details'])
            writer.writeheader()
            
            # Convert dataclass objects to dictionaries, handling nested other_details
            rows = []
            for startup in self.startups:
                row_dict = asdict(startup)
                
                # Convert other_details to string if present
                if row_dict['other_details']:
                    row_dict['other_details'] = json.dumps(row_dict['other_details'])
                else:
                    row_dict['other_details'] = ""
                    
                rows.append(row_dict)
                
            writer.writerows(rows)
        logger.info(f"Saved CSV data to {csv_path}")

        # Save JSON (overwrite)
        json_path = self.config.JSON_OUTPUT_PATH
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(startup) for startup in self.startups], f, indent=4)
        logger.info(f"Saved JSON data to {json_path}")

        # Create historical data directory
        current_date = datetime.now().strftime("%Y-%m-%d")
        historical_dir = os.path.join(PROJECT_ROOT, "data_archive", current_date)
        os.makedirs(historical_dir, exist_ok=True)

        # Save historical copies
        historical_csv = os.path.join(historical_dir, "topstartupio50.csv")
        historical_json = os.path.join(historical_dir, "topstartupio50.json")
        
        try:
            shutil.copy2(csv_path, historical_csv)
            shutil.copy2(json_path, historical_json)
            logger.info(f"Saved historical copies to {historical_dir}")
        except Exception as e:
            logger.error(f"Error saving historical copies: {e}")

        return csv_path, json_path

    def cleanup(self):
        """Close the WebDriver and perform cleanup"""
        try:
            self.driver.quit()
            logger.info("WebDriver closed successfully")
        except Exception as e:
            logger.warning(f"Error closing WebDriver: {str(e)}")

# --- Main Execution & Scheduling Logic (Adapted from ExperimentationsNotUsefull version and fundraiserstartup50.py) ---
def run_scraper_once(headless_arg=True): # Default to headless for single run
    """Runs the scraper a single time."""
    logger.info("Running TopStartupsScraper once.")
    scraper = TopStartupsScraper(headless=headless_arg)
    try:
        scraper.scrape_startups(max_startups=None) # Scrape all available if no limit
        scraper.save_results()
    except Exception as e:
        logger.error(f"Error during single run: {e}", exc_info=True)
    finally:
        scraper.cleanup()
    logger.info("Single run finished.")

def scheduled_job():
    logger.info(f"Starting scheduled data collection for TopStartups.io at {datetime.now()}")
    run_scraper_once(headless_arg=True) # Scheduled jobs should always be headless
    logger.info(f"Scheduled TopStartups.io job finished at {datetime.now()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TopStartups.io Scraper')
    # parser.add_argument('--once', action='store_true', help='Run once and exit')
    # Add a headless flag, defaulting to True (headless) if --schedule is used or if --once is not explicitly non-headless
    parser.add_argument('--no-headless', action='store_false', dest='headless', help='Run with a visible browser window (primarily for debugging single runs)')
    parser.set_defaults(headless=True)

    args = parser.parse_args()

    if False:
        run_scraper_once(headless_arg=args.headless)
    else:
        # Default to scheduling if --once is not provided
        logger.info("Scheduler started for TopStartups.io. Will run daily at 03:00 (local time).")
        schedule.every().day.at("03:00").do(scheduled_job)
        # Run once immediately as well before starting schedule loop
        scheduled_job() 
        while True:
            schedule.run_pending()
            time.sleep(60) # Check every minute
