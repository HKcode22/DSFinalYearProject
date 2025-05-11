import os
import time
import logging
import csv
import json
import random
import schedule
import shutil
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    NoSuchElementException, 
    TimeoutException, 
    StaleElementReferenceException,
    ElementClickInterceptedException,
    ElementNotInteractableException,
    WebDriverException
)
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fundraise_insider_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants for optimization
MAX_PAGES_ESTIMATE = 900  # Based on previous runs showing ~56 pages
TIMEOUT_DURATION = 15    # Reduced timeout duration
MIN_DELAY = 0.1         # Minimum delay for human-like behavior
MAX_DELAY = 0.5         # Maximum delay for human-like behavior
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fundraise_insider_data")

class FundraiseInsiderScraper:
    def __init__(self, headless=False):
        """Initialize the scraper with configurable headless option."""
        self.url = "https://fundraiseinsider.com/blog/recently-funded-startups-san-francisco/#showform-238066"
        self.data = []
        self.extracted_companies = set()
        self.current_page = 1
        self.headless = headless
        
        # Create data directory at project root level (JSONFolder)
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_root = os.path.join(self.project_root, "JSONFolder")
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)
            logger.info(f"Created main data directory at {self.data_root}")
        
        self.stats_file = os.path.join(self.data_root, "fundraise_insider_stats.json")
        self.stats = self.load_stats()
        self.driver = None
        self.max_retries = 3
        self.page_data_counts = {}
        self.empty_page_consecutive_count = 0
        self.max_empty_pages_allowed = 2
        self.ordered_fields = [
            "Company",
            "Total_Employees",
            "Industry",
            "Website",
            "Funding_Date",
            "Funding_Type",
            "Funding_Amount_USD",
            "Headquarters",
            "Extraction_Time",
            "Page"
        ]

    def load_stats(self):
        """Load statistics from previous runs if available."""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading stats: {e}")
                return {"last_run": None, "total_extractions": 0, "companies_extracted": 0}
        return {"last_run": None, "total_extractions": 0, "companies_extracted": 0}
    
    def save_stats(self):
        """Save statistics from current run."""
        self.stats["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.stats["total_extractions"] += 1
        self.stats["companies_extracted"] = len(self.data)
        self.stats["pages_processed"] = self.current_page
        self.stats["companies_per_page"] = self.page_data_counts
        
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving stats: {e}")
    
    def setup_driver(self):
        """Set up the Selenium WebDriver with focus retention capabilities."""
        logger.info("Setting up Chrome WebDriver...")
        options = Options()
        if self.headless:
            options.add_argument("--headless=new")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
        
        # Add window focus handling
        options.add_argument("--disable-backgrounding-occluded-windows")
        options.add_argument("--disable-renderer-backgrounding")
        options.add_experimental_option("detach", True)
        
        # Prevent Chrome from entering "idle" state
        options.add_argument("--disable-features=TranslateUI")
        options.add_argument("--disable-hang-monitor")
        
        # Performance optimization options
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-popup-blocking")
        
        # Add user agent to appear more human-like
        options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36")
        
        try:
            self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
            time.sleep(1.5) # Explicit delay after driver instantiation
            self.driver.set_page_load_timeout(40)  # Increased timeout
            self.driver.set_script_timeout(30)  # Add script timeout
            logger.info("WebDriver setup completed successfully")
            return True
        except Exception as e:
            logger.error(f"WebDriver setup failed: {e}")
            return False

    def ensure_window_focused(self):
        """Ensure the browser window is focused."""
        try:
            self.driver.switch_to.window(self.driver.current_window_handle)
            self.driver.execute_script("window.focus();")
            return True
        except Exception as e:
            logger.warning(f"Error ensuring window focus: {e}")
            return False
    
    def quick_human_delay(self, min_seconds=0.3, max_seconds=1.2):
        """Add a short randomized delay to simulate human behavior but maintain speed."""
        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)
    
    def handle_all_overlays(self):
        """Advanced overlay and popup handling system."""
        try:
            # First try direct approach with known overlay classes
            overlay_selectors = [
                "//div[contains(@class, 'conv-wrap')]",
                "//div[contains(@class, 'popup')]",
                "//div[contains(@class, 'modal')]",
                "//div[contains(@class, 'overlay')]"
            ]
            
            for selector in overlay_selectors:
                elements = self.driver.find_elements(By.XPATH, selector)
                for element in elements:
                    if element.is_displayed():
                        logger.info(f"Found overlay using selector: {selector}")
                        try:
                            # Try to find close buttons within the overlay
                            close_buttons = element.find_elements(By.XPATH, 
                                ".//button | .//a[contains(@class, 'close')] | .//span[contains(@class, 'close')]")
                            
                            if close_buttons:
                                for btn in close_buttons:
                                    if btn.is_displayed():
                                        logger.info("Clicking close button on overlay")
                                        self.driver.execute_script("arguments[0].click();", btn)
                                        self.quick_human_delay()
                                        break
                            else:
                                # If no close button  try to remove the overlay with JavaScript
                                logger.info(f"No close button found, removing overlay with JavaScript")
                                self.driver.execute_script("""
                                    var element = arguments[0];
                                    if(element) {
                                        element.style.display = 'none';
                                        element.classList.add('hidden');
                                        element.setAttribute('aria-hidden', 'true');
                                    }
                                """, element)
                                self.quick_human_delay()
                        except Exception as e:
                            logger.debug(f"Error handling overlay element: {e}")
            
            # Send Escape key as a last resort
            try:
                actions = ActionChains(self.driver)
                actions.send_keys(Keys.ESCAPE).perform()
                self.quick_human_delay()
            except Exception:
                pass
                
        except Exception as e:
            logger.warning(f"Error in overlay handling: {e}")
    
    def efficient_scroll(self):
        """Perform quick but human-like scrolling."""
        logger.info("Performing efficient scrolling...")
        try:
            # Get page height
            total_height = self.driver.execute_script("return document.body.scrollHeight")
            viewport_height = self.driver.execute_script("return window.innerHeight")
            
            # Scroll down in chunks
            scroll_points = [
                total_height * 0.33,  # Scroll to 1/3
                total_height * 0.67,  # Scroll to 2/3
                total_height         # Scroll to bottom
            ]
            
            for point in scroll_points:
                # Add some randomness to the scroll position
                jitter = random.uniform(-50, 50)
                position = min(total_height, point + jitter)
                
                # Scroll with smooth behavior
                self.driver.execute_script(f"window.scrollTo({{top: {position}, behavior: 'smooth'}});")
                self.quick_human_delay()
            
            # Scroll back to the pagination area (typically at the bottom)
            self.driver.execute_script(f"window.scrollTo({{top: {total_height * 0.9}, behavior: 'smooth'}});")
            self.quick_human_delay()
            
            logger.info("Scrolling completed")
        except Exception as e:
            logger.warning(f"Error during scrolling: {e}")
    
    def navigate_to_page(self):
        """Navigate to the target URL and attempt to access detailed view."""
        logger.info(f"Navigating to {self.url}")
        try:
            self.driver.get(self.url)
            time.sleep(1.5) # Explicit delay after navigating
            
            # Wait for initial page load
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Handle initial overlays
            self.handle_all_overlays()
            
            # Try to find and click the detailed view link
            try:
                detailed_view_selectors = [
                    "//a[contains(text(), 'download') and contains(text(), 'full')]",
                    "//a[contains(text(), 'detailed') and contains(text(), 'view')]",
                    "//a[contains(text(), 'complete') and contains(text(), 'list')]"
                ]
                
                for selector in detailed_view_selectors:
                    try:
                        detailed_link = WebDriverWait(self.driver, 5).until(
                            EC.presence_of_element_located((By.XPATH, selector))
                        )
                        if detailed_link.is_displayed():
                            logger.info("Found detailed view link")
                            self.driver.execute_script("arguments[0].click();", detailed_link)
                            
                            # Wait for detailed view to load
                            WebDriverWait(self.driver, 15).until(
                                EC.presence_of_element_located((By.XPATH, 
                                    "//th[contains(text(), 'Funding Amount')] | //div[contains(text(), 'Funding Amount')]"))
                            )
                            logger.info("Successfully accessed detailed view")
                            break
                    except Exception:
                        continue
            except Exception as e:
                logger.warning(f"Could not access detailed view: {e}")
            
            # Perform scrolling
            self.efficient_scroll()
            
            # Final overlay check
            self.handle_all_overlays()
            
            return True
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            return False
    
    def verify_page_changed(self):
        """Verify that the page has actually changed using multiple methods."""
        try:
            # Wait for page load
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Try to get current URL with retry
            current_url = None
            for attempt in range(3):
                try:
                    current_url = self.driver.current_url
                    break
                except Exception as e:
                    logger.debug(f"Error getting URL on attempt {attempt+1}: {e}")
                    time.sleep(1)
            
            if current_url:
                # Look for page indicators in URL
                expected_next_page = self.current_page + 1
                if (f"page={expected_next_page}" in current_url or 
                    f"/page{expected_next_page}" in current_url):
                    logger.info(f"URL confirms we're on page {expected_next_page}")
                    return True
            
            # Use JavaScript to check the DOM 
            # works better when window not focused)
            page_changed = self.driver.execute_script("""
                var nextPage = arguments[0];
                // Check URL and active indicators
                var activeElements = document.querySelectorAll('a.current, .pagination .active');
                for (var i = 0; i < activeElements.length; i++) {
                    if (activeElements[i].textContent.trim() == nextPage) {
                        return true;
                    }
                }
                return document.querySelector('table') !== null;
            """, str(self.current_page + 1))
            
            if page_changed:
                logger.info("JavaScript verification confirmed page changed")
                return True
            
            return True  # Fail open to prevent false stops
            
        except Exception as e:
            logger.warning(f"Error verifying page change: {e}")
            return True
    
    def go_to_next_page(self):
        """Navigate to next page with enhanced focus handling."""
        logger.info(f"Attempting to navigate from page {self.current_page}")
        
        # Ensure window focus
        self.ensure_window_focused()
        
        # Handle overlays and scroll to pagination area
        self.handle_all_overlays()
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.9);")
        self.quick_human_delay(1.5, 2.5)  # Longer delay for stability
        
        # Try JavaScript-first approach for next button
        try:
            next_button = self.driver.execute_script("""
                var buttons = Array.from(document.querySelectorAll('a')).filter(link => {
                    var text = link.textContent.trim();
                    return (text === '›' || text === '»' || text === '>' || 
                            text === 'Next' || link.classList.contains('next'));
                }).find(link => {
                    var style = window.getComputedStyle(link);
                    return style.display !== 'none' && style.visibility !== 'hidden';
                });
                if (buttons) buttons.style.border = '3px solid green';
                return buttons;
            """)
            
            if next_button:
                self.driver.execute_script("arguments[0].click();", next_button)
                time.sleep(3)  # Wait for page load
                
                if self.verify_page_changed():
                    self.current_page += 1
                    self.handle_all_overlays()
                    return True
        except Exception as e:
            logger.debug(f"JavaScript navigation failed: {e}")
        
        # Fallback to traditional Selenium approach
        next_button_selectors = [
            "//a[contains(text(), '>')]",
            "//a[contains(text(), '›')]",
            "//a[contains(@class, 'next')]"
        ]
        
        for selector in next_button_selectors:
            try:
                buttons = self.driver.find_elements(By.XPATH, selector)
                for btn in buttons:
                    if btn.is_displayed():
                        self.driver.execute_script("arguments[0].click();", btn)
                        time.sleep(3)
                        
                        if self.verify_page_changed():
                            self.current_page += 1
                            self.handle_all_overlays()
                            return True
            except Exception as e:
                logger.debug(f"Selector {selector} failed: {e}")
        
        return False
    
    def extract_table_data(self):
        """Extract data with improved funding amount detection."""
        logger.info(f"Extracting data from page {self.current_page}")
        
        self.handle_all_overlays()
        companies_on_page = []
        initial_company_count = len(self.data)
        
        for attempt in range(3):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt} for data extraction")
                    self.quick_human_delay(1.0, 2.0)
                    if attempt == 2:
                        logger.info("Refreshing page for final extraction attempt")
                        self.driver.refresh()
                        self.quick_human_delay(2.0, 3.0)
                        self.handle_all_overlays()
                
                logger.info("Trying standard table extraction")
                
                # First try to find column indices for funding amount
                header_row = None
                funding_amount_idx = -1
                headquarters_idx = -1
                
                try:
                    headers = self.driver.find_elements(By.TAG_NAME, "th")
                    if not headers:
                        headers = self.driver.find_elements(By.XPATH, "//div[contains(@class, 'header')]//div")
                    
                    for idx, header in enumerate(headers):
                        header_text = header.text.strip().lower()
                        if 'funding amount' in header_text or 'amount' in header_text:
                            funding_amount_idx = idx
                            logger.info(f"Found funding amount column at index {idx}")
                        elif 'headquarters' in header_text or 'location' in header_text:
                            headquarters_idx = idx
                            logger.info(f"Found headquarters column at index {idx}")
                except Exception as e:
                    logger.debug(f"Error finding column headers: {e}")
                
                # Extract table data
                tables = self.driver.find_elements(By.TAG_NAME, "table")
                if not tables:
                    tables = self.driver.find_elements(By.XPATH, 
                        "//div[contains(@class, 'table') or contains(@class, 'grid')]")
                
                if tables:
                    for table in tables:
                        rows = table.find_elements(By.TAG_NAME, "tr")
                        if not rows:
                            rows = table.find_elements(By.XPATH, ".//div[contains(@class, 'row')]")
                        
                        if rows:
                            logger.info(f"Found {len(rows)} potential data rows")
                            
                            for row in rows:
                                try:
                                    cells = row.find_elements(By.TAG_NAME, "td")
                                    if not cells or len(cells) < 5:
                                        cells = row.find_elements(By.XPATH, 
                                            ".//div[contains(@class, 'cell') or contains(@class, 'col')]")
                                    
                                    if cells and len(cells) >= 5:
                                        company_name = cells[0].text.strip()
                                        if not company_name:
                                            continue
                                        
                                        # Get funding amount if column exists
                                        funding_amount = "N/A"
                                        if funding_amount_idx >= 0 and funding_amount_idx < len(cells):
                                            amount_text = cells[funding_amount_idx].text.strip()
                                            if amount_text:
                                                funding_amount = amount_text
                                        
                                        # Get headquarters if column exists
                                        headquarters = "San Francisco"  # Default
                                        if headquarters_idx >= 0 and headquarters_idx < len(cells):
                                            hq_text = cells[headquarters_idx].text.strip()
                                            if hq_text:
                                                headquarters = hq_text
                                        
                                        # Create company data record
                                        company_data = {
                                            "Company": company_name,
                                            "Total_Employees": cells[1].text.strip() if len(cells) > 1 else "N/A",
                                            "Industry": cells[2].text.strip() if len(cells) > 2 else "N/A",
                                            "Website": cells[3].text.strip() if len(cells) > 3 else "N/A",
                                            "Funding_Date": cells[4].text.strip() if len(cells) > 4 else "N/A",
                                            "Funding_Type": cells[5].text.strip() if len(cells) > 5 else "N/A",
                                            "Funding_Amount_USD": funding_amount,
                                            "Headquarters": headquarters,
                                            "Extraction_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                            "Page": self.current_page
                                        }
                                        
                                        company_key = f"{company_name}-{company_data['Funding_Date']}"
                                        
                                        if company_key not in self.extracted_companies:
                                            self.data.append(company_data)
                                            self.extracted_companies.add(company_key)
                                            companies_on_page.append(company_name)
                                            logger.info(f"Extracted: {company_name} - {company_data['Funding_Date']}")
                                            
                                            # Log if funding amount was found
                                            if funding_amount != "N/A":
                                                logger.info(f"Found funding amount: {funding_amount}")
                                except Exception as e:
                                    logger.debug(f"Error extracting row data: {e}")
                                    continue
                
                if len(companies_on_page) > 0:
                    break
                    
            except Exception as e:
                logger.warning(f"Extraction attempt {attempt+1} failed: {e}")
        
        self.page_data_counts[str(self.current_page)] = companies_on_page
        new_companies = len(self.data) - initial_company_count
        
        if new_companies > 0:
            logger.info(f"Successfully extracted {new_companies} new companies from page {self.current_page}")
            self.empty_page_consecutive_count = 0
            return True
        else:
            self.empty_page_consecutive_count += 1
            logger.warning(f"No new companies found on page {self.current_page}. Empty page count: {self.empty_page_consecutive_count}")
            return False
    
    def is_truly_last_page(self):
        """Carefully determine if this is truly the last page using multiple indicators."""
        try:
            # Method 1: Check if next button exists but is disabled
            next_buttons = self.driver.find_elements(By.XPATH, 
                "//a[contains(@class, 'next') or contains(text(), '»') or contains(text(), '›') or contains(text(), '>')]")
            
            # If no next buttons exist at all, it might be the last page
            if not next_buttons:
                logger.info("No next page buttons found at all - likely last page")
                return True
            
            # Check if next buttons exist but are all disabled
            all_disabled = True
            for btn in next_buttons:
                if btn.is_displayed():
                    disabled_attr = btn.get_attribute("disabled")
                    aria_disabled = btn.get_attribute("aria-disabled")
                    class_attr = btn.get_attribute("class") or ""
                    
                    if not (disabled_attr == "true" or aria_disabled == "true" or "disabled" in class_attr):
                        all_disabled = False
                        break
            
            if all_disabled and next_buttons:
                logger.info("All next buttons are disabled - confirmed last page")
                return True
            
            # Method 2: Check for last page indicators in URL or active page highlighting
            current_url = self.driver.current_url
            if "last" in current_url or "end" in current_url:
                logger.info("URL indicates last page")
                return True
            
            # Method 3: Check for consecutive empty pages
            if self.empty_page_consecutive_count >= self.max_empty_pages_allowed:
                logger.info(f"Found {self.empty_page_consecutive_count} consecutive empty pages - assuming end of data")
                return True
            
            return False
        except Exception as e:
            logger.warning(f"Error checking if last page: {e}")
            return False
    
    def save_data(self):
        """Save the extracted data with specific field order."""
        if not self.data:
            logger.warning("No data to save")
            return None

        # Save as CSV (overwrite)
        csv_filename = os.path.join(self.data_root, "fundraisestartup50.csv") 
        try:
            with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.ordered_fields)
                writer.writeheader()
                for record in self.data:
                    row = {field: record.get(field, "N/A") for field in self.ordered_fields}
                    writer.writerow(row)
            logger.info(f"Successfully saved CSV data to {csv_filename}")
        except Exception as e:
            logger.error(f"Error saving CSV data: {e}")

        # Save as JSON (overwrite)
        json_filename = os.path.join(self.data_root, "fundraisestartup50.json")
        try:
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump({
                    "extraction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "total_companies": len(self.data),
                    "pages_processed": self.current_page,
                    "companies": self.data
                }, f, indent=4)
            logger.info(f"Successfully saved JSON data to {json_filename}")
        except Exception as e:
            logger.error(f"Error saving JSON data: {e}")

        # Create historical data directory for data archiving
        current_date = datetime.now().strftime("%Y-%m-%d")
        historical_dir = os.path.join(self.project_root, "data_archive", current_date)
        os.makedirs(historical_dir, exist_ok=True)

        # Save historical copies
        historical_csv = os.path.join(historical_dir, "fundraisestartup50.csv")
        historical_json = os.path.join(historical_dir, "fundraisestartup50.json")
        
        try:
            shutil.copy2(csv_filename, historical_csv)
            shutil.copy2(json_filename, historical_json)
            logger.info(f"Successfully saved historical data copies in {historical_dir}")
        except Exception as e:
            logger.error(f"Error saving historical copies: {e}")

    def run_extraction(self):
        """Run the complete extraction process with improved error handling."""
        logger.info("Starting extraction process")
        start_time = datetime.now()
        
        try:
            # Reset data for this run
            self.data = []
            self.extracted_companies = set()
            self.current_page = 1
            self.page_data_counts = {}
            self.empty_page_consecutive_count = 0
            
            # Setup WebDriver
            if not self.setup_driver():
                logger.error("Failed to set up WebDriver, aborting extraction")
                return False
            
            try:
                # Navigate to page and extract data
                if self.navigate_to_page():
                    # Extract data from first page
                    first_page_result = self.extract_table_data()
                    
                    if not first_page_result:
                        logger.warning("Failed to extract data from the first page, attempting alternative methods")
                        
                        # Check if there's a download link as alternative
                        try:
                            download_link = self.driver.find_element(By.XPATH, 
                                "//a[contains(text(), 'download') and contains(text(), 'full')]")
                            if download_link:
                                logger.info(f"Found download link for full data list: {download_link.get_attribute('href')}")
                                logger.info("Consider using this link as an alternative data source")
                        except NoSuchElementException:
                            pass
                    
                    # Continue to next pages until no more
                    max_pages_safety = 200  # Increased limit to catch all pages
                    while self.go_to_next_page() and self.current_page < max_pages_safety:
                        extraction_result = self.extract_table_data()
                        
                        # Check if we've hit too many consecutive empty pages
                        if self.empty_page_consecutive_count >= self.max_empty_pages_allowed:
                            logger.info(f"Stopping after {self.empty_page_consecutive_count} consecutive empty pages")
                            break
                    
                    # Check if we hit the safety limit
                    if self.current_page >= max_pages_safety:
                        logger.warning(f"Reached safety limit of {max_pages_safety} pages, stopping extraction")
                    
                    # Save the data
                    filename = self.save_data()
                    
                    # Update stats
                    self.save_stats()
                    
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    logger.info(f"Extraction completed in {duration:.2f} seconds")
                    logger.info(f"Extracted {len(self.data)} companies from {self.current_page} pages")
                    
                    # Analyze collected data
                    unique_companies = len(set(company['Company'] for company in self.data))
                    logger.info(f"Found {unique_companies} unique companies")
                    
                    # Find most recent funding dates
                    funding_dates = [company['Funding_Date'] for company in self.data 
                                    if company['Funding_Date'] and company['Funding_Date'] != 'N/A']
                    if funding_dates:
                        logger.info(f"Most recent funding dates: {sorted(set(funding_dates))[:5]}")
                    
                    return True
                else:
                    logger.error("Failed to navigate to the target page")
                    return False
            finally:
                # Always close the driver
                if self.driver:
                    self.driver.quit()
                    logger.info("WebDriver closed")
        except WebDriverException as e:
            logger.error(f"WebDriver error during extraction: {e}")
            if self.driver:
                try:
                    self.driver.quit()
                except:
                    pass
            return False
        except Exception as e:
            logger.error(f"Unhandled error during extraction: {e}")
            if self.driver:
                try:
                    self.driver.quit()
                except:
                    pass
            return False
    
    def schedule_extraction(self):
        """Schedule the extraction to run every 24 hours."""
        logger.info("Setting up scheduled extraction (every 24 hours)")
        
        # Run immediately first
        self.run_extraction()
        
        # Schedule future runs
        schedule.every(24).hours.do(self.run_extraction)
        
        next_run = datetime.now() + timedelta(hours=24)
        logger.info(f"Next extraction scheduled for: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Scheduled extraction terminated by user")

# Main execution function
def main():
    scraper = FundraiseInsiderScraper(headless=True)  # Ensure headless=True for production
    scraper.schedule_extraction()

if __name__ == "__main__":
    main()
