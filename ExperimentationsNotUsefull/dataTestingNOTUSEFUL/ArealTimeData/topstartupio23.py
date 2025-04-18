import os
import time
import random
import json
import csv
import logging
import logging.handlers
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    NoSuchWindowException,
    StaleElementReferenceException,
    TimeoutException,
    ElementClickInterceptedException,
    WebDriverException,
    NoSuchElementException
)
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

class UltimateStartupScraper:
    def __init__(self, output_dir="./data", headless=False, log_level=logging.INFO):
        self.output_dir = output_dir
        self.headless = headless
        self.driver = None
        self.logger = self.setup_logging(log_level)
        self.retry_config = {
            'max_retries': 5,
            'base_delay': 10,
            'backoff_factor': 2
        }
        os.makedirs(self.output_dir, exist_ok=True)
        self.setup_directories()
        self.main_window_handle = None

    def setup_logging(self, log_level):
        logger = logging.getLogger("UltimateStartupScraper")
        logger.setLevel(log_level)

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        )

        # Detailed file logging
        file_handler = logging.handlers.RotatingFileHandler(
            './logs/scraper.log', 
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)

        # Simplified console output
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

    def setup_directories(self):
        """Create required directories"""
        for dir_path in ['./logs', './screenshots', self.output_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def init_driver(self):
        """Initialize driver with comprehensive anti-detection measures"""
        try:
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument("--headless=new")
            
            # Extended anti-detection configuration
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option("useAutomationExtension", False)
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            
            # Add random user agent
            user_agents = [
                f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{random.randint(90,115)}.0.0.0 Safari/537.36",
                f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_{random.randint(13,15)}_{random.randint(0,6)}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{random.randint(90,115)}.0.0.0 Safari/537.36",
                f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{random.randint(90,115)}.0.0.0 Safari/537.36 Edg/{random.randint(90,115)}.0.0.0"
            ]
            chrome_options.add_argument(f"user-agent={random.choice(user_agents)}")
            
            # Increase timeouts for stability
            chrome_options.add_argument("--disable-web-security")
            chrome_options.add_argument("--allow-running-insecure-content")
            
            # Initialize driver with enhanced configuration
            service = ChromeService(ChromeDriverManager().install())
            service.start()
            
            self.driver = webdriver.Chrome(
                service=service,
                options=chrome_options
            )
            
            # Configure browser settings
            self.driver.set_page_load_timeout(60)
            self.driver.implicitly_wait(20)
            
            # Execute stealth JavaScript
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            # Add additional stealth scripts
            stealth_js = """
            const newProto = navigator.__proto__;
            delete newProto.webdriver;
            navigator.__proto__ = newProto;
            
            window.chrome = {
              runtime: {},
              loadTimes: function() {},
              csi: function() {},
              app: {}
            };
            """
            self.driver.execute_script(stealth_js)
            
            self.logger.info("Enhanced stealth driver initialized successfully")
        except Exception as e:
            self.logger.error(f"Driver initialization failed: {str(e)}")
            raise

    def navigate_to_page(self, url):
        """Robust navigation with multiple verification strategies"""
        try:
            self.logger.info(f"Navigating to: {url}")
            
            # Add longer timeout for initial connection
            self.driver.set_page_load_timeout(60)
            self.driver.get(url)
            self.main_window_handle = self.driver.current_window_handle
            
            # Wait for document to be ready first
            WebDriverWait(self.driver, 30).until(
                lambda d: d.execute_script("return document.readyState") == 'complete'
            )
            self.logger.info("Basic page load complete")
            
            # Try multiple selectors to verify content
            content_selectors = [
                (By.XPATH, "//div[contains(@class, 'startup-item')]"),
                (By.XPATH, "//div[contains(@class, 'startup-card')]"),
                (By.XPATH, "//*[contains(text(), 'What they do')]"),
                (By.CSS_SELECTOR, ".startup-item, .startup-card"),
                (By.CSS_SELECTOR, "div[data-testid='startup-item']")
            ]
            
            for selector_type, selector in content_selectors:
                try:
                    WebDriverWait(self.driver, 20).until(
                        EC.presence_of_element_located((selector_type, selector))
                    )
                    self.logger.info(f"Content verified using: {selector}")
                    # Add small random delay to appear more human-like
                    time.sleep(random.uniform(1, 3))
                    return
                except TimeoutException:
                    continue
            
            # If we reached here, no content was found
            raise TimeoutException("Could not verify content on page")
            
        except Exception as e:
            self.logger.error(f"Navigation failed: {str(e)}")
            self.capture_screenshot("navigation_failure")
            raise

    def smart_scroller(self):
        """Advanced scrolling with multiple methods and content verification"""
        self.logger.info("Starting enhanced scrolling sequence")
        
        last_count = 0
        same_count = 0
        max_attempts = 30
        scroll_attempt = 0
        
        # Define scroll methods - will rotate through these
        scroll_methods = [
            self._scroll_to_bottom,
            self._scroll_incrementally,
            self._scroll_with_keys,
            self._scroll_with_actions
        ]
        
        while scroll_attempt < max_attempts and same_count < 5:
            # Count current visible elements
            current_count = self._count_startup_elements()
            self.logger.info(f"Scroll attempt {scroll_attempt+1}: Found {current_count} startup elements")
            
            # Check if we're still loading new content
            if current_count > last_count:
                self.logger.info(f"Found {current_count - last_count} new elements")
                same_count = 0
                last_count = current_count
            else:
                same_count += 1
                self.logger.info(f"No new elements found (attempt {same_count}/5)")
                
                # Try clicking show more button before giving up
                if same_count >= 3:
                    if self.handle_load_more_button():
                        same_count = 0  # Reset if button was clicked
                        time.sleep(random.uniform(2, 4))  # Wait for content to load
            
            # Use different scroll methods in rotation
            scroll_method = scroll_methods[scroll_attempt % len(scroll_methods)]
            scroll_method()
            
            # Add randomized human-like delay
            time.sleep(random.uniform(1.5, 3.5))
            scroll_attempt += 1
        
        self.logger.info(f"Scrolling complete. Total elements: {self._count_startup_elements()}")

    def _count_startup_elements(self):
        """Count visible startup elements with fallbacks"""
        try:
            elements = self.driver.find_elements(
                By.XPATH, "//div[contains(@class, 'startup-item') or contains(@class, 'startup-card')]"
            )
            visible_elements = [el for el in elements if self._is_element_visible(el)]
            return len(visible_elements)
        except Exception as e:
            self.logger.warning(f"Error counting elements: {str(e)}")
            return 0

    def _is_element_visible(self, element):
        """Check if element is visible in viewport"""
        try:
            return element.is_displayed() and self.driver.execute_script(
                "var elem = arguments[0], box = elem.getBoundingClientRect(); " +
                "return box.top < window.innerHeight && box.bottom > 0;", 
                element
            )
        except:
            return False

    def _scroll_to_bottom(self):
        """Scroll to bottom of page using JavaScript"""
        self.logger.debug("Performing full-page scroll")
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    def _scroll_incrementally(self):
        """Perform multiple smaller scrolls"""
        self.logger.debug("Performing incremental scroll")
        for _ in range(3):
            scroll_amount = random.randint(500, 1000)
            self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
            time.sleep(random.uniform(0.3, 0.7))

    def _scroll_with_keys(self):
        """Scroll using keyboard PAGE_DOWN"""
        self.logger.debug("Performing keyboard scroll")
        element = self.driver.find_element(By.TAG_NAME, "body")
        for _ in range(3):
            element.send_keys(Keys.PAGE_DOWN)
            time.sleep(random.uniform(0.3, 0.7))

    def _scroll_with_actions(self):
        """Scroll using ActionChains"""
        self.logger.debug("Performing ActionChains scroll")
        ActionChains(self.driver)\
            .send_keys(Keys.END)\
            .pause(0.5)\
            .send_keys(Keys.HOME)\
            .pause(0.5)\
            .send_keys(Keys.END)\
            .perform()

    def handle_load_more_button(self):
        """Multi-strategy button detection and clicking"""
        self.logger.info("Searching for 'Show More' button")
        
        # Define multiple button selectors to try
        button_selectors = [
            {"method": By.XPATH, "value": "//button[contains(., 'Show More')]", "description": "Text: Show More"},
            {"method": By.XPATH, "value": "//button[contains(., 'Load More')]", "description": "Text: Load More"},
            {"method": By.CSS_SELECTOR, "value": "button.load-more", "description": "Class: load-more"},
            {"method": By.CSS_SELECTOR, "value": "button.pagination", "description": "Class: pagination"},
            {"method": By.XPATH, "value": "//button[contains(@class, 'more') or contains(@class, 'load')]", "description": "Class contains: more/load"},
            {"method": By.XPATH, "value": "//a[contains(., 'Show More') or contains(., 'Load More')]", "description": "Link text: Show/Load More"}
        ]
        
        for selector in button_selectors:
            try:
                self.logger.debug(f"Trying button strategy: {selector['description']}")
                
                # Find all matching elements
                elements = self.driver.find_elements(selector["method"], selector["value"])
                
                # Try each element that matches
                for element in elements:
                    if not element.is_displayed() or not element.is_enabled():
                        continue
                        
                    self.logger.info(f"Found button via {selector['description']}")
                    
                    # Scroll button into view with offset to ensure it's visible
                    self.driver.execute_script(
                        "arguments[0].scrollIntoView({block: 'center', behavior: 'smooth'});", 
                        element
                    )
                    time.sleep(1)
                    
                    # Try multiple click methods
                    click_successful = False
                    
                    # Method 1: Direct Click
                    try:
                        element.click()
                        click_successful = True
                        self.logger.info("Button clicked successfully (direct)")
                    except Exception as e:
                        self.logger.debug(f"Direct click failed: {str(e)}")
                    
                    # Method 2: JavaScript Click
                    if not click_successful:
                        try:
                            self.driver.execute_script("arguments[0].click();", element)
                            click_successful = True
                            self.logger.info("Button clicked successfully (JavaScript)")
                        except Exception as e:
                            self.logger.debug(f"JavaScript click failed: {str(e)}")
                    
                    # Method 3: ActionChains Click
                    if not click_successful:
                        try:
                            ActionChains(self.driver).move_to_element(element).click().perform()
                            click_successful = True
                            self.logger.info("Button clicked successfully (ActionChains)")
                        except Exception as e:
                            self.logger.debug(f"ActionChains click failed: {str(e)}")
                    
                    if click_successful:
                        # Verify click effect (wait for loading indicator or new content)
                        time.sleep(2)
                        return True
            
            except Exception as e:
                self.logger.debug(f"Error with button strategy {selector['description']}: {str(e)}")
                continue
        
        self.logger.warning("No 'Show More' button found or clickable")
        return False

    def run_with_retries(self):
        """Enhanced retry logic with component-level recovery"""
        attempt = 1
        max_retries = self.retry_config['max_retries']
        
        while attempt <= max_retries:
            try:
                self.logger.info(f"Starting scraping attempt {attempt}/{max_retries}")
                
                # Initialize new driver for each attempt
                if self.driver:
                    self.cleanup()
                
                self.init_driver()
                
                # Navigation phase with dedicated error handling
                try:
                    self.navigate_to_page("https://topstartups.io/?hq_location=San+Francisco+Bay+Area")
                except Exception as e:
                    self.logger.error(f"Navigation failed: {str(e)}")
                    raise
                
                # Handle any overlays or popups
                self.handle_overlays()
                
                # Scrolling phase with dedicated error handling
                try:
                    self.smart_scroller()
                except Exception as e:
                    self.logger.error(f"Scrolling failed: {str(e)}")
                    self.capture_screenshot("scrolling_failure")
                    raise
                
                # Data extraction phase
                data = self.extract_startup_data()
                if not data:
                    self.logger.warning("No data extracted")
                    raise Exception("Data extraction failed - no items found")
                
                # Data saving phase
                self.save_data(data)
                
                self.logger.info(f"Successfully scraped {len(data)} startup items")
                return True
                
            except NoSuchWindowException:
                self.logger.error("Browser window closed unexpectedly")
                self.driver = None
                
            except Exception as e:
                self.logger.error(f"Attempt {attempt} failed: {str(e)}")
                self.capture_screenshot(f"error_attempt_{attempt}")
                
            finally:
                # Increment attempt counter and handle retries
                if attempt < max_retries:
                    delay = self.retry_config['base_delay'] * (self.retry_config['backoff_factor'] ** (attempt - 1))
                    self.logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                
                attempt += 1
                self.cleanup()
        
        self.logger.error("All scraping attempts failed")
        return False

    def handle_overlays(self):
        """Comprehensive overlay handling with multiple strategies"""
        overlay_selectors = [
            (By.ID, "cookie-consent"),
            (By.XPATH, "//button[contains(., 'Accept')]"),
            (By.CSS_SELECTOR, "button[aria-label='Close']"),
            (By.CSS_SELECTOR, ".cookie-banner button"),
            (By.ID, "CybotCookiebotDialogBodyButtonAccept")
        ]
        
        for selector_type, selector in overlay_selectors:
            try:
                button = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((selector_type, selector))
                )
                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
                self.driver.execute_script("arguments[0].click();", button)
                self.logger.info(f"Closed overlay using {selector}")
                time.sleep(1)
                return True
            except Exception:
                continue
        return False

    def extract_startup_data(self):
        """Adaptive data extraction with multiple fallback strategies"""
        self.logger.info("Starting data extraction process")
        containers = self.get_startup_elements()
        
        data = []
        for idx, container in enumerate(containers, 1):
            try:
                item = {
                    'name': self.safe_extract(container, ["h3", ".company-name"]),
                    'description': self.safe_extract(container, ["p.what-they-do", ".description"]),
                    'location': self.safe_extract(container, ["div.location", "span.hq-location"]),
                    'funding': self.safe_extract(container, ["div.funding", ".investment-details"]),
                    'links': self.extract_links(container),
                    'timestamp': datetime.now().isoformat()
                }
                data.append(item)
                self.logger.debug(f"Processed item {idx}/{len(containers)}")
            except Exception as e:
                self.logger.warning(f"Error processing item {idx}: {str(e)}")
                continue
        
        return data

    def safe_extract(self, parent, selectors):
        """Safely extract text from multiple possible selectors"""
        for selector in selectors:
            try:
                element = parent.find_element(By.CSS_SELECTOR, selector)
                return element.text.strip()
            except NoSuchElementException:
                continue
        return "N/A"

    def extract_links(self, parent):
        """Extract all links from a container"""
        links = {}
        try:
            elements = parent.find_elements(By.CSS_SELECTOR, "a")
            for el in elements:
                try:
                    text = el.text.strip().lower().replace(' ', '_') or 'unnamed_link'
                    href = el.get_attribute('href')
                    if href:
                        links[text] = href
                except StaleElementReferenceException:
                    continue
        except NoSuchElementException:
            pass
        return links

    def save_data(self, data):
        """Save data in multiple formats with validation"""
        if not data:
            self.logger.warning("No data to save")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON Saving
        json_path = os.path.join(self.output_dir, f"startups_{timestamp}.json")
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved JSON data to {json_path}")
        except Exception as e:
            self.logger.error(f"JSON save failed: {str(e)}")
        
        # CSV Saving
        csv_path = os.path.join(self.output_dir, f"startups_{timestamp}.csv")
        try:
            fieldnames = list(data[0].keys()) + [f"link_{k}" for k in data[0]['links'].keys()]
            fieldnames.remove('links')
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for item in data:
                    row = {k: v for k, v in item.items() if k != 'links'}
                    row.update({f"link_{k}": v for k, v in item['links'].items()})
                    writer.writerow(row)
            self.logger.info(f"Saved CSV data to {csv_path}")
        except Exception as e:
            self.logger.error(f"CSV save failed: {str(e)}")

    def capture_screenshot(self, prefix):
        """Capture diagnostic screenshot"""
        try:
            timestamp = datetime.now().strftime("%H%M%S")
            filename = f"{prefix}_{timestamp}.png"
            path = os.path.join("./screenshots", filename)
            self.driver.save_screenshot(path)
            self.logger.info(f"Screenshot saved: {filename}")
        except Exception as e:
            self.logger.error(f"Failed to capture screenshot: {str(e)}")

    def cleanup(self):
        """Safe cleanup with window handle check"""
        if self.driver:
            try:
                if self.main_window_handle in self.driver.window_handles:
                    self.driver.close()
                self.driver.quit()
            except Exception as e:
                self.logger.error(f"Cleanup failed: {str(e)}")
            finally:
                self.driver = None

if __name__ == "__main__":
    scraper = UltimateStartupScraper(headless=False)
    scraper.run_with_retries()
