import os
import time
import json
import csv
import logging
import schedule
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException
)
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options

class TopStartupsScraper:
    def __init__(self, output_dir="./data", headless=True, log_level=logging.INFO):
        self.output_dir = output_dir
        self.headless = headless
        self.driver = None
        self.logger = self.setup_logging(log_level)
        os.makedirs(self.output_dir, exist_ok=True)

    def setup_logging(self, log_level):
        """Configure comprehensive logging system"""
        logger = logging.getLogger("TopStartupsScraper")
        logger.setLevel(log_level)

        # Create logs directory if not exists
        log_dir = "./logs"
        os.makedirs(log_dir, exist_ok=True)

        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(f"{log_dir}/scraper_{timestamp}.log")
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger

    def init_driver(self):
        """Initialize Chrome WebDriver with advanced options"""
        try:
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920x1080")
            chrome_options.add_argument("--disable-notifications")
            chrome_options.add_argument("--lang=en-US")

            # Initialize WebDriver with automatic management
            self.driver = webdriver.Chrome(
                service=ChromeService(ChromeDriverManager().install()),
                options=chrome_options
            )
            self.driver.implicitly_wait(10)
            self.logger.info("WebDriver initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize WebDriver: {str(e)}")
            raise

    def navigate_to_page(self, url="https://topstartups.io/?hq_location=San+Francisco+Bay+Area"):
        """Navigate to target URL with error handling"""
        try:
            self.logger.info(f"Navigating to: {url}")
            self.driver.get(url)
            
            # Wait for initial content to load
            WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.startup-item, div.startup-card"))
            )
            self.logger.info("Page loaded successfully")
        except TimeoutException:
            self.logger.error("Timed out waiting for page to load")
            raise
        except Exception as e:
            self.logger.error(f"Error navigating to page: {str(e)}")
            raise

    def handle_cookie_consent(self):
        """Handle cookie consent popups if present"""
        try:
            consent_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.ID, "accept-cookies"))
            )
            consent_button.click()
            self.logger.info("Cookie consent accepted")
            time.sleep(1)
        except (TimeoutException, NoSuchElementException):
            pass  # No consent dialog found

    def scroll_and_load_all(self, max_attempts=50, scroll_pause_time=2.0):
        """Automated scrolling and load more button handling"""
        self.logger.info("Starting automated scrolling and loading")
        
        last_height = 0
        scroll_count = 0
        show_more_clicks = 0
        consecutive_no_change = 0
        button_selectors = [
            (By.XPATH, "//button[contains(., 'Show more')]"),
            (By.XPATH, "//button[contains(., 'Load more')]"),
            (By.CSS_SELECTOR, "button.load-more"),
            (By.CSS_SELECTOR, "button[data-aut-id='btnLoadMore']")
        ]

        while scroll_count < max_attempts and consecutive_no_change < 3:
            current_height = self.driver.execute_script("return document.body.scrollHeight")
            
            # Check for content height change
            if current_height == last_height:
                consecutive_no_change += 1
            else:
                consecutive_no_change = 0
                last_height = current_height

            # Scroll to bottom
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            scroll_count += 1
            time.sleep(scroll_pause_time)

            # Attempt to click load more buttons
            for selector_type, selector in button_selectors:
                try:
                    buttons = self.driver.find_elements(selector_type, selector)
                    if buttons:
                        for button in buttons:
                            if button.is_displayed() and button.is_enabled():
                                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
                                button.click()
                                show_more_clicks += 1
                                consecutive_no_change = 0
                                time.sleep(scroll_pause_time * 2)
                                break
                except (NoSuchElementException, StaleElementReferenceException):
                    continue

        self.logger.info(f"Scroll complete. Total scrolls: {scroll_count}, Show more clicks: {show_more_clicks}")

    def extract_startup_data(self):
        """Robust data extraction with multiple selector strategies"""
        self.logger.info("Starting data extraction")
        
        # Try multiple selector strategies
        selectors = [
            {
                "container": "div.startup-item",
                "name": "h3",
                "description": "p.what-they-do",
                "location": "div.quick-facts p.location",
                "funding": "div.funding",
                "links": "div.action-links a"
            },
            {
                "container": "div.startup-card",
                "name": "h3.startup-name",
                "description": "p.startup-description",
                "location": "div.location-info",
                "funding": "div.funding-details",
                "links": "div.links-container a"
            }
        ]

        startups = []
        current_selectors = None

        # Find appropriate selector set
        for selector_set in selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector_set["container"])
                if elements:
                    current_selectors = selector_set
                    self.logger.info(f"Using selector set: {selector_set['container']}")
                    break
            except NoSuchElementException:
                continue

        if not current_selectors:
            raise Exception("No valid selector set found for startup elements")

        # Extract data for each startup
        startup_elements = self.driver.find_elements(By.CSS_SELECTOR, current_selectors["container"])
        self.logger.info(f"Found {len(startup_elements)} startup elements")

        for index, element in enumerate(startup_elements, 1):
            try:
                startup_data = {
                    "name": self._safe_extract(element, current_selectors["name"]),
                    "description": self._safe_extract(element, current_selectors["description"]),
                    "location": self._safe_extract(element, current_selectors["location"]),
                    "funding": self._safe_extract(element, current_selectors["funding"]),
                    "links": self._extract_links(element, current_selectors["links"]),
                    "timestamp": datetime.now().isoformat()
                }
                startups.append(startup_data)
                self.logger.debug(f"Extracted startup {index}/{len(startup_elements)}")
            except Exception as e:
                self.logger.warning(f"Error extracting startup {index}: {str(e)}")
                continue

        return startups

    def _safe_extract(self, parent_element, selector):
        """Safe element text extraction with error handling"""
        try:
            element = parent_element.find_element(By.CSS_SELECTOR, selector)
            return element.text.strip()
        except NoSuchElementException:
            return "N/A"
        except StaleElementReferenceException:
            return "N/A"

    def _extract_links(self, parent_element, selector):
        """Extract hrefs from link elements"""
        links = {}
        try:
            elements = parent_element.find_elements(By.CSS_SELECTOR, selector)
            for element in elements:
                text = element.text.strip().lower().replace(" ", "_")
                href = element.get_attribute("href")
                if text and href:
                    links[text] = href
            return links
        except NoSuchElementException:
            return {}

    def save_data(self, data, timestamp=None):
        """Save data in JSON and CSV formats with error checking"""
        if not data:
            self.logger.warning("No data to save")
            return

        timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"bay_area_startups_{timestamp}"

        # Save JSON
        json_path = os.path.join(self.output_dir, f"{base_filename}.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved JSON data: {json_path}")
        except IOError as e:
            self.logger.error(f"Failed to save JSON: {str(e)}")

        # Save CSV
        csv_path = os.path.join(self.output_dir, f"{base_filename}.csv")
        try:
            # Flatten data structure
            fieldnames = ["name", "description", "location", "funding"] + \
                        ["link_" + key for key in data[0]["links"].keys()] if data else []
            
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for item in data:
                    row = {
                        "name": item["name"],
                        "description": item["description"],
                        "location": item["location"],
                        "funding": item["funding"]
                    }
                    # Add links
                    for key, value in item["links"].items():
                        row[f"link_{key}"] = value
                    writer.writerow(row)
            self.logger.info(f"Saved CSV data: {csv_path}")
        except IOError as e:
            self.logger.error(f"Failed to save CSV: {str(e)}")

    def run_scraper(self, max_retries=3):
        """Main execution flow with retry logic"""
        for attempt in range(1, max_retries + 1):
            try:
                self.logger.info(f"Starting scraping attempt {attempt}/{max_retries}")
                self.init_driver()
                self.navigate_to_page()
                self.handle_cookie_consent()
                self.scroll_and_load_all()
                data = self.extract_startup_data()
                self.save_data(data)
                return True
            except Exception as e:
                self.logger.error(f"Attempt {attempt} failed: {str(e)}")
                self.capture_screenshot(f"error_attempt_{attempt}")
                if attempt < max_retries:
                    self.logger.info(f"Retrying in {attempt * 10} seconds...")
                    time.sleep(attempt * 10)
                else:
                    self.logger.error("All scraping attempts failed")
                    return False
            finally:
                self.close_driver()

    def capture_screenshot(self, filename):
        """Capture screenshot for debugging"""
        try:
            screenshots_dir = "./screenshots"
            os.makedirs(screenshots_dir, exist_ok=True)
            path = os.path.join(screenshots_dir, f"{filename}_{datetime.now().strftime('%H%M%S')}.png")
            self.driver.save_screenshot(path)
            self.logger.info(f"Screenshot saved: {path}")
        except Exception as e:
            self.logger.error(f"Failed to capture screenshot: {str(e)}")

    def close_driver(self):
        """Clean up WebDriver resources"""
        if self.driver:
            try:
                self.driver.quit()
                self.logger.info("WebDriver closed successfully")
            except Exception as e:
                self.logger.error(f"Error closing WebDriver: {str(e)}")
            finally:
                self.driver = None

def schedule_scraping():
    """Schedule weekly scraping jobs"""
    logger = logging.getLogger("Scheduler")
    logger.setLevel(logging.INFO)

    def scraping_job():
        logger.info("Starting scheduled scraping job")
        try:
            scraper = TopStartupsScraper(headless=True)
            success = scraper.run_scraper()
            if success:
                logger.info("Scheduled job completed successfully")
            else:
                logger.error("Scheduled job failed")
        except Exception as e:
            logger.error(f"Error in scheduled job: {str(e)}")

    # Schedule every Monday at 3 AM
    schedule.every().monday.at("03:00").do(scraping_job)

    logger.info("Scheduler started. Running weekly on Mondays at 3 AM.")
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    # For manual execution with logging
    scraper = TopStartupsScraper(headless=False, log_level=logging.DEBUG)
    scraper.run_scraper()

    # Uncomment to enable weekly scheduling
    # schedule_scraping()
