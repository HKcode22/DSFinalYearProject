import csv
import json
import logging
import time
from dataclasses import dataclass
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from pathlib import Path
import schedule

DATA_DIR = Path("growthlist_data")
DATA_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(DATA_DIR / "growthlist_scraping.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class GrowthListStartup:
    name: str
    website: str
    industry: str
    country: str
    funding_amount: str
    funding_type: str
    last_funding_date: str
    funding_usd: float = None

class GrowthListScraper:
    BASE_URL = "https://growthlist.co/san-francisco-startups/"
    
    def __init__(self, headless=True):
        self.startups = []
        self.driver = self._setup_driver(headless)
        
    def handle_popup(self) -> bool:
        """Handle the subscription popup with multiple fallback strategies"""
        logger.info("Checking for popup...")
        self.driver.save_screenshot(DATA_DIR / "before_popup_handling.png")
        
        # Strategy 1: Try to find and click the close button
        try:
            logger.info("Strategy 1: Looking for × close button")
            close_selectors = ["button.close", ".modal-close", ".popup-close", "[aria-label='Close']"]
            for selector in close_selectors:
                try:
                    close_buttons = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for button in close_buttons:
                        if button.is_displayed():
                            button.click()
                            logger.info("Clicked close button")
                            time.sleep(1)
                            return True
                except Exception as e:
                    logger.warning(f"Error with selector {selector}: {str(e)}")
        except Exception as e:
            logger.warning(f"Error in Strategy 1: {str(e)}")
        
        # Strategy 2: Try ESC key to close modal
        try:
            logger.info("Strategy 2: Sending ESC key")
            ActionChains(self.driver).send_keys(Keys.ESCAPE).perform()
            time.sleep(1)
        except Exception as e:
            logger.warning(f"Error in Strategy 2: {str(e)}")
        
        # Strategy 3: Use JavaScript to hide popup elements
        try:
            logger.info("Strategy 3: Using JavaScript to remove popup")
            self.driver.execute_script("""
                const popups = document.querySelectorAll('.modal, .popup, [class*="modal"], [class*="popup"]');
                popups.forEach(el => el.style.display = 'none');
                document.body.style.overflow = 'auto';
            """)
            time.sleep(1)
        except Exception as e:
            logger.warning(f"Error in Strategy 3: {str(e)}")
        
        self.driver.save_screenshot(DATA_DIR / "after_popup_handling.png")
        return True

    def extract_data(self):
        """Extract startup data from the website with improved table detection"""
        logger.info(f"Navigating to {self.BASE_URL}")
        self.driver.get(self.BASE_URL)
        
        # Wait for initial page load
        logger.info("Waiting for page to load...")
        WebDriverWait(self.driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        logger.info("Page loaded successfully")
        
        # Handle popup that appears on page load
        self.handle_popup()
        
        # Force full page load with scroll operations
        logger.info("Scrolling to trigger content loading...")
        self.driver.execute_script("window.scrollTo(0, 500);")
        time.sleep(2)
        self.driver.execute_script("window.scrollTo(0, 1000);")
        time.sleep(2)
        self.driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(2)
        
        # Try to find the table with multiple selectors and longer wait time
        logger.info("Looking for startup table with multiple selectors...")
        table_found = False
        table_selectors = [
            "table", 
            "table.wp-block-table", 
            ".wp-block-table table", 
            "table[class*='table']",
            ".table",
            "[role='table']"
        ]
        
        table = None
        for selector in table_selectors:
            try:
                logger.info(f"Trying selector: {selector}")
                table = WebDriverWait(self.driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                if table:
                    logger.info(f"Found table with selector: {selector}")
                    table_found = True
                    break
            except TimeoutException:
                logger.warning(f"Timeout waiting for table with selector: {selector}")
                continue
        
        if not table_found:
            # Log the page source for debugging
            logger.error("Table not found with any selector")
            self.driver.save_screenshot(DATA_DIR / "table_not_found.png")
            with open(DATA_DIR / "page_source.html", "w", encoding="utf-8") as f:
                f.write(self.driver.page_source)
            logger.info("Saved page source to page_source.html for debugging")
            return
        
        # Extract data from table rows
        logger.info("Extracting data from table...")
        try:
            rows = table.find_elements(By.CSS_SELECTOR, "tbody tr")
            logger.info(f"Found {len(rows)} table rows")
            
            for i, row in enumerate(rows):
                try:
                    cols = row.find_elements(By.TAG_NAME, "td")
                    if len(cols) != 7:
                        continue
                    
                    website = ""
                    try:
                        website = cols[1].find_element(By.TAG_NAME, "a").get_attribute("href")
                    except:
                        website = cols[1].text.strip()
                    
                    startup = GrowthListStartup(
                        name=cols[0].text.strip(),
                        website=website,
                        industry=cols[2].text.strip(),
                        country=cols[3].text.strip(),
                        funding_amount=cols[4].text.strip(),
                        funding_type=cols[5].text.strip(),
                        last_funding_date=cols[6].text.strip()
                    )
                    self.startups.append(startup)
                    logger.info(f"Extracted startup #{len(self.startups)}: {startup.name}")
                except Exception as e:
                    logger.error(f"Error extracting data from row {i+1}: {str(e)}")
            
            logger.info(f"Successfully extracted {len(self.startups)} startups")
        except Exception as e:
            logger.error(f"Error extracting data from table: {str(e)}")
            self.driver.save_screenshot(DATA_DIR / "extraction_error.png")
            logger.info("Saved error screenshot to extraction_error.png")
        
        # Save results if startups were extracted
        if self.startups:
            self._save_results()
            self.normalize_funding()
            self.analyze_industries()

    def normalize_funding(self):
        """Convert funding amounts to numerical values"""
        for startup in self.startups:
            amount = startup.funding_amount
            if not amount:
                continue
                
            # Handle different formats
            if amount.startswith('$'):
                amount = amount[1:].replace(',', '')
                if 'M' in amount:
                    startup.funding_usd = float(amount.replace('M', '')) * 1_000_000
                elif 'B' in amount:
                    startup.funding_usd = float(amount.replace('B', '')) * 1_000_000_000
                else:
                    try:
                        startup.funding_usd = float(amount)
                    except:
                        startup.funding_usd = None
            else:
                startup.funding_usd = None

    def analyze_industries(self):
        """Generate industry distribution report"""
        industry_counts = {}
        for s in self.startups:
            industries = [i.strip() for i in s.industry.split(',')]
            for industry in industries:
                industry_counts[industry] = industry_counts.get(industry, 0) + 1
        
        report = {
            "total_startups": len(self.startups),
            "industry_distribution": sorted(
                industry_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            ),
            "average_funding": self._calculate_average_funding()
        }
        
        with open(DATA_DIR / "industry_analysis.json", 'w') as f:
            json.dump(report, f, indent=4)

    def _calculate_average_funding(self):
        valid = [s.funding_usd for s in self.startups if s.funding_usd]
        return sum(valid)/len(valid) if valid else 0

    def _save_results(self):
        """Save the extracted startup data to CSV and JSON files"""
        if not self.startups:
            logger.warning("No startups to save")
            return
        
        # Save to CSV
        csv_path = DATA_DIR / "growthlist_startups.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Name","Website","Industry","Country","Funding Amount","Funding Type","Last Funding Date"])
            for s in self.startups:
                writer.writerow([s.name, s.website, s.industry, s.country, 
                               s.funding_amount, s.funding_type, s.last_funding_date])
        
        # Save to JSON
        json_path = DATA_DIR / "growthlist_startups.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump([s.__dict__ for s in self.startups], f, indent=4)

    def cleanup(self):
        """Close the WebDriver and perform cleanup"""
        try:
            self.driver.quit()
            logger.info("WebDriver closed successfully")
        except Exception as e:
            logger.warning(f"Error closing WebDriver: {str(e)}")

    def _setup_driver(self, headless_mode):
        logger.info(f"Setting up Chrome WebDriver (Headless: {headless_mode})")
        chrome_options = Options()
        if headless_mode:
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.set_page_load_timeout(30)
            logger.info("WebDriver setup completed successfully")
            return driver
        except Exception as e:
            logger.error(f"WebDriver setup failed: {e}")
            raise

def main():
    """Main function to run the scraper"""
    logger.info("Starting GrowthList scraper script")
    scraper = None
    try:
        scraper = GrowthListScraper(headless=True)
        scraper.extract_data()
        scraper.normalize_funding()
        scraper.analyze_industries()
        if scraper.startups:
            logger.info(f"Extraction completed successfully. Extracted {len(scraper.startups)} startups.")
        else:
            logger.warning("No startups were extracted. Check logs for errors.")
    except Exception as e:
        logger.error(f"An error occurred during scraping: {str(e)}")
    finally:
        scraper.cleanup()


def schedule_scraper():
    """Schedule the scraper to run every 24 hours."""
    logger.info("Scheduling the scraper to run every 24 hours.")
    schedule.every(24).hours.do(main)
    
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
    schedule_scraper()  # Uncomment to run the scraper on a schedule