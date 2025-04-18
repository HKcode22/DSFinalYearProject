import time
import json
import csv
import logging
import os
import re
import random
from datetime import datetime
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
import undetected_chromedriver as uc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('scraper.log'), logging.StreamHandler()]
)

class TopStartupsScraper:
    def __init__(self, url="https://topstartups.io/?hq_location=San+Francisco+Bay+Area&sort=funding", headless=True):
        self.url = url
        self.driver = None
        self.startups_data = []
        self.output_dir = "topstartup_io_data"
        self.headless = headless
        self._create_output_dir()

    def _create_output_dir(self):
        """Create output directory with error handling"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            logging.info(f"Created output directory: {self.output_dir}")
        except Exception as e:
            logging.error(f"Directory creation failed: {str(e)}")
            self.output_dir = "."

    def setup_browser(self):
        """Configure undetected ChromeDriver with human-like settings"""
        try:
            options = uc.ChromeOptions()
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--disable-popup-blocking")
            options.add_argument("--disable-notifications")
            
            if self.headless:
                options.add_argument("--headless=new")
            
            # Anti-detection parameters
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--disable-infobars")
            options.add_argument("--no-first-run")
            options.add_argument("--no-service-autorun")
            
            self.driver = uc.Chrome(
                options=options,
                version_main=120,
                driver_executable_path=ChromeDriverManager().install()
            )
            
            # Human-like browser fingerprint
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            return True
            
        except Exception as e:
            logging.error(f"Browser setup failed: {str(e)}")
            return False

    def navigate_to_url(self):
        """Human-like navigation with random delays"""
        try:
            # Random delay before navigation
            time.sleep(random.uniform(1.0, 3.0))
            
            self.driver.get(self.url)
            logging.info("Navigating to target URL...")
            
            # Wait for core content
            WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.startup-card"))
            )
            return True
            
        except Exception as e:
            logging.error(f"Navigation failed: {str(e)}")
            return False

    def _human_scroll(self):
        """Gradual scrolling with randomized patterns"""
        container = self.driver.find_element(By.CSS_SELECTOR, "div.startup-list")
        last_height = 0
        scroll_attempts = 0
        
        while scroll_attempts < 5:
            # Random scroll distance and delay
            scroll_distance = random.randint(300, 800)
            scroll_delay = random.uniform(0.5, 1.5)
            
            self.driver.execute_script(
                "arguments[0].scrollTop += arguments[1]", 
                container, 
                scroll_distance
            )
            time.sleep(scroll_delay)
            
            # Check for new content
            new_height = self.driver.execute_script(
                "return arguments[0].scrollHeight", container
            )
            
            if new_height == last_height:
                scroll_attempts += 1
            else:
                scroll_attempts = 0
                last_height = new_height
                
            # Attempt to click show more button
            self._click_show_more()

    def _click_show_more(self):
        """Advanced button interaction with multiple fallbacks"""
        try:
            # Wait for button to be clickable
            button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Show More')]"))
            )
            
            # Human-like hover before click
            ActionChains(self.driver)\
                .move_to_element(button)\
                .pause(random.uniform(0.5, 1.2))\
                .click()\
                .perform()
                
            # Random post-click delay
            time.sleep(random.uniform(1.0, 2.5))
            return True
            
        except Exception:
            # JavaScript fallback click
            try:
                self.driver.execute_script(
                    """document.querySelector('button:contains("Show More")').click()"""
                )
                return True
            except:
                return False

    def extract_data(self):
        """Robust data extraction with multiple fallback selectors"""
        try:
            cards = self.driver.find_elements(By.CSS_SELECTOR, "div.startup-card")
            logging.info(f"Found {len(cards)} startup cards")
            
            for card in cards:
                try:
                    data = {
                        "name": self._get_element_text(card, [
                            "h3.company-title", 
                            "h3.startup-name",
                            "div.header h3"
                        ]),
                        "description": self._get_element_text(card, [
                            "div.description", 
                            "div.bio",
                            "div.what-they-do"
                        ]),
                        "funding": self._clean_funding(
                            self._get_element_text(card, [
                                "div.funding",
                                "div.financials",
                                "div.investment-status"
                            ])
                        ),
                        "links": self._get_links(card),
                        "timestamp": datetime.now().isoformat()
                    }
                    self.startups_data.append(data)
                    
                except Exception as e:
                    logging.error(f"Error processing card: {str(e)}")
            
            return True
            
        except Exception as e:
            logging.error(f"Extraction failed: {str(e)}")
            return False

    def _get_element_text(self, element, selectors):
        """Multi-selector text extraction"""
        for selector in selectors:
            try:
                return element.find_element(By.CSS_SELECTOR, selector).text.strip()
            except:
                continue
        return ""

    def _clean_funding(self, text):
        """Funding amount extraction"""
        match = re.search(r'\$\d+[\d,]+', text or "")
        return match.group() if match else ""

    def _get_links(self, element):
        """Link collection with multiple fallbacks"""
        links = {}
        try:
            for link in element.find_elements(By.CSS_SELECTOR, "a[href]"):
                url = link.get_attribute("href")
                text = link.text.strip() or link.get_attribute("aria-label") or "Link"
                links[text] = url
        except:
            pass
        return links

    def save_data(self):
        """Data saving with error handling"""
        if not self.startups_data:
            logging.warning("No data to save")
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            # Save JSON
            json_path = os.path.join(self.output_dir, f"startups_{timestamp}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.startups_data, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved JSON: {json_path}")
            
            # Save CSV
            csv_path = os.path.join(self.output_dir, f"startups_{timestamp}.csv")
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.startups_data[0].keys())
                writer.writeheader()
                writer.writerows(self.startups_data)
            logging.info(f"Saved CSV: {csv_path}")
            
            return True
            
        except Exception as e:
            logging.error(f"Data save failed: {str(e)}")
            return False

    def run(self):
        """Main execution flow with error handling"""
        try:
            if not self.setup_browser():
                return False

            if not self.navigate_to_url():
                return False

            # Initial content load
            self._human_scroll()
            
            # Extract data
            if not self.extract_data():
                return False
                
            # Save results
            self.save_data()
            return True
            
        except Exception as e:
            logging.error(f"Critical error: {str(e)}")
            return False
            
        finally:
            if self.driver:
                self.driver.quit()
                logging.info("Browser session closed")

if __name__ == "__main__":
    scraper = TopStartupsScraper(headless=False)  # Set headless=False for debugging
    if scraper.run():
        print("Scraping completed successfully")
    else:
        print("Scraping failed - check logs for details")
