import time
import json
import csv
import logging
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import (TimeoutException, 
                                        NoSuchElementException,
                                        StaleElementReferenceException,
                                        ElementClickInterceptedException)
import os

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('scraper.log'), logging.StreamHandler()]
)

class TopStartupsScraper:
    def __init__(self, url="https://topstartups.io/?hq_location=San+Francisco+Bay+Area&sort=funding"):
        self.url = url
        self.driver = None
        self.startups_data = []
        self.show_more_clicks = 0
        self.scroll_attempts = 0
        self.total_startups = 0

        self.output_dir = "topstartup_io_real_time_data"
        self._create_output_dir()


        self.card_selector = None
        self.output_dir = "topstartup_io_real_time_data"
        self._create_output_directory()

    def _print_status(self, message):
        """Visual feedback with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")


    def _create_output_directory(self):
        """Ensure output directory exists"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self._print_status(f"📁 Created output directory: {self.output_dir}")
        except Exception as e:
            self._print_status(f"⚠️ Directory creation failed: {str(e)}")
            self.output_dir = "."


    def _create_output_dir(self):
        """Create output directory if missing"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self._print_status(f"📁 Using output directory: {self.output_dir}")
        except Exception as e:
            self._print_status(f"⚠️ Directory creation failed: {str(e)}")
            self.output_dir = "."



    def setup_browser(self):
        """Initialize browser with appropriate settings"""
        try:
            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument("--start-maximized")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option("useAutomationExtension", False)
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.set_window_size(1440, 900)
            self.driver.implicitly_wait(10)
            self._print_status("✅ Browser initialized successfully")
            return True
        except Exception as e:
            self._print_status(f"❌ Browser setup failed: {str(e)}")
            return False

    def navigate_to_url(self):
        """Navigate to target URL with extended timeout"""
        try:
            self.driver.get(self.url)
            self._print_status("🌐 Navigating to target URL...")
            # Long timeout to ensure page loads completely
            time.sleep(5)  # Give extra time for initial load
            return True
        except Exception as e:
            self._print_status(f"❌ Navigation failed: {str(e)}")
            return False

    def identify_card_elements(self):
        """Identify the correct card elements using multiple selectors"""
        # Try multiple potential selectors for startup cards
        card_selectors = [
            "div.startup-card", 
            "div.company-card",
            ".startup-container",
            ".card",
            "div[class*='startup']",
            "div[class*='company']",
            "div.product-item",
            ".startup-grid > div",
            "#product-grid > div"
        ]
        
        for selector in card_selectors:
            try:
                self._print_status(f"🔍 Trying selector: {selector}")
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if elements and len(elements) > 0:
                    self._print_status(f"✅ Found {len(elements)} elements with selector: {selector}")
                    return selector, elements
            except Exception:
                continue
        
        # If no selectors work, analyze page structure and dump it for debugging
        self._print_status("⚠️ Could not identify card elements with predefined selectors")
        self._analyze_page_structure()
        return None, []


    def click_show_more_direct(self):
        """Directly target the 'Show more' text using precise locators"""
        try:
            # Method 1: Direct text match with case sensitivity
            button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//div[text()='Show more']"))
            )
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
            self.driver.execute_script("arguments[0].click();", button)
            self._print_status(f"🎯 Clicked 'Show more' text element directly")
            return True
        except Exception as e:
            self._print_status(f"⚠️ Direct text match failed: {str(e)}")
        
        try:
            # Method 2: Look for any element containing exactly "Show more"
            button = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//*[contains(text(), 'Show more')]"))
            )
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
            self.driver.execute_script("arguments[0].click();", button)
            self._print_status(f"🎯 Clicked element containing 'Show more' text")
            return True
        except Exception as e:
            self._print_status(f"⚠️ Text contains match failed: {str(e)}")
        
        return False

    def click_show_more_with_javascript(self):
        """Use JavaScript to find and click the button by text content"""
        try:
            # Find any element containing "Show more" text (case insensitive)
            result = self.driver.execute_script("""
                var elements = document.querySelectorAll('*');
                for (var i = 0; i < elements.length; i++) {
                    var element = elements[i];
                    if (element.innerText && 
                        element.innerText.trim() === 'Show more' && 
                        element.offsetParent !== null) {
                        // Found visible element with exact text
                        element.click();
                        return true;
                    }
                }
                return false;
            """)
            
            if result:
                self._print_status("🎯 Clicked 'Show more' using JavaScript text search")
                time.sleep(1)
                return True
            else:
                self._print_status("⚠️ JavaScript text search found no matching elements")
        except Exception as e:
            self._print_status(f"⚠️ JavaScript click failed: {str(e)}")
        
        return False

    def take_screenshot_of_button_area(self):
        """Take screenshot to help debug button location"""
        try:
            # Scroll to where the button would typically be
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight - 300);")
            screenshot_path = f"button_area_{datetime.now().strftime('%H%M%S')}.png"
            self.driver.save_screenshot(screenshot_path)
            self._print_status(f"📸 Saved screenshot of button area to {screenshot_path}")
            
            # Dump HTML of the bottom section
            html = self.driver.execute_script("""
                return document.querySelector('body').innerHTML.substring(
                    document.querySelector('body').innerHTML.length - 10000
                );
            """)
            with open("bottom_html.txt", "w", encoding="utf-8") as f:
                f.write(html)
            self._print_status("💾 Saved HTML of bottom section for analysis")
        except Exception as e:
            self._print_status(f"⚠️ Screenshot failed: {str(e)}")








    def detect_end_of_content(self):
        """Determine if we've reached the end of all available content"""
        # Track card count over multiple attempts
        initial_count = len(self.driver.find_elements(By.CSS_SELECTOR, self.card_selector))
        
        # Try aggressive scrolling a few more times
        for _ in range(3):
            self._smart_scroll()
            time.sleep(1)
        
        # Check if card count increased
        final_count = len(self.driver.find_elements(By.CSS_SELECTOR, self.card_selector))
        
        if final_count == initial_count:
            self._print_status(f"🏁 Reached end of content - No more cards loaded after {self.scroll_attempts} scroll attempts")
            return True
        return False





    def _analyze_page_structure(self):
        """Analyze page structure to identify potential elements"""
        try:
            # Get all div elements to analyze structure
            all_divs = self.driver.find_elements(By.TAG_NAME, "div")
            self._print_status(f"Found {len(all_divs)} div elements on page")
            
            # Get the visible text content of the page
            page_text = self.driver.find_element(By.TAG_NAME, "body").text
            if "545 startups" in page_text:
                self._print_status("✅ Confirmed page contains '545 startups' text")
            
            # Take a screenshot for debugging
            self.driver.save_screenshot("page_structure.png")
            self._print_status("📸 Saved page screenshot to 'page_structure.png'")
            
            # Dump page source to file
            with open("page_source.html", "w", encoding="utf-8") as f:
                f.write(self.driver.page_source)
            self._print_status("💾 Saved page source to 'page_source.html'")
        except Exception as e:
            self._print_status(f"❌ Error analyzing page structure: {str(e)}")

    def scroll_with_multiple_methods(self):
        """Implement multiple scrolling approaches"""
        self._print_status("📜 Starting multi-method scrolling")
        success = False
        
        # Method 1: JavaScript scroll to bottom
        try:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            self.scroll_attempts += 1
            self._print_status(f"🖱 Method 1: Scrolled to bottom (attempt {self.scroll_attempts})")
            success = True
            time.sleep(2)
        except Exception as e:
            self._print_status(f"⚠️ Method 1 scroll failed: {str(e)}")
        
        # Method 2: Incremental scrolling
        try:
            total_height = self.driver.execute_script("return document.body.scrollHeight")
            for i in range(0, total_height, 500):
                self.driver.execute_script(f"window.scrollTo(0, {i});")
                time.sleep(0.2)
            self._print_status("🖱 Method 2: Incremental scrolling completed")
            success = True
            time.sleep(1)
        except Exception as e:
            self._print_status(f"⚠️ Method 2 scroll failed: {str(e)}")
        
        # Method 3: Scroll to specific element
        try:
            elements = self.driver.find_elements(By.TAG_NAME, "footer")
            if elements:
                self.driver.execute_script("arguments[0].scrollIntoView(true);", elements[0])
                self._print_status("🖱 Method 3: Scrolled to footer element")
                success = True
                time.sleep(1)
        except Exception as e:
            self._print_status(f"⚠️ Method 3 scroll failed: {str(e)}")
        
        return success


    # def click_show_more_with_multiple_methods(self):
    #     """Comprehensive approach to find and click 'Show more' button"""
    #     self._print_status("🔍 Attempting to find and click 'Show more' button")
        
    #     # Try specialized methods first (the new methods we added)
    #     if self.click_show_more_direct() or self.click_show_more_with_javascript():
    #         return True
        
    #     # Continue with existing methods but fix the invalid selectors
    #     button_selectors = [
    #         "button.show-more",
    #         "button.load-more", 
    #         "button[class*='more']",
    #         ".show-more",
    #         ".load-more",
    #         "[class*='more']",  # Any element with 'more' in class
    #         "a:has(> span:contains('Show more'))"  # For jQuery-enabled browsers
    #     ]
        
    #     for selector in button_selectors:
    #         try:
    #             # Skip the known invalid selectors
    #             if ":contains" in selector:
    #                 continue
                    
    #             self._print_status(f"🔍 Trying to find button with selector: {selector}")
    #             buttons = self.driver.find_elements(By.CSS_SELECTOR, selector)
    #             if buttons:
    #                 for button in buttons:
    #                     if button.is_displayed():
    #                         self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
    #                         time.sleep(0.5)
    #                         self.driver.execute_script("arguments[0].click();", button)
    #                         self.show_more_clicks += 1
    #                         self._print_status(f"🔄 Clicked button with selector: {selector} ({self.show_more_clicks} times)")
    #                         time.sleep(2)
    #                         return True
    #         except Exception as e:
    #             if "invalid selector" not in str(e).lower():
    #                 self._print_status(f"⚠️ Button selector failed for {selector}: {str(e)}")
        
    #     # Try XPath approaches - fixed for correct casing of "Show more"
    #     xpath_patterns = [
    #         "//button[contains(text(), 'Show more')]",  # Note lowercase 'm'
    #         "//a[contains(text(), 'Show more')]",
    #         "//div[contains(text(), 'Show more')]",
    #         "//span[contains(text(), 'Show more')]",
    #         "//*[contains(text(), 'Show more')]",  # Match any element
    #         "//div[contains(@class, 'more')]",
    #         "//a[contains(@class, 'more')]"
    #     ]
        
    #     for xpath in xpath_patterns:
    #         try:
    #             self._print_status(f"🔍 Trying button with XPath: {xpath}")
    #             buttons = self.driver.find_elements(By.XPATH, xpath)
    #             if buttons:
    #                 for button in buttons:
    #                     if button.is_displayed():
    #                         self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
    #                         time.sleep(0.5)
    #                         self.driver.execute_script("arguments[0].click();", button)
    #                         self.show_more_clicks += 1
    #                         self._print_status(f"🔄 Clicked with XPath: {xpath} ({self.show_more_clicks} times)")
    #                         time.sleep(2)
    #                         return True
    #         except Exception as e:
    #             self._print_status(f"⚠️ XPath approach failed for {xpath}: {str(e)}")
        
    #     # Take a debug screenshot if all methods fail
    #     self.take_screenshot_of_button_area()
    #     self._print_status("ℹ No 'Show more' button found with any method")
    #     return False


    # def extract_startup_data(self, card_selector):
    #     """Extract data from startup cards"""
    #     try:
    #         cards = self.driver.find_elements(By.CSS_SELECTOR, card_selector)
    #         self._print_status(f"📊 Found {len(cards)} startup cards to extract")
            
    #         for idx, card in enumerate(cards, 1):
    #             try:
    #                 data = {}
                    
    #                 # Extract company name with multiple approaches
    #                 try:
    #                     name_elements = card.find_elements(By.CSS_SELECTOR, "h3")
    #                     if name_elements:
    #                         data['name'] = name_elements[0].text.strip()
    #                     else:
    #                         # Try alternative selectors
    #                         name_element = card.find_element(By.CSS_SELECTOR, "[class*='name']")
    #                         data['name'] = name_element.text.strip()
    #                 except:
    #                     data['name'] = f"Unknown Company {idx}"
                    
    #                 # Extract description 
    #                 try:
    #                     description = card.find_element(By.XPATH, ".//div[contains(text(), 'What they do')]/following-sibling::div")
    #                     data['description'] = description.text.strip()
    #                 except:
    #                     try:
    #                         description = card.find_element(By.CSS_SELECTOR, ".description")
    #                         data['description'] = description.text.strip()
    #                     except:
    #                         data['description'] = ""
                    
    #                 # Extract funding info
    #                 try:
    #                     funding = card.find_element(By.XPATH, ".//div[contains(text(), 'Funding')]/following-sibling::div")
    #                     data['funding'] = funding.text.strip().split("·")[0].strip()
    #                 except:
    #                     data['funding'] = "Unknown"
                    
    #                 # Extract location
    #                 try:
    #                     location = card.find_element(By.XPATH, ".//*[contains(text(), 'HQ:')]")
    #                     data['location'] = location.text.replace("📍HQ:", "").strip()
    #                 except:
    #                     data['location'] = "San Francisco Bay Area"
                    
    #                 # Extract links
    #                 data['links'] = {}
    #                 try:
    #                     links = card.find_elements(By.TAG_NAME, "a")
    #                     for link in links:
    #                         link_text = link.text.strip() or "Website"
    #                         data['links'][link_text] = link.get_attribute("href")
    #                 except:
    #                     pass
                    
    #                 # Additional metadata
    #                 data['extracted_at'] = datetime.now().isoformat()
                    
    #                 self.startups_data.append(data)
                    
    #                 if idx % 10 == 0:
    #                     self._print_status(f"📥 Extracted {idx}/{len(cards)} startup cards")
                
    #             except Exception as e:
    #                 self._print_status(f"⚠️ Error extracting card {idx}: {str(e)}")
            
    #         self._print_status(f"✅ Successfully extracted {len(self.startups_data)} startup records")
    #         return len(self.startups_data) > 0
        
    #     except Exception as e:
    #         self._print_status(f"❌ Data extraction failed: {str(e)}")
    #         return False





    def schedule_runs(self, interval='weekly', day_of_week=0, hour=3, minute=0):
        """Schedule the scraper to run at specified intervals
        
        Args:
            interval: 'daily' or 'weekly'
            day_of_week: 0=Monday, 1=Tuesday, etc. (for weekly)
            hour: Hour in 24-hour format (0-23)
            minute: Minute (0-59)
        """
        import schedule
        
        def job():
            self._print_status(f"🕒 Running scheduled scrape at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            try:
                self.run_scraper()
                self._print_status(f"✅ Scheduled scrape completed successfully")
            except Exception as e:
                self._print_status(f"❌ Scheduled scrape failed: {str(e)}")
        
        if interval == 'daily':
            schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(job)
            self._print_status(f"📅 Scheduled daily runs at {hour:02d}:{minute:02d}")
        else:
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            schedule_day = getattr(schedule.every(), days[day_of_week].lower())
            schedule_day.at(f"{hour:02d}:{minute:02d}").do(job)
            self._print_status(f"📅 Scheduled weekly runs on {days[day_of_week]} at {hour:02d}:{minute:02d}")
        
        # Run once immediately
        job()
        
        # Keep the script running
        while True:
            schedule.run_pending()
            time.sleep(60)



    def schedule_daily_run(self, hour=3, minute=0):
        """Schedule daily runs using the schedule library"""
        import schedule
        
        def job():
            self._print_status("⏰ Running scheduled scrape...")
            try:
                self.run()
            except Exception as e:
                self._print_status(f"⚠️ Scheduled job failed: {str(e)}")
        
        schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(job)
        
        # Run immediately first time
        self._print_status(f"⏰ Next scrape scheduled for daily {hour:02d}:{minute:02d}")
        job()
        
        while True:
            schedule.run_pending()
            time.sleep(60)


    def click_show_more_optimized(self):
        """Direct click using the working selector from logs"""
        try:
            button = WebDriverWait(self.driver, 3).until(
                EC.element_to_be_clickable((By.XPATH, "//*[contains(text(), 'Show more')]"))
            )
            self.driver.execute_script("arguments[0].scrollIntoView(true);", button)
            self.driver.execute_script("arguments[0].click();", button)
            self.show_more_clicks += 1
            self._print_status(f"🔄 Clicked 'Show more' ({self.show_more_clicks} times)")
            time.sleep(1.5)  # Reduced from 3 seconds
            return True
        except Exception as e:
            self._print_status(f"⚠️ Show more click failed: {str(e)}")
            return False













    # def extract_startup_data2(self):
    #     """Extract all data from startup cards"""
    #     cards = self.driver.find_elements(By.CSS_SELECTOR, self.card_selector)
    #     self._print_status(f"📊 Extracting data from {len(cards)} startup cards")
        
    #     for idx, card in enumerate(cards, 1):
    #         try:
    #             # Company name
    #             name = card.find_element(By.CSS_SELECTOR, "h3").text.strip()
                
    #             # Description (What they do)
    #             description = card.find_element(By.XPATH, ".//div[contains(text(), 'What they do')]/following-sibling::div").text.strip()
                
    #             # Location
    #             location = ""
    #             try:
    #                 location_el = card.find_element(By.XPATH, ".//*[contains(text(), '📍HQ:')]")
    #                 location = location_el.text.replace("📍HQ:", "").strip()
    #             except Exception:
    #                 pass
                
    #             # Funding information
    #             funding = ""
    #             try:
    #                 funding_el = card.find_element(By.XPATH, ".//div[contains(text(), 'Funding')]/following-sibling::div")
    #                 funding = funding_el.text.strip()
    #             except Exception:
    #                 pass
                
    #             # Links
    #             links = {}
    #             link_elements = card.find_elements(By.CSS_SELECTOR, "a[href]")
    #             for link in link_elements:
    #                 text = link.text.strip() or "Company Website"
    #                 links[text] = link.get_attribute("href")
                
    #             self.startups_data.append({
    #                 "name": name,
    #                 "description": description,
    #                 "location": location,
    #                 "funding": funding,
    #                 "links": links,
    #                 "scraped_at": datetime.now().isoformat()
    #             })
                
    #             if idx % 10 == 0:
    #                 self._print_status(f"📥 Extracted {idx}/{len(cards)} startup cards")
                    
    #         except Exception as e:
    #             self._print_status(f"⚠️ Error extracting card {idx}: {str(e)}")
        
    #     self._print_status(f"✅ Successfully extracted {len(self.startups_data)} startup records")
    #     return self.startups_data












    

    def save_data(self):
        """Save extracted data to JSON and CSV files"""
        if not self.startups_data:
            self._print_status("⚠️ No data to save")
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # # Save to JSON
        # try:
        #     json_file = f"startups_{timestamp}.json"
        #     with open(json_file, "w", encoding="utf-8") as f:
        #         json.dump(self.startups_data, f, indent=4, ensure_ascii=False)
        #     self._print_status(f"💾 Saved {len(self.startups_data)} records to {json_file}")
        # except Exception as e:
        #     self._print_status(f"❌ JSON save failed: {str(e)}")
        

            # JSON
        json_path = os.path.join(self.output_dir, f"startups_{timestamp}.json")
        # CSV
        csv_path = os.path.join(self.output_dir, f"startups_{timestamp}.csv")

        # Save to CSV
        try:
            csv_file = f"startups_{timestamp}.csv"
            
            # Get all possible keys
            keys = set()
            for data in self.startups_data:
                keys.update(data.keys())
            
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(keys))
                writer.writeheader()
                
                # Handle the links dictionary for CSV format
                for data in self.startups_data:
                    row_data = data.copy()
                    if 'links' in row_data and isinstance(row_data['links'], dict):
                        row_data['links'] = json.dumps(row_data['links'])
                    writer.writerow(row_data)
            
            self._print_status(f"💾 Saved {len(self.startups_data)} records to {csv_file}")
            return True
        
        except Exception as e:
            self._print_status(f"❌ CSV save failed: {str(e)}")
            return False

    def run(self):
        """Main execution flow with comprehensive error handling"""
        try:
            # Setup browser
            if not self.setup_browser():
                return False
            
            # Navigate to target URL
            if not self.navigate_to_url():
                return False
            
            # Give page time to load
            self._print_status("⏳ Waiting for page to fully load...")
            time.sleep(5)
            
            # Identify card elements
            card_selector, cards = self.identify_card_elements()
            if not card_selector:
                self._print_status("❌ Could not identify card elements")
                return False
            
            # # Load more content
            # self._print_status("🔄 Loading more content...")
            # load_more_failed_count = 0
            # max_load_attempts = 30
            
            # while load_more_failed_count < 3 and self.show_more_clicks < max_load_attempts:
            #     # Scroll down
            #     if not self.scroll_with_multiple_methods():
            #         load_more_failed_count += 1
                
            #     # Click show more with enhanced methods
            #     if self.click_show_more_with_multiple_methods():
            #         load_more_failed_count = 0  # Reset counter on success
            #     else:
            #         load_more_failed_count += 1
                
            #     # Check for new cards
            #     new_cards = self.driver.find_elements(By.CSS_SELECTOR, card_selector)
            #     if len(new_cards) > len(cards):
            #         self._print_status(f"📈 Now showing {len(new_cards)} cards (was {len(cards)})")
            #         cards = new_cards
            #     else:
            #         # Try again with a refresh if no new cards loaded
            #         if load_more_failed_count >= 2:
            #             try:
            #                 self._print_status("🔄 Refreshing page to retry loading...")
            #                 self.driver.refresh()
            #                 time.sleep(5)  # Wait for refresh
            #                 load_more_failed_count = 0  # Reset counter
            #             except Exception:
            #                 pass
                


            # Optimized loading loop
            MAX_ATTEMPTS = 50  # Increased from 30
            last_card_count = 0
            stale_count = 0
            
            while self.show_more_clicks < MAX_ATTEMPTS and stale_count < 5:
                # Scroll directly to bottom
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
                
                # Click using optimized method
                if self.click_show_more_optimized():
                    stale_count = 0  # Reset counter on success
                else:
                    stale_count += 1
                
                # Check card count
                current_cards = self.driver.find_elements(By.CSS_SELECTOR, ".card")
                if len(current_cards) > last_card_count:
                    self._print_status(f"📈 Cards loaded: {len(current_cards)}")
                    last_card_count = len(current_cards)
                else:
                    self._print_status(f"♻ No new cards ({stale_count}/5)")
                
                time.sleep(1)  # Reduced from 2 seconds


                
            # Extract startup data
            if not self.extract_startup_data(card_selector):
                self._print_status("❌ Data extraction failed")
            
            # Save data
            self.save_data()
            
            self._print_status("✅ Scraping process completed")
            return True
            
        except Exception as e:
            self._print_status(f"🔥 Critical error: {str(e)}")
            if self.driver:
                self.driver.save_screenshot("error.png")
                self._print_status("📸 Error screenshot saved to 'error.png'")
            return False
            
        finally:
            if self.driver:
                self.driver.quit()
                self._print_status("🛑 Browser closed")

if __name__ == "__main__":

    
    scraper = TopStartupsScraper()
    scraper.schedule_daily_run(hour=3, minute=0)  # 3 AM daily
