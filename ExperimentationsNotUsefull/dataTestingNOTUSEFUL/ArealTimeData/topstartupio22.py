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
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import (TimeoutException, 
                                        NoSuchElementException,
                                        StaleElementReferenceException,
                                        ElementClickInterceptedException)
import os
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('scraper.log'), logging.StreamHandler()]
)

class TopStartupsScraper:
    def __init__(self, url="https://topstartups.io/?hq_location=San+Francisco+Bay+Area&sort=funding", headless=True):
        self.url = url
        self.headless = headless
        self.driver = None
        self.startups_data = []
        self.show_more_clicks = 0
        self.scroll_attempts = 0
        self.card_selector = None
        self.output_dir = "topstartup_io_real_time_data"
        self._create_output_dir()

    def _print_status(self, message):
        """Visual feedback with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

    def _create_output_dir(self):
        """Create output directory"""
        try:
            pass  # Add your code here
        except Exception as e:
            self._print_status(f"⚠️ An error occurred: {str(e)}")
            os.makedirs(self.output_dir, exist_ok=True)
            self._print_status(f"📁 Created output directory: {self.output_dir}")
        except Exception as e:
            self._print_status(f"⚠️ Directory creation failed: {str(e)}")
            self.output_dir = "."

 

    def setup_browser(self):
        """Fixed headless configuration"""
        chrome_options = webdriver.ChromeOptions()
        
        # Anti-detection measures
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        
        # Headless configuration
        if self.headless:
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--window-size=1920,1080")
        else:
            chrome_options.add_argument("--start-maximized")
        
        # Essential performance settings
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--single-process")
        
        # Modern user agent
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        # Connection stability settings
        chrome_options.add_argument("--disable-http2")
        chrome_options.add_argument("--disable-features=NetworkService")
        
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.set_page_load_timeout(45)
            return True
        except Exception as e:
            self._print_status(f"❌ Browser setup failed: {str(e)}")
            return False





    def navigate_to_url(self):
        """Robust navigation with connection retries"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.driver.get(self.url)
                self._print_status(f"🌐 Navigation attempt {attempt+1}/{max_retries}")
                
                # Immediate connection check
                if "driver" in self.driver.page_source.lower():
                    raise ConnectionError("Initial connection failed")
                    
                return True
            except Exception as e:
                if attempt == max_retries - 1:
                    self._print_status(f"❌ Final navigation failure: {str(e)}")
                    return False
                self._print_status(f"⚠️ Retrying navigation ({attempt+1}/{max_retries})")
                self.driver.quit()
                self.setup_browser()


    def wait_for_page_load(self):
        """Wait for page to be fully loaded"""
        try:
            # Wait for document ready state
            WebDriverWait(self.driver, 10).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            
            # Wait for network activity to settle
            WebDriverWait(self.driver, 5).until(
                lambda d: d.execute_script("""
                    return window.performance.getEntriesByType('resource')
                        .filter(r => !r.responseEnd && Date.now() - r.startTime < 5000).length == 0
                """)
            )
            
            self._print_status("✅ Page fully loaded")
            return True
        except Exception as e:
            self._print_status(f"⚠️ Page load wait timeout: {str(e)}")
            return False

    def identify_card_elements(self):
        """Enhanced card detection for topstartups.io"""
        # Execute JavaScript to find the actual card container
        container_script = """
        return (function() {
            // Find elements containing startup information
            const possibleCards = Array.from(document.querySelectorAll('div')).filter(div => {
                // Look for divs that contain company name, funding info, etc.
                return (div.querySelector('h3') || 
                    div.querySelector('[class*="company"]') || 
                    div.querySelector('[class*="startup"]')) &&
                    div.offsetHeight > 100 && 
                    div.offsetWidth > 200;
            });
            
            // Return the most common parent class name
            if (possibleCards.length > 0) {
                return possibleCards[0].getAttribute('class') || 'div';
            }
            return null;
        })();
        """
        
        try:
            card_class = self.driver.execute_script(container_script)
            if card_class:
                self._print_status(f"✅ Identified cards with class: {card_class}")
                return f"div.{card_class.split(' ')[0]}", self.driver.find_elements(By.CSS_SELECTOR, f"div.{card_class.split(' ')[0]}")
        except Exception as e:
            self._print_status(f"⚠️ Card detection script failed: {str(e)}")
        
        # Fallback to visual detection
        return "div:not(:empty):has(h3, div.description)", self.driver.find_elements(By.CSS_SELECTOR, "div:not(:empty):has(h3)")

    def _get_visible_cards(self):
        """Robust element retrieval with health checks"""
        try:
            # Verify browser connection
            self.driver.execute_script("return 1")
            
            elements = self.driver.find_elements(By.CSS_SELECTOR, self.card_selector)
            if not elements:
                raise NoSuchElementException("No cards found")
                
            return elements
        except (StaleElementReferenceException, NoSuchElementException):
            self._print_status("⚠️ Element staleness detected, refreshing...")
            return self.driver.find_elements(By.CSS_SELECTOR, self.card_selector)
        except Exception as e:
            self._print_status(f"🚨 Critical element retrieval error: {str(e)}")
            self.driver.quit()
            raise

    def _smart_scroll(self):
        """Hybrid scrolling approach with multiple techniques"""
        # Technique 1: Progressive scroll increments
        for percent in [0.25, 0.5, 0.75, 1.0, 1.1]:
            self.driver.execute_script(
                f"window.scrollTo(0, document.body.scrollHeight * {percent});"
            )
            time.sleep(0.3)

        # Technique 2: Element-focused scrolling
        try:
            last_card = self._get_visible_cards()[-1]
            self.driver.execute_script(
                "arguments[0].scrollIntoView({behavior: 'smooth', block: 'end'});", 
                last_card
            )
        except Exception:
            pass

        # Technique 3: Pixel-based scroll simulation
        self.driver.execute_script("window.scrollBy(0, 500);")

        # Technique 4: JavaScript polyfill for older sites
        self.driver.execute_script(
            """window.__originalScroll = window.scrollTo;
            window.scrollTo = function(x,y) {
                document.documentElement.scrollTop = y;
            };"""
        )

    def click_show_more_optimized(self):
        """Modern button detection with live DOM search"""
        try:
            # Find by visible text using XPath
            button = WebDriverWait(self.driver, 2).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//button[normalize-space()='Show more' or normalize-space()='Load more']")
                )
            )
            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", button)
            button.click()
            self.show_more_clicks += 1
            self._print_status(f"🔄 Clicked 'Show more' ({self.show_more_clicks})")
            time.sleep(1)
            return True
        except Exception:
            pass
        
        # JavaScript fallback for SPAs
        try:
            result = self.driver.execute_script(r"""
                const buttons = Array.from(document.querySelectorAll('button'));
                const target = buttons.find(b => 
                    b.textContent.match(/show\s*more/gi) && 
                    getComputedStyle(b).display !== 'none'
                );
                if (target) {
                    target.click();
                    return true;
                }
                return false;
            """)

            if result:
                self.show_more_clicks += 1
                self._print_status(f"🎯 Clicked via JS ({self.show_more_clicks})")
                return True
        except Exception as e:
            self._print_status(f"⚠️ JS click failed: {str(e)}")
        
        return False


    def load_all_content(self):
        """Enhanced loading with DOM stability checks"""
        self._print_status("🌀 Starting advanced content loading")
        last_count = 0
        stale_count = 0
        max_stale = 10  # Increased from 5

        while stale_count < max_stale:
            # Add visual feedback during loading
            self.driver.execute_script(
                """document.body.style.border = '2px solid #4CAF50';
                setTimeout(() => document.body.style.border = '', 500);"""
            )

            # Hybrid scroll approach
            self._smart_scroll()

            # Wait for network stability
            WebDriverWait(self.driver, 10).until(
                lambda d: d.execute_script("""
                    return window.performance.getEntriesByType('resource')
                        .filter(r => !r.responseEnd).length === 0
                """)
            )

            # Check for new content
            current_cards = self._get_visible_cards()
            current_count = len(current_cards)
            
            # Add dynamic delay based on content growth
            if current_count > last_count:
                growth = current_count - last_count
                delay = max(1.5, 3 - (growth * 0.1))
                time.sleep(delay)
                last_count = current_count
                stale_count = 0
            else:
                stale_count += 1
                time.sleep(1.5)

            # Final check for infinite scroll endpoints
            if self.driver.execute_script(
                "return document.documentElement.scrollHeight <= window.innerHeight + window.pageYOffset + 100"
            ):
                break





    def load_all_content(self):
        """Intelligent loading with DOM stabilization"""
        self._print_status("🌀 Starting adaptive content loading")
        
        last_count = 0
        stale_count = 0
        max_stale = 5
        scroll_pattern = [0.7, 1.0, 0.3]  # Scroll position sequence
        
        while stale_count < max_stale:
            # Strategic scrolling
            scroll_pos = scroll_pattern[stale_count % len(scroll_pattern)]
            self.driver.execute_script(
                f"window.scrollTo(0, document.body.scrollHeight * {scroll_pos});"
            )
            
            # DOM stabilization technique
            self.driver.execute_script("""
                const body = document.body;
                const original = body.style.minHeight;
                body.style.minHeight = '100vh';
                setTimeout(() => body.style.minHeight = original, 50);
            """)
            
            # Attempt button click
            if self.click_show_more_optimized():
                stale_count = 0
                time.sleep(1.5)  # Network quiet period
            else:
                stale_count += 1
            
            # Validate content growth
            current_cards = self._get_visible_cards()
            current_count = len(current_cards)
            
            if current_count > last_count:
                self._print_status(f"📊 Content growth: {current_count} (+{current_count - last_count})")
                last_count = current_count
                stale_count = 0
            else:
                self._print_status(f"🔄 Stale iteration {stale_count}/{max_stale}")
            
            # Final content check
            if stale_count == max_stale - 1:
                final_check = self.driver.execute_script("""
                    return document.documentElement.scrollHeight > 
                        Math.max(document.body.scrollHeight, document.documentElement.clientHeight)
                """)
                if not final_check:
                    break

        self._print_status(f"🏁 Final card count: {last_count}")
        return last_count > 0

    def extract_startup_data(self):
        """Extract startup data from cards"""
        try:
            if not self.card_selector:
                self._print_status("❌ No card selector defined")
                return False

            cards = self.driver.find_elements(By.CSS_SELECTOR, self.card_selector)
            self._print_status(f"🔍 Starting data extraction from {len(cards)} cards")

            for idx, card in enumerate(cards, 1):
                try:
                    data = {
                        "name": self._extract_text(card, "h3"),
                        "description": self._extract_text(card, "div.what-they-do"),
                        "location": self._clean_location(
                            self._extract_text(card, "div.hq-location")
                        ),
                        "funding": self._clean_funding(
                            self._extract_text(card, "div.funding")
                        ),
                        "links": self._extract_links(card),
                        "timestamp": datetime.now().isoformat()
                    }
                    self.startups_data.append(data)

                    if idx % 5 == 0:
                        self._print_status(f"📥 Processed {idx}/{len(cards)} cards")

                except Exception as e:
                    self._print_status(f"⚠️ Error processing card {idx}: {str(e)}")

            self._print_status(f"✅ Successfully extracted {len(self.startups_data)} records")
            return True

        except Exception as e:
            self._print_status(f"🔥 Extraction failed: {str(e)}")
            return False

    def _extract_text(self, element, selector):
        """Modern selector patterns with XPath fallbacks"""
        selectors = {
            "h3": [
                "h3.company-title", 
                "h3.startup-name",
                "//h3[contains(@class, 'title')]",  # XPath fallback
            ],
            "div.what-they-do": [
                "div.description", 
                "div.bio",
                "//div[contains(text(), 'description')]",  # XPath text match
            ],
            "div.hq-location": [
                "div.location", 
                "div.hq-info",
                "//*[contains(text(), '📍')]",  # Emoji detection
            ],
            "div.funding": [
                "div.funding", 
                "div.financials",
                "//div[contains(text(), '$')]",  # Currency symbol detection
            ]
        }
        
        # Try CSS selectors first
        for sel in selectors.get(selector, [selector]):
            try:
                if sel.startswith("//"):
                    return element.find_element(By.XPATH, sel).text.strip()
                else:
                    return element.find_element(By.CSS_SELECTOR, sel).text.strip()
            except:
                continue
        
        # Final fallback: Text pattern matching
        try:
            if selector == "div.funding":
                return re.search(r"\$\d+[\d,]+", element.text).group()
        except:
            return ""


    def _clean_location(self, text):
        """Clean location data"""
        return text.replace("📍HQ:", "").strip()

    def _clean_funding(self, text):
        """Clean funding data"""
        return text.split("·")[0].strip()

    def _extract_links(self, element):
        """Extract all links from a card"""
        links = {}
        try:
            elements = element.find_elements(By.CSS_SELECTOR, "a[href]")
            for el in elements:
                url = el.get_attribute("href")
                text = el.text.strip() or f"link_{len(links)+1}"
                links[text] = url
        except:
            pass
        return links

    def save_data(self):
        """Save extracted data to JSON and CSV files"""
        if not self.startups_data:
            self._print_status("⚠️ No data to save")
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save to JSON
        json_path = os.path.join(self.output_dir, f"startups_{timestamp}.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.startups_data, f, indent=2, ensure_ascii=False)
            self._print_status(f"💾 Saved JSON: {json_path}")
        except Exception as e:
            self._print_status(f"❌ JSON save failed: {str(e)}")

        # Save to CSV
        csv_path = os.path.join(self.output_dir, f"startups_{timestamp}.csv")
        try:
            keys = set()
            for data in self.startups_data:
                keys.update(data.keys())
            
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(keys))
                writer.writeheader()
                
                for data in self.startups_data:
                    row_data = data.copy()
                    if 'links' in row_data and isinstance(row_data['links'], dict):
                        row_data['links'] = json.dumps(row_data['links'])
                    writer.writerow(row_data)
            
            self._print_status(f"💾 Saved CSV: {csv_path}")
            return True
        
        except Exception as e:
            self._print_status(f"❌ CSV save failed: {str(e)}")
            return False

    def run(self):




        # try:
        #     if not self.setup_browser():
        #         return False
                
        #     # Add page load verification
        #     WebDriverWait(self.driver, 30).until(
        #         lambda d: d.execute_script(
        #             "return document.readyState === 'complete' && "
        #             "document.body.innerText.length > 100"
        #         )
        #     )
            
        #     # Add visual debug markers
        #     self.driver.execute_script(
        #         """document.body.style.outline = '2px solid red';
        #         document.documentElement.style.outline = '2px solid blue';"""
        #     )
        # except TimeoutException:
        #     self._print_status("❌ Page load verification failed")
           
        try:
            if not self.setup_browser():
                return False
                
            # Immediate alive check
            self.driver.execute_script("return 1")
            
            # Navigation with retries
            if not self.navigate_to_url():
                return False

            # Visual confirmation
            self.driver.save_screenshot("initial_load.png")
            
            # Progressive loading checks
            WebDriverWait(self.driver, 30).until(
                lambda d: d.execute_script("""
                    return document.body.scrollHeight > 500 && 
                    document.querySelectorAll('div').length > 10
                """)
            )
        except TimeoutException:
            self._print_status("❌ Page load verification failed")  
            
           
           
           
           
            return False

if __name__ == "__main__":
    scraper = TopStartupsScraper(headless=False)  # Toggle visibility
    scraper.run()