import os
import time
import json
import csv
from seleniumbase import Driver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import re
from selenium.webdriver.remote.webelement import WebElement  # Add this import
import undetected_chromedriver as uc  # Import undetected_chromedriver
import random
from selenium.webdriver.remote.webelement import WebElement  # Add this import


import schedule
import time
from datetime import datetime

import sqlite3
import argparse


import logging
import smtplib
from email.mime.text import MIMEText

def setup_directories():
    """Create required directories with absolute paths"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'bay_area_startups')
    
    # Create directory and parents if needed
    os.makedirs(data_dir, exist_ok=True)
    
    # Verify creation
    if not os.path.exists(data_dir):
        raise RuntimeError(f"Failed to create directory: {data_dir}")
    
    return data_dir

data_dir = setup_directories()

# Then configure logging
logging.basicConfig(
    filename=os.path.join(data_dir, 'scraper.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def save_data_files(startups, data_dir):
    """Save data to JSON and CSV with proper directory handling"""
    # JSON
    json_path = os.path.join(data_dir, 'startups.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(startups, f, indent=2, ensure_ascii=False)
    
    # CSV
    csv_path = os.path.join(data_dir, 'startups.csv')
    if startups:
        fieldnames = list(startups[0].keys())
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(startups)




# Modified logging configuration
def setup_logging(data_dir):
    """Configure logging with directory creation"""
    log_path = os.path.join(data_dir, 'scraper.log')
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def scrape_with_retry(url, max_retries=3, delay=5):
    """Scrape with retry logic and backoff"""
    for attempt in range(1, max_retries+1):
        try:
            driver = configure_driver()
            driver.get(url)
            # Rest of your scraping logic
            data = extract_startup_data(driver)  # Assuming extract_startup_data is the intended function
            return data
        except Exception as e:
            logging.error(f"Attempt {attempt}/{max_retries} failed: {str(e)}")
            if driver:
                driver.save_screenshot(f'error_screenshot_attempt_{attempt}.png')
                driver.quit()
            if attempt < max_retries:
                sleep_time = delay * attempt  # Exponential backoff
                logging.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logging.critical("All retry attempts failed")
                raise

def check_website_structure(driver):
    """Check if website structure has changed"""
    try:
        # Look for key elements that should be present
        driver.find_element(By.XPATH, "//*[contains(text(), 'What they do')]")
        return True
    except Exception:
        logging.warning("Website structure may have changed - key elements not found")
        driver.save_screenshot('structure_change_detected.png')
        send_notification(
            "TopStartups.io Structure Change Detected",
            "The scraper detected a possible change in the website structure. Manual inspection needed."
        )
        return False



# Setup logging
logging.basicConfig(
    filename='bay_area_startups/scraper.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def send_notification(subject, message):
    """Send email notification"""
    # Configure your email settings
    sender = "your_email@example.com"
    recipient = "your_email@example.com"
    password = "your_password"
    
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = recipient
    
    try:
        with smtplib.SMTP_SSL('smtp.example.com', 465) as server:
            server.login(sender, password)
            server.send_message(msg)
        logging.info("Notification email sent successfully")
    except Exception as e:
        logging.error(f"Failed to send notification: {str(e)}")



def scheduled_scraping():
    """Function to run the scraping on schedule"""
    print(f"Starting scheduled scraping at {datetime.now()}")
    try:
        count = scrape_topstartups()
        print(f"Scheduled scraping completed. Found {count} startups.")
    except Exception as e:
        print(f"Scheduled scraping failed: {str(e)}")


def update_data_files_from_db(conn, data_dir):
    """Generate fresh JSON/CSV from database"""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM startups")
    columns = [col[0] for col in cursor.description]
    
    startups = []
    for row in cursor.fetchall():
        startups.append(dict(zip(columns, row)))
    
    save_data_files(startups, data_dir)


def setup_schedule():
    """Configure weekly schedule with randomization"""
    # Randomize start time between 8:00-10:00 AM
    import random
    start_hour = 8
    start_minute = random.randint(0, 120)
    
    # Weekly schedule with jitter
    schedule.every().monday.at(f"{start_hour + start_minute//60}:{start_minute%60:02d}").do(scheduled_scraping)
    
    # Add Wednesday check for mid-week updates
    schedule.every().wednesday.at("12:00").do(check_for_urgent_updates)
    
    logging.info(f"Scheduler configured for Mondays at {start_hour + start_minute//60}:{start_minute%60:02d}")
    
    while True:
        schedule.run_pending()
        time.sleep(60)

def check_for_urgent_updates():
    """Lightweight check for unexpected updates"""
    try:
        driver = configure_driver()
        driver.get("https://topstartups.io/latest")
        if "New Startups Added" in driver.page_source:
            send_notification("Urgent Update Detected", "New startups added mid-week")
    except Exception as e:
        logging.warning(f"Mid-week check failed: {str(e)}")
    finally:
        driver.quit()

def validate_initial_collection(conn):
    """Ensure minimum data quality before scheduling"""
    cursor = conn.cursor()
    
    # Check for at least 500 startups (site's advertised minimum)
    cursor.execute("SELECT COUNT(*) FROM startups")
    count = cursor.fetchone()[0]
    
    if count < 500:
        raise ValueError(f"Initial collection only found {count} startups - below site minimum")
    
    # Check for recent entries
    cursor.execute("SELECT MAX(date_added) FROM startups")
    latest_date = cursor.fetchone()[0]
    
    if latest_date < datetime.now().strftime('%Y-%m-%d'):
        raise ValueError("Initial collection data appears outdated")


def setup_database(data_dir):
    """Setup SQLite database with proper path handling"""
    db_path = os.path.join(data_dir, 'startups.db')
    conn = sqlite3.connect(db_path)
    
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS startups (
        id INTEGER PRIMARY KEY,
        name TEXT UNIQUE,
        description TEXT,
        hq TEXT,
        funding TEXT,
        website TEXT,
        employees_link TEXT,
        reviews_link TEXT,
        date_added TEXT DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    return conn


def save_to_database(conn, startups):
    """Improved upsert functionality"""
    cursor = conn.cursor()
    for startup in startups:
        cursor.execute('''
        INSERT OR REPLACE INTO startups 
        (name, description, hq, funding, website, employees_link, reviews_link, date_added)
        VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT date_added FROM startups WHERE name = ?), DATE('now')))
        ''', (
            startup['name'],
            startup['description'],
            startup['hq'],
            startup['funding'],
            startup['website'],
            startup['employees_link'],
            startup['reviews_link'],
            startup['name']  # For COALESCE
        ))
    conn.commit()


def configure_driver():
    """Configure driver with improved compatibility"""
    try:
        # First try using seleniumbase Driver
        driver = Driver(
            browser="chrome",
            headless=False,
            undetectable=True
        )
        return driver
    except Exception as e:
        print(f"SeleniumBase driver failed: {str(e)}, trying undetected_chromedriver")
        
        # Fallback to direct undetected_chromedriver
        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        
        # Don't specify version_main to let it auto-detect
        driver = uc.Chrome(
            options=options,
            headless=False,
            use_subprocess=True  # Changed to True for better stability
        )
        
        # Add stealth settings
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        return driver


def extract_startup_data(driver):
    """Robust extraction using proper WebElement handling"""
    try:
        # First try Selenium native method
        cards = driver.find_elements(By.CSS_SELECTOR, 'div.card')
        if not cards:
            # Fallback to BS4
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            cards = soup.find_all('div', class_='card')
            
        startups = []
        
        for card in cards:
            try:
                if isinstance(card, WebElement):  # Proper Selenium element check
                    # Use text attribute for Selenium elements
                    text = card.text
                    links = card.find_elements(By.TAG_NAME, 'a')
                else:  # BS4 element
                    text = card.get_text()
                    links = card.find_all('a')
                
                # Extract data using regex (improved pattern)
                startup = {
                    'name': re.search(r'^(.*?)\nWhat they do', text).group(1).strip() if text else '',
                    'description': re.search(r'What they do:\s*(.*?)\nQuick facts', text, re.DOTALL).group(1).strip() if text else '',
                    'hq': re.search(r'📍HQ:\s*(.*?)\n', text).group(1).strip() if text else '',
                    'funding': re.search(r'Funding:\s*(.*?)\nTake action', text, re.DOTALL).group(1).strip() if text else '',
                    'website': '',
                    'employees_link': '',
                    'reviews_link': ''
                }
                
                # Process links with proper attribute handling
                for link in links:
                    href = link.get_attribute('href') if isinstance(link, WebElement) else link.get('href')
                    link_text = link.text.lower() if isinstance(link, WebElement) else link.text.lower()
                    
                    if 'company site' in link_text:
                        startup['website'] = href
                    elif 'who works here' in link_text:
                        startup['employees_link'] = href
                    elif 'reviews' in link_text:
                        startup['reviews_link'] = href
                
                if startup['name']:
                    startups.append(startup)
                    
            except Exception as e:
                logging.error(f"Error processing card: {str(e)}")
                continue
                
        return startups
        
    except Exception as e:
        logging.error(f"Extraction error: {str(e)}")
        return []



def load_existing_data(data_dir):
    """Load from JSON with directory awareness"""
    json_path = os.path.join(data_dir, 'startups.json')
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def identify_new_startups(current_data, new_data):
    """Identify newly added startups"""
    existing_names = {startup['name'] for startup in current_data}
    return [startup for startup in new_data if startup['name'] not in existing_names]




# if __name__ == "__main__":
#     max_attempts = 3
    
#     for attempt in range(1, max_attempts+1):
#         print(f"\n=== Attempt {attempt}/{max_attempts} ===\n")
#         count = scrape_topstartups()
#         if count >= 545:
#             print(f"Success! Scraped all {count} startups")
#             break
#         elif count > 0:
#             print(f"Partial success: Scraped {count} startups. Trying again...")
#         else:
#             print("Scraping failed completely. Trying again...")
        
#         if attempt < max_attempts:
#             time.sleep(5)
    
#     print("\nScraping process complete")



def scrape_topstartups(data_dir):
    driver = configure_driver()
    url = "https://topstartups.io/?hq_location=San+Francisco+Bay+Area&sort=funding"
    
    try:
        print("Navigating to URL...")
        driver.get(url)
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'What they do')]"))
        )

        last_count = 0
        max_attempts = 50
        progress_stalled = 0
        startups = []

        for attempt in range(1, max_attempts+1):
            print(f"\n--- Attempt {attempt}/{max_attempts} ---")
            
            # Always get fresh elements
            current_startups = extract_startup_data(driver)
            new_count = len(current_startups)
            
            if new_count > last_count:
                print(f"Found {new_count - last_count} new startups")
                startups = current_startups
                last_count = new_count
                progress_stalled = 0
            else:
                progress_stalled += 1
                print(f"Stalled progress: {progress_stalled}/3")

            # Exit conditions
            if progress_stalled >= 3:
                print("No progress for 3 attempts. Assuming completion.")
                break
            if new_count >= 545:
                print("Reached target startup count")
                break

            # Improved click handling with multiple strategies
            try:
                # Strategy 1: Try finding the button with a more flexible XPath
                button_xpath_options = [
                    "//button[contains(., 'Show more')]",
                    "//button[contains(text(), 'Show more')]",
                    "//button[contains(@class, 'show-more')]",
                    "//button[contains(@class, 'load-more')]",
                    "//div[contains(@class, 'pagination')]//button"
                ]
                
                button = None
                for xpath in button_xpath_options:
                    try:
                        button = WebDriverWait(driver, 5).until(
                            EC.element_to_be_clickable((By.XPATH, xpath))
                        )
                        if button:
                            print(f"Found button using: {xpath}")
                            break
                    except:
                        continue
                
                if not button:
                    # Strategy 2: Try scrolling to the bottom to trigger lazy loading
                    print("Button not found, trying to scroll to bottom")
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(3)
                    
                    # Check if scrolling loaded more content
                    new_startups_after_scroll = extract_startup_data(driver)
                    if len(new_startups_after_scroll) > new_count:
                        print(f"Scrolling loaded {len(new_startups_after_scroll) - new_count} more startups")
                        continue
                    else:
                        # Strategy 3: Try clicking any button at the bottom of the page
                        buttons = driver.find_elements(By.TAG_NAME, "button")
                        bottom_buttons = [b for b in buttons if driver.execute_script(
                            "return (window.innerHeight + window.scrollY) >= arguments[0].getBoundingClientRect().top", b
                        )]
                        
                        if bottom_buttons:
                            button = bottom_buttons[-1]
                            print("Trying to click the bottommost button")
                        else:
                            print("No buttons found at the bottom of the page")
                            break
                
                # Now try to click the button with multiple methods
                if button:
                    # First scroll to make sure it's in view
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center', behavior: 'smooth'});", button)
                    time.sleep(1)
                    
                    try:
                        # Try regular click first
                        button.click()
                        print("Clicked button normally")
                    except:
                        try:
                            # Try JavaScript click
                            driver.execute_script("arguments[0].click();", button)
                            print("Clicked button via JavaScript")
                        except:
                            # Try moving to element and clicking
                            from selenium.webdriver.common.action_chains import ActionChains
                            ActionChains(driver).move_to_element(button).click().perform()
                            print("Clicked button via ActionChains")
                    
                    # Wait for new content to load
                    time.sleep(3)
                else:
                    print("Could not find a clickable button")
                    break
                    
            except Exception as e:
                print(f"Show more click failed: {str(e)}")
                # Take a screenshot to debug
                screenshot_path = os.path.join(data_dir, f'error_screenshot_{attempt}.png')
                driver.save_screenshot(screenshot_path)
                print(f"Screenshot saved to {screenshot_path}")
                
                # One last attempt - try to find any interactive element at the bottom
                try:
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(2)
                    # Continue the loop - maybe scrolling triggered content loading
                    continue
                except:
                    break

        # Final save
        save_data_files(startups, data_dir)
        return len(startups)

    except Exception as e:
        print(f"Critical error: {str(e)}")
        return 0
    finally:
        if driver:
            driver.quit()


def configure_driver():
    """Enhanced driver configuration with anti-detection measures"""
    options = uc.ChromeOptions()
    options.add_argument("--no-first-run")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-web-security")
    options.add_argument("--allow-running-insecure-content")
    options.add_argument("--disable-webgl")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--disable-notifications")
    options.add_argument("--mute-audio")
    
    # Randomized viewport dimensions
    viewport_w = random.randint(1200, 1920)
    viewport_h = random.randint(800, 1080)
    options.add_argument(f"--window-size={viewport_w},{viewport_h}")

    # Configure undetected ChromeDriver
    driver = uc.Chrome(
        options=options,
        headless=False,
        use_subprocess=False,
        version_main=114  # Match your Chrome version
    )

    # Stealth configurations
    driver.execute_cdp_cmd('Network.setUserAgentOverride', {
        "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.6045.159 Safari/537.36"
    })
    
    # Disable WebDriver flag
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver


# Modify the main block
if __name__ == "__main__":
    # Setup directories FIRST
    data_dir = setup_directories()
    
    # Configure logging AFTER directories exist
    setup_logging(data_dir)
    
    # Initialize database
    conn = setup_database(data_dir)
    
    # Argument parsing
    parser = argparse.ArgumentParser(description='TopStartups.io Scraper')
    parser.add_argument('--schedule', action='store_true', help='Run continuously')
    args = parser.parse_args()


    
    try:
        print("Starting TopStartups.io scraper...")
        count = scrape_topstartups(data_dir)
        print(f"Scraping completed. Found {count} startups.")
    except Exception as e:
        print(f"Critical error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()

    try:
        if args.schedule:
            # Initial collection
            existing = load_existing_data(data_dir)
            new_count = scrape_topstartups(data_dir)
            new_data = load_existing_data(data_dir)
            
            # Database update
            if new_data:
                save_to_database(conn, new_data)
                update_data_files_from_db(conn, data_dir)
            
            # Start scheduler
            schedule.every().monday.at("08:00").do(scrape_topstartups, data_dir=data_dir)
            while True:
                schedule.run_pending()
                time.sleep(60)
        else:
            # Single run
            scrape_topstartups(data_dir)
            
    except Exception as e:
        logging.critical(f"Main execution failed: {str(e)}")
        send_notification("Scraper Crash", f"Error: {str(e)}")
    finally:
        conn.close()

