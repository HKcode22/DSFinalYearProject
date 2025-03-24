import requests
from bs4 import BeautifulSoup
import json
import time

def scrape_yc_companies(batch="W24"):
    """
    Scrape YCombinator companies from a specific batch.
    
    Args:
        batch (str): The batch code (e.g., "W24" for Winter 2024)
    
    Returns:
        list: A list of dictionaries containing company data
    """
    url = f"https://www.ycombinator.com/companies?batch={batch}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to fetch data: {response.status_code}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    companies = []
    
    # Find company cards - adjust the selector based on actual HTML structure
    company_cards = soup.select('div.CompanyCard_root__hK85u')
    
    for card in company_cards:
        name_element = card.select_one('h3.CompanyCard_name__eAkro')
        desc_element = card.select_one('div.CompanyCard_tagline__MZrn7')
        url_element = card.select_one('a.CompanyCard_companyLink__L3crw')
        
        name = name_element.text.strip() if name_element else "Unknown"
        description = desc_element.text.strip() if desc_element else "No description"
        company_url = url_element['href'] if url_element and 'href' in url_element.attrs else None
        
        company_data = {
            'name': name,
            'description': description,
            'url': f"https://www.ycombinator.com{company_url}" if company_url else None,
            'batch': batch
        }
        
        companies.append(company_data)
        
    return companies

if __name__ == "__main__":
    # Scrape multiple batches
    all_companies = []
    batches = ["W24", "S23", "W23"]
    
    for batch in batches:
        print(f"Scraping batch {batch}...")
        companies = scrape_yc_companies(batch)
        all_companies.extend(companies)
        print(f"Found {len(companies)} companies in batch {batch}")
        time.sleep(2)  # Be respectful with rate limiting
    
    # Save to JSON file
    with open('yc_companies.json', 'w') as f:
        json.dump(all_companies, f, indent=2)
    
    print(f"Total companies scraped: {len(all_companies)}")
    print("Data saved to yc_companies.json")


# Install the package
# pip install ycombinator-scraper

from ycombinator_scraper import YComboScraper

# Initialize scraper
scraper = YComboScraper()

# Get jobs from specific companies
jobs = scraper.get_jobs(companies=["openai", "anthropic"])

# Export to CSV
scraper.export_to_csv(jobs, "yc_jobs.csv")


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import json
import random

def scrape_crunchbase_company(company_slug):
    """
    Scrape company data from Crunchbase using Selenium.
    
    Args:
        company_slug (str): The company identifier in Crunchbase URL
        
    Returns:
        dict: Company information
    """
    # Set up Chrome options
    chrome_options = Options()
    # Uncomment if you want to run in headless mode
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # Use a realistic user agent
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")
    
    # Initialize the WebDriver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    
    company_data = {}
    
    try:
        # Navigate to the company page
        url = f"https://www.crunchbase.com/organization/{company_slug}"
        driver.get(url)
        
        # Handle possible Cloudflare or other challenges
        # Sometimes just waiting helps bypass simple checks
        time.sleep(random.uniform(5, 10))
        
        # Wait for the content to load
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "h1.profile-name"))
        )
        
        # Extract basic company information
        company_data['name'] = driver.find_element(By.CSS_SELECTOR, "h1.profile-name").text.strip()
        
        # Extract description
        try:
            description = driver.find_element(By.CSS_SELECTOR, "div.description").text.strip()
            company_data['description'] = description
        except:
            company_data['description'] = "Not available"
        
        # Extract website
        try:
            website = driver.find_element(By.CSS_SELECTOR, "a.website-link").get_attribute("href")
            company_data['website'] = website
        except:
            company_data['website'] = "Not available"
        
        # Extract funding information
        try:
            funding_elements = driver.find_elements(By.CSS_SELECTOR, "div.funding-rounds")
            funding_data = []
            for element in funding_elements:
                funding_data.append(element.text.strip())
            company_data['funding'] = funding_data
        except:
            company_data['funding'] = []
        
        # Extract team members
        try:
            team_elements = driver.find_elements(By.CSS_SELECTOR, "div.team-members")
            team_data = []
            for element in team_elements:
                team_data.append(element.text.strip())
            company_data['team'] = team_data
        except:
            company_data['team'] = []
            
        return company_data
    
    except Exception as e:
        print(f"Error scraping {company_slug}: {str(e)}")
        return {"error": str(e)}
    
    finally:
        driver.quit()

def scrape_multiple_companies(company_slugs):
    """
    Scrape multiple companies with delay between requests.
    
    Args:
        company_slugs (list): List of company slugs to scrape
        
    Returns:
        dict: Dictionary with company data
    """
    results = {}
    
    for slug in company_slugs:
        print(f"Scraping {slug}...")
        results[slug] = scrape_crunchbase_company(slug)
        
        # Random delay to avoid being blocked
        delay = random.uniform(30, 60)
        print(f"Waiting {delay:.2f} seconds before next request...")
        time.sleep(delay)
    
    return results

if __name__ == "__main__":
    # Example company slugs
    companies = ["openai", "anthropic", "stripe"]
    
    results = scrape_multiple_companies(companies)
    
    # Save results to JSON
    with open('crunchbase_data.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Scraping completed. Data saved to crunchbase_data.json")


# Install playwright
# pip install playwright
# playwright install

from playwright.sync_api import sync_playwright
import time
import json
import random

def scrape_with_playwright(url, company_name):
    """
    Scrape website using Playwright for better JS handling and bot detection evasion.
    
    Args:
        url (str): URL to scrape
        company_name (str): Name of company for logging
        
    Returns:
        dict: Scraped data
    """
    with sync_playwright() as p:
        # Use Chromium browser
        browser = p.chromium.launch(headless=False)  # Set to True for headless mode
        
        # Create a new context with specific viewport and user agent
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        
        # Create a new page
        page = context.new_page()
        
        # Navigate to the URL
        page.goto(url, wait_until="networkidle")
        
        # Wait for content to load
        page.wait_for_selector("body", timeout=30000)
        
        # Add random human-like behavior
        page.mouse.move(random.randint(100, 500), random.randint(100, 500))
        page.wait_for_timeout(random.uniform(1000, 3000))
        
        # Extract data based on the website structure
        data = {}
        
        # Example: extract text from specific elements
        if "crunchbase.com" in url:
            # Crunchbase specific extraction
            try:
                data["name"] = page.text_content("h1.profile-name").strip()
                data["description"] = page.text_content("div.description").strip()
                # Add more selectors as needed
            except Exception as e:
                print(f"Error extracting Crunchbase data: {e}")
        
        elif "ycombinator.com" in url:
            # YCombinator specific extraction
            try:
                data["companies"] = []
                company_cards = page.query_selector_all("div.CompanyCard_root__hK85u")
                
                for card in company_cards:
                    company_info = {}
                    try:
                        company_info["name"] = card.query_selector("h3.CompanyCard_name__eAkro").text_content().strip()
                        company_info["description"] = card.query_selector("div.CompanyCard_tagline__MZrn7").text_content().strip()
                        # Add more fields as needed
                        data["companies"].append(company_info)
                    except:
                        continue
            except Exception as e:
                print(f"Error extracting YCombinator data: {e}")
        
        # Take a screenshot for debugging
        page.screenshot(path=f"{company_name}_screenshot.png")
        
        # Close browser
        browser.close()
        
        return data

if __name__ == "__main__":
    # Example usage
    yc_data = scrape_with_playwright("https://www.ycombinator.com/companies?batch=W24", "ycombinator")
    
    with open('yc_playwright_data.json', 'w') as f:
        json.dump(yc_data, f, indent=2)
    
    # Add a significant delay before the next request
    time.sleep(random.uniform(60, 120))
    
    crunchbase_data = scrape_with_playwright("https://www.crunchbase.com/organization/openai", "openai")
    
    with open('crunchbase_playwright_data.json', 'w') as f:
        json.dump(crunchbase_data, f, indent=2)
    
    print("Scraping completed!")
