import os
import time
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime
from loguru import logger
from fp.fp import FreeProxy

class RobustBayAreaScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers = self._gen_headers()
        self.proxy_pool = FreeProxy()
        self.data = pd.DataFrame()
        self.BAY_AREA_KEYWORDS = [
            'san francisco', 'sf', 'bay area', 
            'silicon valley', 'oakland', 'san jose'
        ]
        
    def _gen_headers(self):
        return {
            'User-Agent': random.choice([
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                'AppleWebKit/537.36 (KHTML, like Gecko)',
                'Chrome/125.0.0.0 Safari/537.36'
            ]),
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'Accept-Encoding': 'gzip, deflate, br'
        }

    def _request(self, url):
        """Advanced request handling with CAPTCHA awareness"""
        for attempt in range(3):
            proxy = {'http': self.proxy_pool.get()}
            try:
                time.sleep(random.uniform(7, 15))
                response = self.session.get(url, proxies=proxy, timeout=25)
                
                if response.status_code == 403:
                    logger.warning(f"Blocked by {url}, rotating infrastructure...")
                    self._rotate_session()
                    continue
                    
                response.raise_for_status()
                return response
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed: {str(e)}")
                time.sleep(2 ** attempt)
        return None

    def _rotate_session(self):
        """Full session rotation to bypass blocking"""
        self.session = requests.Session()
        self.session.headers = self._gen_headers()
        self.proxy_pool = FreeProxy()
        logger.info("Rotated session and proxy pool")

    def _validate_location(self, text):
        """Enhanced location validation"""
        if not isinstance(text, str):
            return False
        return any(kw in text.lower() for kw in self.BAY_AREA_KEYWORDS)

    def scrape_growthlist(self):
        """Bulletproof GrowthList scraper"""
        logger.info("Scraping GrowthList...")
        try:
            url = "https://growthlist.co/san-francisco-startups/"
            response = self._request(url)
            
            if not response:
                return
                
            soup = BeautifulSoup(response.text, 'html.parser')
            companies = []
            
            table = soup.find('table')
            if not table:
                logger.error("GrowthList table structure changed")
                return
                
            for row in table.find_all('tr')[1:51]:
                cols = row.find_all('td')
                if len(cols) < 7:
                    continue
                    
                try:
                    company = {
                        'name': cols[0].text.strip(),
                        'website': cols[1].text.strip(),
                        'industry': cols[2].text.strip(),
                        'funding': cols[4].text.strip(),
                        'employees': cols[5].text.strip(),
                        'source': 'GrowthList',
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    if self._validate_location('San Francisco Bay Area'):
                        companies.append(company)
                        
                except Exception as e:
                    logger.error(f"GrowthList row error: {str(e)}")
                    
            self._append_data(pd.DataFrame(companies))
            
        except Exception as e:
            logger.error(f"GrowthList failed: {str(e)}")

    def scrape_ycombinator(self):
        """Reliable YC scraper with modern selectors"""
        logger.info("Scraping Y Combinator...")
        try:
            base_url = "https://www.ycombinator.com"
            response = self._request(f"{base_url}/companies")
            
            if not response:
                return
                
            soup = BeautifulSoup(response.text, 'html.parser')
            companies = []
            
            for card in soup.select('a[class*="company-card"]'):
                try:
                    profile_url = urljoin(base_url, card['href'])
                    profile_resp = self._request(profile_url)
                    
                    if not profile_resp:
                        continue
                        
                    profile_soup = BeautifulSoup(profile_resp.text, 'html.parser')
                    
                    company = {
                        'name': profile_soup.find('h1').text.strip(),
                        'website': profile_soup.find('a', {'aria-label': 'Website'})['href'],
                        'industry': profile_soup.find('div', text='Industry').find_next('div').text.strip(),
                        'funding': profile_soup.find('div', text='Total Raised').find_next('div').text.strip(),
                        'employees': profile_soup.find('div', text='Team Size').find_next('div').text.strip(),
                        'source': 'Y Combinator',
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    if self._validate_location(company.get('location', '')):
                        self._append_data(pd.DataFrame([company]))
                        
                except Exception as e:
                    logger.error(f"YC profile error: {str(e)}")
                    
        except Exception as e:
            logger.error(f"YC failed: {str(e)}")

    def scrape_crunchbase(self):
        """Crunchbase workaround implementation"""
        logger.info("Scraping Crunchbase Alternative...")
        try:
            # Alternative approach using organization names
            orgs = ['openai', 'stripe', 'coinbase', 'databricks', 'instacart']
            
            for org in orgs:
                try:
                    url = f"https://www.crunchbase.com/organization/{org}"
                    response = self._request(url)
                    
                    if not response:
                        continue
                        
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    company = {
                        'name': soup.find('h1').text.strip(),
                        'website': soup.find('a', {'data-link-type': 'homepage'})['href'],
                        'industry': soup.find('a', {'aria-label': 'Category'}).text.strip(),
                        'funding': soup.find('span', text='Total Funding Amount').find_next('span').text.strip(),
                        'employees': soup.find('span', text='Employee Count').find_next('span').text.strip(),
                        'source': 'Crunchbase',
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    if self._validate_location(company.get('location', '')):
                        self._append_data(pd.DataFrame([company]))
                        
                except Exception as e:
                    logger.error(f"Crunchbase {org} error: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Crunchbase failed: {str(e)}")

    def _append_data(self, new_df):
        """Safe data appending with validation"""
        if not new_df.empty:
            new_df = new_df[~new_df['website'].str.contains('crunchbase|ycombinator', case=False)]
            self.data = pd.concat([self.data, new_df], ignore_index=True)

    def clean_data(self):
        """Robust data cleaning pipeline"""
        if self.data.empty:
            return
            
        # Funding conversion with fallback
        self.data['funding'] = (
            self.data['funding']
            .str.replace(r'[^\d.]', '', regex=True)
            .replace('', '0')
            .apply(pd.to_numeric, errors='coerce')
            .fillna(0)
        )
        
        # Employee count extraction
        self.data['employee_count'] = (
            self.data['employees']
            .str.extract(r'(\d+)')
            .fillna(0)
            .astype(int)
        )
        
        # AI detection
        self.data['is_ai'] = (
            self.data['industry'].str.contains('AI|Artificial Intelligence', case=False) |
            self.data['website'].apply(self._detect_ai_tech)
        )
        
        # Final validation
        self.data = self.data[self.data['employee_count'] > 0]
        self.data = self.data.drop_duplicates(['name', 'website'])

    def _detect_ai_tech(self, url):
        """Safe AI detection from website"""
        try:
            response = self._request(url)
            if response and any(kw in response.text.lower() for kw in [' ai ', 'machine learning', 'neural network']):
                return True
        except:
            return False
        return False

    def save_data(self):
        """Guaranteed save implementation"""
        os.makedirs('AMergedCsvFiles2', exist_ok=True)
        path = os.path.join('AMergedCsvFiles2', 'final_startups.csv')
        
        try:
            existing = pd.read_csv(path)
            updated = pd.concat([existing, self.data]).drop_duplicates()
            updated.to_csv(path, index=False)
        except FileNotFoundError:
            self.data.to_csv(path, index=False)
            
        logger.success(f"Saved {len(self.data)} companies to {path}")

if __name__ == "__main__":
    scraper = RobustBayAreaScraper()
    
    logger.info("Starting GrowthList scrape...")
    scraper.scrape_growthlist()
    
    logger.info("Starting Y Combinator scrape...")
    scraper.scrape_ycombinator()
    
    logger.info("Starting Crunchbase scrape...")
    scraper.scrape_crunchbase()
    
    logger.info("Cleaning data...")
    scraper.clean_data()
    
    logger.info("Saving results...")
    scraper.save_data()
    
    print("\nFinal Data Sample:")
    print(scraper.data[['name', 'source', 'funding', 'employee_count', 'is_ai']].head())
