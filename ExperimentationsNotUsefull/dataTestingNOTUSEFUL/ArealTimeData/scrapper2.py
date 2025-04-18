import pandas as pd
import requests
import random
import time
from bs4 import BeautifulSoup
from loguru import logger
from fp.fp import FreeProxy

class ReliableScraper:
    def __init__(self):
        self.data = pd.DataFrame(columns=[
            'name', 'website', 'industry', 
            'funding', 'employees', 'source'
        ])
        self.proxy_pool = FreeProxy()
        self.session = requests.Session()
        self._rotate_headers()
        
    def _rotate_headers(self):
        self.session.headers = {
            'User-Agent': random.choice([
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
            ]),
            'Accept-Language': 'en-US,en;q=0.5'
        }
    
    def _get_proxy(self):
        return {'http': self.proxy_pool.get()}
    
    def _scrape_growthlist(self):
        """Robust GrowthList implementation"""
        logger.info("Scraping GrowthList...")
        url = "https://growthlist.co/san-francisco-startups/"
        
        try:
            response = self.session.get(url, proxies=self._get_proxy(), timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Current structure analysis-based selectors
            cards = soup.select('div.company-card')
            if not cards:
                logger.error("GrowthList structure changed - no cards found")
                return
                
            for card in cards:
                try:
                    self.data = self.data.append({
                        'name': card.select_one('h3').text.strip(),
                        'website': card.select_one('a.company-link')['href'],
                        'industry': card.select_one('.industry-tag').text.strip(),
                        'funding': card.select_one('.funding').text.replace('$', ''),
                        'employees': card.select_one('.team-size').text.split()[0],
                        'source': 'GrowthList'
                    }, ignore_index=True)
                except Exception as e:
                    logger.error(f"GrowthList card error: {str(e)}")
                    
        except Exception as e:
            logger.error(f"GrowthList failed: {str(e)}")

    def _scrape_ycombinator(self):
        """Conservative YC scraping"""
        logger.info("Scraping Y Combinator...")
        try:
            response = self.session.get(
                "https://www.ycombinator.com/companies?location=San+Francisco+Bay+Area",
                proxies=self._get_proxy(),
                timeout=20
            )
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for company in soup.select('div.company-listing'):
                try:
                    self.data = self.data.append({
                        'name': company.select_one('h3').text.strip(),
                        'website': company.select_one('a')['href'],
                        'industry': company.select_one('.category').text.strip(),
                        'source': 'Y Combinator'
                    }, ignore_index=True)
                except Exception as e:
                    logger.error(f"YC company error: {str(e)}")
                    
        except Exception as e:
            logger.error(f"YC failed: {str(e)}")

    def _validate_data(self):
        """Ensure data integrity"""
        required_cols = ['name', 'source']
        if not all(col in self.data.columns for col in required_cols):
            logger.error("Missing critical columns in dataset")
            return False
            
        self.data = self.data.dropna(subset=required_cols)
        return True

    def save_data(self):
        if self._validate_data():
            self.data.to_csv('AMergedCsvFiles2/verified_startups.csv', index=False)
            logger.success(f"Saved {len(self.data)} validated entries")
        else:
            logger.error("No valid data to save")

if __name__ == "__main__":
    scraper = ReliableScraper()
    scraper._scrape_growthlist()
    scraper._scrape_ycombinator()
    scraper.save_data()




class BayAreaScraper:
    def scrape_source(self, source_name: str):
        """Unified scraping interface"""
        try:
            if source_name == "GrowthList": self._scrape_growthlist()
            elif source_name == "YC": self._scrape_ycombinator()
            elif source_name == "Crunchbase": self._scrape_crunchbase()
        except Exception as e:
            logger.error(f"{source_name} failed: {str(e)}")

        GROWTHLIST_SELECTORS = {
        'table': 'div.startup-table',  # Updated based on current structure
        'rows': 'div.table-row',
        'name': 'h3.company-name',
        'funding': 'span.funding-amount'
        }



def _make_request(self, url):
    for _ in range(3):
        proxy = self._get_verified_proxy()
        time.sleep(random.uniform(5, 15))
        response = self.session.get(url, proxies=proxy, timeout=30)
        if response.status_code == 200: 
            return response
        self._rotate_infrastructure()
    return None
