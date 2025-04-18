import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import logging
import json
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("startup_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("bay_area_startup_scraper")

class StartupScraper:
    def __init__(self, output_dir="startup_data"):
        """Initialize the scraper with configuration settings."""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        }
        self.session = self._setup_session()
        self.output_dir = output_dir
        self.ensure_output_dir()
        self.startup_data = []
        
    def ensure_output_dir(self):
        """Create the output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")
    
    def _setup_session(self):
        """Configure a session with retry capabilities."""
        session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update(self.headers)
        return session
    
    def random_delay(self, min_seconds=1, max_seconds=5):
        """Implement a random delay to avoid overloading servers."""
        delay = random.uniform(min_seconds, max_seconds)
        logger.debug(f"Waiting for {delay:.2f} seconds")
        time.sleep(delay)
    
    def scrape_y_combinator(self, location="san-francisco-bay-area", limit=100):
        """Scrape startup data from Y Combinator."""
        logger.info(f"Scraping Y Combinator data for {location}")
        url = f"https://www.ycombinator.com/companies/location/{location}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            startups = []
            company_cards = soup.select(".CompanyCard")
            
            for card in company_cards[:limit]:
                try:
                    name_elem = card.select_one(".CompanyCard__name")
                    desc_elem = card.select_one(".CompanyCard__description")
                    batch_elem = card.select_one(".CompanyCard__batch-name")
                    
                    startup = {
                        "name": name_elem.text.strip() if name_elem else "Unknown",
                        "description": desc_elem.text.strip() if desc_elem else "No description",
                        "batch": batch_elem.text.strip() if batch_elem else "Unknown",
                        "source": "Y Combinator",
                        "location": "Bay Area"
                    }
                    
                    startups.append(startup)
                except Exception as e:
                    logger.warning(f"Error parsing company card: {e}")
            
            logger.info(f"Scraped {len(startups)} startups from Y Combinator")
            return startups
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Y Combinator data: {e}")
            return []
    
    def scrape_growth_list(self, limit=100):
        """Scrape startup data from Growth List."""
        logger.info("Scraping Growth List data")
        url = "https://growthlist.co/san-francisco-startups/"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            startups = []
            table = soup.select_one("table")
            
            if table:
                rows = table.select("tr")
                
                for row in rows[1:limit+1]:  # Skip header row
                    try:
                        cols = row.select("td")
                        if len(cols) >= 5:
                            startup = {
                                "name": cols[0].text.strip(),
                                "website": cols[1].text.strip(),
                                "category": cols[2].text.strip(),
                                "location": "Bay Area",
                                "funding_amount": cols[4].text.strip(),
                                "funding_round": cols[5].text.strip() if len(cols) > 5 else "Unknown",
                                "date": cols[6].text.strip() if len(cols) > 6 else "Unknown",
                                "source": "Growth List"
                            }
                            startups.append(startup)
                    except Exception as e:
                        logger.warning(f"Error parsing Growth List row: {e}")
            
            logger.info(f"Scraped {len(startups)} startups from Growth List")
            return startups
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Growth List data: {e}")
            return []
    
    def scrape_crunchbase(self, location="San Francisco Bay Area", limit=50):
        """
        Simulate Crunchbase scraping (note: actual Crunchbase scraping requires API access)
        This is a simulation based on the search results data
        """
        logger.info(f"Simulating Crunchbase data for {location}")
        
        # Using data from search results to simulate Crunchbase data
        simulated_data = [
            {"name": "OpenAI", "description": "AI research and product company", "funding": "$13B+", "category": "Artificial Intelligence"},
            {"name": "Databricks", "description": "Data and AI company for cloud storage", "funding": "$3.5B+", "category": "Data Analytics"},
            {"name": "Anthropic", "description": "AI safety and research company", "funding": "$8B+", "category": "Artificial Intelligence"},
            {"name": "Scale AI", "description": "Data platform for AI", "funding": "$3.5B", "category": "Artificial Intelligence"},
            {"name": "Perplexity", "description": "AI-powered search engine", "funding": "$500M+", "category": "Artificial Intelligence"},
            {"name": "xAI", "description": "AI research and development", "funding": "$12B", "category": "Artificial Intelligence"},
            {"name": "Stripe", "description": "Online payment processing", "funding": "$2B+", "category": "FinTech"},
            {"name": "Discord", "description": "Communication platform", "funding": "$1B+", "category": "Social"},
            {"name": "Waymo", "description": "Autonomous driving technology", "funding": "$5.6B", "category": "Transportation"},
            {"name": "Cradlewise", "description": "Smart bassinet with baby monitor", "funding": "$7M", "category": "Hardware"},
        ]
        
        startups = []
        for data in simulated_data[:limit]:
            startup = {
                **data,
                "location": "Bay Area",
                "source": "Crunchbase (Simulated)",
                "date_accessed": datetime.now().strftime("%Y-%m-%d")
            }
            startups.append(startup)
        
        # Add some startups from search results
        from_search_results = self._extract_from_search_results()
        startups.extend(from_search_results[:limit-len(startups)])
        
        logger.info(f"Generated {len(startups)} simulated Crunchbase entries")
        return startups
    
    def _extract_from_search_results(self):
        """Extract additional startup data from the search results."""
        # Extracting data from search result [3] - Exploding Topics
        startups = [
            {"name": "ZeroTier", "description": "Platform for secure peer-to-peer networks", "funding": "$15.9M", "category": "Networking", "location": "Los Angeles"},
            {"name": "Deepgram", "description": "Speech recognition and voice AI platform", "funding": "$85.9M", "category": "AI", "location": "Bay Area"},
            {"name": "Cradlewise", "description": "Smart bassinets with baby monitors", "funding": "$7M", "category": "Hardware", "location": "Bay Area"},
            {"name": "PhotoRoom", "description": "AI photo and video editing app", "funding": "$64M", "category": "Media", "location": "Paris"},
            {"name": "Preply", "description": "Language tutoring marketplace", "funding": "$171M", "category": "Education", "location": "Massachusetts"},
            {"name": "Airwallex", "description": "Global payment solutions", "funding": "$1.1B", "category": "FinTech", "location": "Australia"},
            {"name": "Cohere", "description": "Enterprise large language models", "funding": "$970M", "category": "AI", "location": "Toronto"},
            {"name": "Shiprocket", "description": "E-commerce logistics platform", "funding": "$323M", "category": "Logistics", "location": "India"},
            {"name": "Airbyte", "description": "Open-source data integration platform", "funding": "$181M", "category": "Data", "location": "Bay Area"},
            {"name": "Codeium", "description": "AI-powered coding assistant", "funding": "$150M", "category": "Developer Tools", "location": "Bay Area"},
        ]
        
        # Filter to Bay Area only
        bay_area_startups = [s for s in startups if s["location"] == "Bay Area"]
        
        for startup in bay_area_startups:
            startup["source"] = "Exploding Topics via Search Results"
            startup["date_accessed"] = datetime.now().strftime("%Y-%m-%d")
        
        return bay_area_startups
    
    def collect_data(self, min_startups=50):
        """Collect data from all sources until we have at least the minimum number of startups."""
        all_startups = []
        
        # Try Y Combinator first
        yc_startups = self.scrape_y_combinator(limit=min_startups)
        all_startups.extend(yc_startups)
        self.random_delay(2, 5)
        
        # If we don't have enough, try Growth List
        if len(all_startups) < min_startups:
            growth_list_startups = self.scrape_growth_list(limit=min_startups - len(all_startups))
            all_startups.extend(growth_list_startups)
            self.random_delay(2, 5)
        
        # If we still don't have enough, use simulated Crunchbase data
        if len(all_startups) < min_startups:
            crunchbase_startups = self.scrape_crunchbase(limit=min_startups - len(all_startups))
            all_startups.extend(crunchbase_startups)
        
        # Remove duplicates based on company name
        seen_names = set()
        unique_startups = []
        
        for startup in all_startups:
            if startup["name"] not in seen_names:
                seen_names.add(startup["name"])
                unique_startups.append(startup)
        
        self.startup_data = unique_startups
        logger.info(f"Collected data on {len(self.startup_data)} unique startups")
        
        return self.startup_data
    
    def save_to_csv(self, filename="bay_area_startups.csv"):
        """Save the collected data to a CSV file."""
        if not self.startup_data:
            logger.warning("No data to save. Run collect_data() first.")
            return False
        
        filepath = os.path.join(self.output_dir, filename)
        try:
            df = pd.DataFrame(self.startup_data)
            df.to_csv(filepath, index=False)
            logger.info(f"Data saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
            return False
    
    def save_to_json(self, filename="bay_area_startups.json"):
        """Save the collected data to a JSON file."""
        if not self.startup_data:
            logger.warning("No data to save. Run collect_data() first.")
            return False
        
        filepath = os.path.join(self.output_dir, filename)
        try:
            with open(filepath, 'w') as f:
                json.dump(self.startup_data, f, indent=4)
            logger.info(f"Data saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")
            return False
    
    def analyze_data(self):
        """Perform basic analysis on the collected data."""
        if not self.startup_data:
            logger.warning("No data to analyze. Run collect_data() first.")
            return None
        
        analysis = {
            "total_startups": len(self.startup_data),
            "by_source": {},
            "by_category": {}
        }
        
        # Count by source
        sources = [s.get("source", "Unknown") for s in self.startup_data]
        for source in set(sources):
            analysis["by_source"][source] = sources.count(source)
        
        # Count by category if available
        if any("category" in s for s in self.startup_data):
            categories = [s.get("category", "Unknown") for s in self.startup_data]
            for category in set(categories):
                analysis["by_category"][category] = categories.count(category)
        
        logger.info("Data analysis complete")
        return analysis

# Example usage
def main():
    """Main function to demonstrate the scraper usage."""
    logger.info("Starting Bay Area startup data collection")
    
    scraper = StartupScraper()
    
    # Collect data from all sources
    startups = scraper.collect_data(min_startups=50)
    
    # Save the data
    scraper.save_to_csv()
    scraper.save_to_json()
    
    # Analyze the data
    analysis = scraper.analyze_data()
    if analysis:
        print(f"\nCollected {analysis['total_startups']} startups")
        print("\nBreakdown by source:")
        for source, count in analysis['by_source'].items():
            print(f"- {source}: {count}")
        
        if analysis.get('by_category'):
            print("\nBreakdown by category:")
            for category, count in analysis['by_category'].items():
                print(f"- {category}: {count}")
    
    logger.info("Data collection complete")

if __name__ == "__main__":
    main()
