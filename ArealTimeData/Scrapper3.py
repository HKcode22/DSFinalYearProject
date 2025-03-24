import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from loguru import logger

class GrowthListScraper:
    def __init__(self):
        self.driver = self._init_driver()
        self.data = []
        
    def _init_driver(self):
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        options.add_argument("--disable-blink-features=AutomationControlled")
        return webdriver.Chrome(service=webdriver.ChromeService(ChromeDriverManager().install()), options=options)

    def scrape(self):
        logger.info("Scraping GrowthList...")
        try:
            self.driver.get("https://growthlist.co/san-francisco-startups/")
            
            # Wait for main content container
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.prose"))
            )
            
            # Find all markdown table rows
            rows = self.driver.find_elements(By.CSS_SELECTOR, "div.prose table tr")
            if not rows:
                logger.error("No table found - structure changed")
                return
                
            for row in rows[1:101]:  # Skip header + first 100 startups
                cols = row.find_elements(By.TAG_NAME, "td")
                if len(cols) >= 7:
                    self.data.append({
                        'name': cols[0].text.strip(),
                        'website': cols[1].text.strip(),
                        'industry': cols[2].text.strip(),
                        'country': cols[3].text.strip(),
                        'funding': cols[4].text.replace('$', '').replace(',', ''),
                        'funding_type': cols[5].text.strip(),
                        'last_funding': cols[6].text.strip()
                    })
                    
            logger.success(f"Found {len(self.data)} startups")
            
        except Exception as e:
            logger.error(f"Scraping failed: {str(e)}")
            self.driver.save_screenshot("error.png")
        finally:
            self.driver.quit()

    def save_data(self):
        if self.data:
            df = pd.DataFrame(self.data)
            # Convert funding to numeric
            df['funding'] = pd.to_numeric(
                df['funding'].replace({'Pre-Seed': '0', 'Seed': '0', '-': '0'}),
                errors='coerce'
            ).fillna(0)
            
            df.to_csv("AMergedCsvFiles2/growthlist_startups.csv", index=False)
            logger.success(f"Saved {len(df)} startups to CSV")
        else:
            logger.error("No data to save")

if __name__ == "__main__":
    scraper = GrowthListScraper()
    scraper.scrape()
    scraper.save_data()
