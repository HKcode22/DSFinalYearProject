


# import requests
# import pandas as pd
# import time
# import os
# import json
# from bs4 import BeautifulSoup
# from datetime import datetime
# import re
# import csv
# import logging
# from tqdm import tqdm
# import numpy as np
# from dotenv import load_dotenv
# from urllib.parse import urljoin
# import random
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager
# from newsapi import NewsApiClient
# import backoff

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("startup_data_collection.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Load environment variables for API keys
# load_dotenv()

# # API Keys (required for certain data sources)

# class BayAreaStartupDataCollector:
#     def __init__(self, output_file="bay_area_startups_dataset.csv"):
#         """Initialize the data collector with output file path."""
#         self.output_file = output_file
#         self.startup_data = []
#         self.headers = {
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#         }
#         # Initialize API clients
#         self.news_api = NewsApiClient(api_key=NEWS_API_KEY) if NEWS_API_KEY else None
        
#         # Initialize web driver for dynamic content
#         self.driver = None
        
#         # Bay Area cities and terms for filtering
#         self.bay_area_locations = [
#             'san francisco', 'oakland', 'berkeley', 'palo alto', 'menlo park', 
#             'mountain view', 'sunnyvale', 'santa clara', 'san jose', 'redwood city',
#             'south san francisco', 'fremont', 'san mateo', 'cupertino', 'emeryville',
#             'hayward', 'milpitas', 'burlingame', 'foster city', 'san carlos',
#             'walnut creek', 'pleasanton', 'san ramon', 'bay area', 'silicon valley'
#         ]
        
#     def _initialize_webdriver(self):
#         """Initialize Chrome webdriver for scraping dynamic content."""
#         if self.driver is None:
#             try:
#                 chrome_options = Options()
#                 chrome_options.add_argument("--headless")
#                 chrome_options.add_argument("--no-sandbox")
#                 chrome_options.add_argument("--disable-dev-shm-usage")
#                 chrome_options.add_argument("--disable-gpu")
#                 chrome_options.add_argument(f"user-agent={self.headers['User-Agent']}")
                
#                 service = Service(ChromeDriverManager().install())
#                 self.driver = webdriver.Chrome(service=service, options=chrome_options)
#                 logger.info("Initialized Chrome WebDriver")
#             except Exception as e:
#                 logger.error(f"Failed to initialize WebDriver: {e}")
#                 return False
#         return True
                
#     @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=5)
#     def _make_request(self, url, params=None, headers=None):
#         """Make HTTP request with exponential backoff for failed requests."""
#         if not headers:
#             headers = self.headers
            
#         response = requests.get(url, params=params, headers=headers, timeout=30)
#         response.raise_for_status()
#         return response
    
#     def collect_crunchbase_data(self):
#         """
#         Collect Bay Area startup data from Crunchbase API.
#         Uses proper API if key is available, otherwise logs the limitation.
#         """
#         logger.info("Collecting Crunchbase data...")
        
#         if not CRUNCHBASE_API_KEY:
#             logger.warning("No Crunchbase API key provided. Skipping Crunchbase data collection.")
#             return
            
#         # Use the official Crunchbase API with proper endpoint
#         base_url = "https://api.crunchbase.com/api/v4/searches/organizations"
        
#         # We'll collect multiple pages of results
#         has_more_pages = True
#         after_id = None
#         total_collected = 0
        
#         while has_more_pages:
#             params = {
#                 "user_key": CRUNCHBASE_API_KEY,
#                 "query": "location:(\"San Francisco Bay Area\" OR \"Silicon Valley\")",
#                 "field_ids": ["name", "short_description", "website_url", "linkedin_url", 
#                              "twitter_url", "founded_on", "funding_total", "funding_stage",
#                              "employee_count", "categories", "location_identifiers"],
#                 "limit": 100  # Maximum allowed by API
#             }
            
#             if after_id:
#                 params["after_id"] = after_id
                
#             try:
#                 response = self._make_request(base_url, params=params)
#                 data = response.json()
                
#                 for entity in data.get('entities', []):
#                     properties = entity.get('properties', {})
                    
#                     # Check if location is in Bay Area (double check)
#                     location = properties.get('location_identifiers', {}).get('value', '')
                    
#                     # Extract categories as list
#                     categories = []
#                     for category in properties.get('categories', []):
#                         if isinstance(category, dict) and 'value' in category:
#                             categories.append(category['value'])
                    
#                     # Format the data
#                     startup = {
#                         'name': properties.get('name', ''),
#                         'description': properties.get('short_description', {}).get('value', ''),
#                         'founded_date': properties.get('founded_on', {}).get('value', ''),
#                         'website': properties.get('website_url', {}).get('value', ''),
#                         'linkedin': properties.get('linkedin_url', {}).get('value', ''),
#                         'twitter': properties.get('twitter_url', {}).get('value', ''),
#                         'industry': ','.join(categories),
#                         'location': location,
#                         'funding_total': properties.get('funding_total', {}).get('value_usd', 0),
#                         'funding_stage': properties.get('funding_stage', {}).get('value', ''),
#                         'employee_count': properties.get('employee_count', {}).get('value', 0),
#                         'data_source': 'Crunchbase API'
#                     }
                    
#                     # Only add startups with Bay Area location
#                     if any(loc in location.lower() for loc in self.bay_area_locations):
#                         self.startup_data.append(startup)
#                         total_collected += 1
                
#                 # Check for more pages
#                 after_id = data.get('entities')[-1].get('uuid') if data.get('entities') else None
#                 has_more_pages = after_id is not None and data.get('count', 0) > 0
                
#                 logger.info(f"Retrieved batch of {len(data.get('entities', []))} records from Crunchbase API")
                
#                 # Sleep to respect rate limits
#                 time.sleep(1)
                
#             except Exception as e:
#                 logger.error(f"Error fetching Crunchbase data via API: {e}")
#                 has_more_pages = False
        
#         logger.info(f"Collected {total_collected} Bay Area startups from Crunchbase")
    
#     def collect_ycombinator_data(self):
#         """
#         Collect YCombinator companies that are in the Bay Area.
#         Uses web scraping with proper parsing of YC's company list.
#         """
#         logger.info("Collecting Y Combinator data...")
        
#         # Initialize WebDriver for YC's JavaScript-rendered content
#         if not self._initialize_webdriver():
#             logger.error("Skipping Y Combinator data collection due to WebDriver initialization failure")
#             return
            
#         # YC's companies directory page
#         yc_url = "https://www.ycombinator.com/companies"
        
#         try:
#             # Load the page with Selenium
#             self.driver.get(yc_url)
            
#             # Wait for dynamic content to load
#             time.sleep(5)
            
#             # Get the page source after JavaScript execution
#             page_source = self.driver.page_source
#             soup = BeautifulSoup(page_source, 'html.parser')
            
#             # Find company cards/elements (adjust selectors based on actual YC website structure)
#             # These selectors need to match YC's actual DOM structure
#             company_elements = soup.select('div.CompanyCard')  # This is an example selector
            
#             if not company_elements:
#                 # Try alternative selectors if the first one didn't work
#                 company_elements = soup.select('div.company-card')
                
#             if not company_elements:
#                 # Last attempt with broader selector
#                 company_elements = soup.find_all('div', class_=lambda c: c and ('company' in c.lower()))
            
#             logger.info(f"Found {len(company_elements)} company elements on YC page")
            
#             yc_companies = []
#             for company in tqdm(company_elements, desc="Processing YC companies"):
#                 # Extract company data (adjust selectors based on YC's HTML structure)
#                 # These are examples and need to be updated based on actual structure
#                 name_elem = company.select_one('h4') or company.select_one('.company-name')
#                 desc_elem = company.select_one('p') or company.select_one('.company-description')
#                 location_elem = company.select_one('.location') or company.select_one('.company-location')
                
#                 company_name = name_elem.text.strip() if name_elem else ""
#                 company_desc = desc_elem.text.strip() if desc_elem else ""
#                 company_location = location_elem.text.strip() if location_elem else ""
                
#                 # Only include Bay Area companies
#                 if any(loc in company_location.lower() for loc in self.bay_area_locations):
#                     # Extract additional data if available
#                     website = ""
#                     website_elem = company.select_one('a[href^="http"]')
#                     if website_elem and 'href' in website_elem.attrs:
#                         website = website_elem['href']
                    
#                     # Get batch information
#                     batch_elem = company.select_one('.batch') or company.select_one('span')
#                     batch = batch_elem.text.strip() if batch_elem else ""
                    
#                     # Create company data entry
#                     yc_companies.append({
#                         'name': company_name,
#                         'description': company_desc,
#                         'location': company_location,
#                         'website': website,
#                         'ycombinator_batch': batch,
#                         'data_source': 'Y Combinator'
#                     })
            
#             self.startup_data.extend(yc_companies)
#             logger.info(f"Added {len(yc_companies)} Y Combinator companies from Bay Area")
            
#         except Exception as e:
#             logger.error(f"Error scraping Y Combinator data: {e}")
    
#     def collect_angel_data(self):
#         """
#         Collect startup data from AngelList/Wellfound using web scraping.
#         """
#         logger.info("Collecting AngelList/Wellfound data...")
        
#         # Initialize WebDriver if needed
#         if not self._initialize_webdriver():
#             logger.error("Skipping AngelList/Wellfound data collection due to WebDriver initialization failure")
#             return
        
#         # Base URL for San Francisco startups on Wellfound
#         base_url = "https://wellfound.com/startups/location/san-francisco-bay-area"
        
#         try:
#             # Load the page with Selenium
#             self.driver.get(base_url)
            
#             # Wait for dynamic content to load
#             time.sleep(5)
            
#             # Scroll down a few times to load more startups
#             for _ in range(5):
#                 self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#                 time.sleep(2)
            
#             # Get the page source after JavaScript execution
#             page_source = self.driver.page_source
#             soup = BeautifulSoup(page_source, 'html.parser')
            
#             # Find startup cards (adjust selector based on actual Wellfound structure)
#             startup_cards = soup.select('div.startup-card') or soup.select('div.styles_container__RgOYl')
            
#             logger.info(f"Found {len(startup_cards)} startup cards on Wellfound page")
            
#             angel_data = []
#             for card in tqdm(startup_cards, desc="Processing Wellfound startups"):
#                 try:
#                     # Extract data (adjust selectors as needed)
#                     name_elem = card.select_one('h4') or card.select_one('.title')
#                     name = name_elem.text.strip() if name_elem else ""
                    
#                     desc_elem = card.select_one('p') or card.select_one('.description')
#                     description = desc_elem.text.strip() if desc_elem else ""
                    
#                     # Extract funding information if available
#                     funding_elem = card.select_one('.funding') or card.select_one('.styles_funding')
#                     funding_info = funding_elem.text.strip() if funding_elem else ""
                    
#                     # Get company URL for more details
#                     link_elem = card.select_one('a')
#                     detail_url = ""
#                     if link_elem and 'href' in link_elem.attrs:
#                         detail_url = urljoin("https://wellfound.com", link_elem['href'])
                    
#                     angel_data.append({
#                         'name': name,
#                         'description': description, 
#                         'funding_info': funding_info,
#                         'wellfound_url': detail_url,
#                         'data_source': 'AngelList/Wellfound'
#                     })
                    
#                     # If we have a detail URL, we can extract more information
#                     if detail_url:
#                         try:
#                             # Visit company detail page
#                             self.driver.get(detail_url)
#                             time.sleep(2)
#                             detail_soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                            
#                             # Extract company website
#                             website_elem = detail_soup.select_one('a[href^="http"]:not([href*="wellfound.com"])')
#                             website = website_elem['href'] if website_elem and 'href' in website_elem.attrs else ""
                            
#                             # Extract location to confirm Bay Area
#                             location_elem = detail_soup.select_one('.location') or detail_soup.select_one('.styles_location')
#                             location = location_elem.text.strip() if location_elem else ""
                            
#                             # Extract founders
#                             founders_elem = detail_soup.select('.founder-card') or detail_soup.select('.team-member')
#                             founders = [f.text.strip() for f in founders_elem] if founders_elem else []
                            
#                             # Update with additional information
#                             angel_data[-1].update({
#                                 'website': website,
#                                 'location': location,
#                                 'founders': ','.join(founders)
#                             })
#                         except Exception as e:
#                             logger.warning(f"Error fetching detail page for {name}: {e}")
                    
#                 except Exception as e:
#                     logger.warning(f"Error processing Wellfound startup card: {e}")
            
#             # Add to main dataset
#             self.startup_data.extend(angel_data)
#             logger.info(f"Added {len(angel_data)} startups from AngelList/Wellfound")
            
#         except Exception as e:
#             logger.error(f"Error scraping AngelList/Wellfound data: {e}")
    
#     def collect_techcrunch_data(self):
#         """
#         Collect startup data from TechCrunch articles and Crunchbase information embedded in TC.
#         """
#         logger.info("Collecting TechCrunch data...")
        
#         # Search for Bay Area startup news on TechCrunch
#         url = "https://techcrunch.com/search/San+Francisco+startup"
        
#         try:
#             response = self._make_request(url)
#             soup = BeautifulSoup(response.text, 'html.parser')
            
#             # Find article cards
#             article_cards = soup.select('div.post-block')
            
#             logger.info(f"Found {len(article_cards)} TechCrunch articles about Bay Area startups")
            
#             techcrunch_data = []
#             for card in tqdm(article_cards[:30], desc="Processing TechCrunch articles"):  # Limit to 30 to avoid too many requests
#                 try:
#                     # Extract article headline and link
#                     headline_elem = card.select_one('h2 a')
#                     if not headline_elem:
#                         continue
                        
#                     headline = headline_elem.text.strip()
#                     article_url = headline_elem['href'] if 'href' in headline_elem.attrs else ""
                    
#                     if not article_url:
#                         continue
                    
#                     # Visit the article page to extract startup information
#                     article_response = self._make_request(article_url)
#                     article_soup = BeautifulSoup(article_response.text, 'html.parser')
                    
#                     # Extract company tags
#                     company_tags = article_soup.select('.article__tags a')
#                     company_names = []
                    
#                     for tag in company_tags:
#                         tag_text = tag.text.strip()
#                         # Only include if it looks like a company name (not a category)
#                         if not any(cat in tag_text.lower() for cat in ['category', 'tag', 'series']):
#                             company_names.append(tag_text)
                    
#                     # Extract funding information from article content
#                     article_content = article_soup.select_one('.article-content')
#                     article_text = article_content.text if article_content else ""
                    
#                     # Look for funding patterns like "$X million" or "raised $Y"
#                     funding_matches = re.findall(r'\$(\d+(?:\.\d+)?)\s*(million|billion|m|b)', article_text, re.IGNORECASE)
#                     funding_amount = None
#                     if funding_matches:
#                         amount, unit = funding_matches[0]
#                         amount = float(amount)
#                         if unit.lower() in ['billion', 'b']:
#                             amount *= 1000
#                         funding_amount = amount
                    
#                     # Add data for each company mentioned
#                     for company in company_names:
#                         techcrunch_data.append({
#                             'name': company,
#                             'article_headline': headline,
#                             'article_url': article_url,
#                             'funding_amount_millions': funding_amount,
#                             'article_date': card.select_one('time').get('datetime') if card.select_one('time') else "",
#                             'data_source': 'TechCrunch'
#                         })
                    
#                     # Sleep to be respectful
#                     time.sleep(2)
                    
#                 except Exception as e:
#                     logger.warning(f"Error processing TechCrunch article: {e}")
            
#             # Add to main dataset
#             self.startup_data.extend(techcrunch_data)
#             logger.info(f"Added {len(techcrunch_data)} startup mentions from TechCrunch")
            
#         except Exception as e:
#             logger.error(f"Error collecting TechCrunch data: {e}")
    
#     def collect_news_data(self):
#         """
#         Collect news mentions for Bay Area startups using NewsAPI.
#         """
#         logger.info("Collecting news data for startups...")
        
#         if not self.news_api:
#             logger.warning("NewsAPI key not available. Skipping news data collection.")
#             return
            
#         # Get unique company names from our dataset
#         company_names = set()
#         for startup in self.startup_data:
#             if 'name' in startup and startup['name']:
#                 company_names.add(startup['name'])
        
#         logger.info(f"Collecting news for {len(company_names)} unique companies")
        
#         # For each startup, collect recent news
#         for company_name in tqdm(company_names, desc="Fetching news data"):
#             try:
#                 # Query NewsAPI for articles about this startup
#                 news_results = self.news_api.get_everything(
#                     q=f'"{company_name}" AND ("Bay Area" OR "San Francisco" OR "Silicon Valley")',
#                     language='en',
#                     sort_by='relevancy',
#                     page_size=10  # Limit to 10 articles per company
#                 )
                
#                 # Extract relevant news data
#                 articles = news_results.get('articles', [])
                
#                 # Find the corresponding startup entries and update them
#                 for startup in self.startup_data:
#                     if startup.get('name') == company_name:
#                         # Add news count and latest news date
#                         startup['news_mention_count'] = len(articles)
                        
#                         if articles:
#                             # Get the most recent article date
#                             latest_date = max([article['publishedAt'] for article in articles])
#                             startup['latest_news_date'] = latest_date
                            
#                             # Store the most recent article title and URL
#                             latest_article = sorted(articles, key=lambda x: x['publishedAt'], reverse=True)[0]
#                             startup['latest_news_title'] = latest_article['title']
#                             startup['latest_news_url'] = latest_article['url']
                            
#                             # Extract simple sentiment indicators (basic approach)
#                             sentiment_keywords = {
#                                 'positive': ['success', 'growth', 'funding', 'launch', 'innovation', 'partnership'],
#                                 'negative': ['layoff', 'shutdown', 'failure', 'struggle', 'lawsuit', 'downturn']
#                             }
                            
#                             # Basic sentiment analysis on article titles
#                             positive_count = 0
#                             negative_count = 0
                            
#                             for article in articles:
#                                 title = article['title'].lower()
#                                 for word in sentiment_keywords['positive']:
#                                     if word in title:
#                                         positive_count += 1
#                                 for word in sentiment_keywords['negative']:
#                                     if word in title:
#                                         negative_count += 1
                            
#                             startup['news_sentiment'] = 'positive' if positive_count > negative_count else 'negative' if negative_count > positive_count else 'neutral'
                
#                 # Rate limit - avoid hitting API limits
#                 time.sleep(0.2)
                
#             except Exception as e:
#                 logger.warning(f"Error fetching news for {company_name}: {e}")
    
#     def collect_github_data(self):
#         """
#         Collect GitHub data for technical startups.
#         """
#         logger.info("Collecting GitHub data for startups...")
        
#         github_headers = self.headers.copy()
#         if GITHUB_API_KEY:
#             github_headers['Authorization'] = f'token {GITHUB_API_KEY}'
        
#         # Get unique company names from our dataset
#         processed_companies = set()
        
#         for startup in tqdm(self.startup_data, desc="Fetching GitHub data"):
#             try:
#                 company_name = startup.get('name', '').lower().replace(' ', '')
                
#                 # Skip if already processed
#                 if company_name in processed_companies or not company_name:
#                     continue
                    
#                 processed_companies.add(company_name)
                
#                 # Get domain from website if available
#                 domain = ''
#                 if 'website' in startup and startup['website']:
#                     domain = startup['website'].lower()
#                     # Extract domain name without protocol and www
#                     domain = re.sub(r'^https?://(www\.)?', '', domain)
#                     # Remove .com, .io, etc.
#                     domain = domain.split('/')[0].split('.')[0]
                
#                 # Try multiple potential GitHub organization names
#                 potential_orgs = [company_name]
#                 if domain and domain != company_name:
#                     potential_orgs.append(domain)
                
#                 github_found = False
                
#                 for org_name in potential_orgs:
#                     if github_found:
#                         break
                        
#                     github_url = f"https://api.github.com/orgs/{org_name}"
                    
#                     try:
#                         response = requests.get(github_url, headers=github_headers)
#                         if response.status_code == 200:
#                             github_found = True
#                             org_data = response.json()
                            
#                             # Get repository data
#                             repos_url = f"https://api.github.com/orgs/{org_name}/repos"
#                             repos_response = requests.get(repos_url, headers=github_headers)
#                             repos_data = repos_response.json()
                            
#                             # GitHub metrics to collect
#                             github_metrics = {
#                                 'github_organization': org_name,
#                                 'github_public_repos': len(repos_data) if isinstance(repos_data, list) else 0,
#                                 'github_followers': org_data.get('followers', 0),
#                                 'github_url': f"https://github.com/{org_name}",
#                                 'github_created_at': org_data.get('created_at', '')
#                             }
                            
#                             # Calculate average stars and forks
#                             if isinstance(repos_data, list) and repos_data:
#                                 github_metrics['github_avg_stars'] = sum(repo.get('stargazers_count', 0) for repo in repos_data) / len(repos_data)
#                                 github_metrics['github_avg_forks'] = sum(repo.get('forks_count', 0) for repo in repos_data) / len(repos_data)
#                                 github_metrics['github_total_stars'] = sum(repo.get('stargazers_count', 0) for repo in repos_data)
                                
#                                 # Get top repository
#                                 top_repo = max(repos_data, key=lambda x: x.get('stargazers_count', 0))
#                                 github_metrics['github_top_repo'] = top_repo.get('name', '')
#                                 github_metrics['github_top_repo_stars'] = top_repo.get('stargazers_count', 0)
#                             else:
#                                 github_metrics['github_avg_stars'] = 0
#                                 github_metrics['github_avg_forks'] = 0
#                                 github_metrics['github_total_stars'] = 0
#                                 github_metrics['github_top_repo'] = ''
#                                 github_metrics['github_top_repo_stars'] = 0
                            
#                             # Update all matching startup entries with this GitHub data
#                             for s in self.startup_data:
#                                 if s.get('name', '').lower() == startup.get('name', '').lower():
#                                     s.update(github_metrics)
                    
#                     except Exception as e:
#                         if 'rate limit' in str(e).lower():
#                             logger.warning(f"GitHub API rate limit reached. Slowing down...")
#                             time.sleep(60)  # Wait a minute before continuing
#                         else:
#                             logger.debug(f"Error checking GitHub for {org_name}: {e}")
                
#                 # Don't hit the API too fast
#                 time.sleep(1)
                
#             except Exception as e:
#                 logger.warning(f"Error processing GitHub data for {startup.get('name', 'Unknown')}: {e}")
    
#     def collect_opencorporates_data(self):
#         """
#         Collect company registration data from OpenCorporates API.
#         """
#         logger.info("Collecting OpenCorporates registration data...")
        
#         if not OPENCORPORATES_API_KEY:
#             logger.warning("No OpenCorporates API key provided. Skipping corporate registration data.")
#             return
            
#         # Get unique company names
#         company_names = set()
#         for startup in self.startup_data:
#             if 'name' in startup and startup['name']:
#                 company_names.add(startup['name'])
        
#         for company_name in tqdm(company_names, desc="Fetching corporate registration data"):
#             try:
#                 # Search OpenCorporates for this company in California
#                 url = "https://api.opencorporates.com/v0.4/companies/search"
#                 params = {
#                     "api_token": OPENCORPORATES_API_KEY,
#                     "q": company_name,
#                     "jurisdiction_code": "us_ca"  # California
#                 }
                
#                 response = self._make_request(url, params=params)
#                 data = response.json()
                
#                 companies = data.get('results', {}).get('companies', [])
                
#                 if companies:
#                     # Get the most relevant company (first result)
#                     company = companies[0]['company']
                    
#                     # Add registration data to matching startup entries
#                     for startup in self.startup_data:
#                         if startup.get('name') == company_name:
#                             startup.update({
#                                 'registration_number': company.get('company_number', ''),
#                                 'registration_status': company.get('current_status', ''),
#                                 'incorporation_date': company.get('incorporation_date', ''),
#                                 'company_type': company.get('company_type', ''),
#                                 'registry_url': company.get('registry_url', ''),
#                                 'data_source_registration': 'OpenCorporates'
#                             })
                
#                 # Respect rate limits
#                 time.sleep(0.5)
                
#             except Exception as e:
#                 logger.warning(f"Error fetching OpenCorporates data for {company_name}: {e}")
    
#     def collect_all_data(self):
#         """
#         Run all data collection methods and create a comprehensive dataset.
#         """
#         # Start with basic data collection
#         logger.info("Starting comprehensive data collection for Bay Area startups")
        
#         # Core data sources
#         self.collect_crunchbase_data()
#         self.collect_ycombinator_data()
#         self.collect_angel_data()
#         self.collect_techcrunch_data()
        
#         # Enrichment data sources (only if we have any data)
#         if self.startup_data:
#             self.collect_news_data()
#             self.collect_github_data()
#             self.collect_opencorporates_data()
        
#         # Convert to DataFrame for processing
#         df = pd.DataFrame(self.startup_data)
        
#         # Clean and deduplicate data
#         if not df.empty:
#             logger.info("Processing and cleaning collected data")
            
#             # Remove duplicates based on company name (case insensitive)
#             df['name_lower'] = df['name'].str.lower() if 'name' in df.columns else ''
            
#             # If we have duplicates, merge their data
#             if df.duplicated('name_lower').any():
#                 logger.info("Merging duplicate company entries")
                
#                 # Group by company name and merge data
#                 df_merged = []
#                 for name, group in df.groupby('name_lower'):
#                     # Get first row as base
#                     merged_row = {}
                    
#                     # For each column, get the first non-null value
#                     for col in df.columns:
#                         non_null_values = group[col].dropna()
#                         if not non_null_values.empty:
#                             merged_row[col] = non_null_values.iloc[0]
#                         else:
#                             merged_row[col] = None
                    
#                     # Special handling for certain fields
#                     merged_row['data_source'] = ', '.join(group['data_source'].unique())
                    
#                     df_merged.append(merged_row)
                
#                 df = pd.DataFrame(df_merged)
            
#             # Drop the temporary column
#             df = df.drop(columns=['name_lower'])
            
#             # Fill missing values appropriately
#             df = df.fillna({
#                 'description': '',
#                 'founded_date': '',
#                 'location': 'Bay Area',
#                 'industry': '',
#                 'funding_total': 0,
#                 'news_mention_count': 0,
#                 'github_public_repos': 0
#             })
            
#             # Add timestamp
#             df['data_collected_date'] = datetime.now().strftime('%Y-%m-%d')
            
#             # Calculate metrics for predictive analysis (if we have enough data)
#             if all(col in df.columns for col in ['news_mention_count', 'github_total_stars']):
#                 logger.info("Calculating predictive metrics")
                
#                 # Initialize score column
#                 df['growth_potential_score'] = np.NaN
                
#                 # Define columns to use for scoring
#                 score_cols = [
#                     ('news_mention_count', 0.3),
#                     ('github_total_stars', 0.25),
#                     ('funding_total', 0.3),
#                     ('github_public_repos', 0.15)
#                 ]
                
#                 # Check which columns actually exist
#                 valid_cols = [(col, weight) for col, weight in score_cols if col in df.columns]
                
#                 if valid_cols:
#                     # Normalize each column first (0-1 scale)
#                     for col, _ in valid_cols:
#                         # Skip if column is empty
#                         if df[col].isna().all() or (df[col] == 0).all():
#                             continue
                            
#                         # Create normalized column
#                         max_val = df[col].max()
#                         if max_val > 0:  # Avoid division by zero
#                             df[f'{col}_norm'] = df[col] / max_val
                    
#                     # Calculate weighted score
#                     df['growth_potential_score'] = 0
#                     total_weight = 0
                    
#                     for col, weight in valid_cols:
#                         norm_col = f'{col}_norm'
#                         if norm_col in df.columns:
#                             df['growth_potential_score'] += df[norm_col] * weight
#                             total_weight += weight
                    
#                     # Normalize final score to 0-100 scale
#                     if total_weight > 0:
#                         df['growth_potential_score'] = (df['growth_potential_score'] / total_weight) * 100
                        
#                         # Clean up normalization columns
#                         for col, _ in valid_cols:
#                             if f'{col}_norm' in df.columns:
#                                 df = df.drop(columns=[f'{col}_norm'])
            
#             # Save to CSV
#             df.to_csv(self.output_file, index=False)
#             logger.info(f"Saved {len(df)} startup records to {self.output_file}")
            
#             return df
#         else:
#             logger.warning("No data collected. Output file not created.")
#             return pd.DataFrame()
        
#     def cleanup(self):
#         """Clean up resources when done."""
#         if self.driver:
#             try:
#                 self.driver.quit()
#                 logger.info("WebDriver closed successfully")
#             except Exception as e:
#                 logger.error(f"Error closing WebDriver: {e}")


# # Main execution
# if __name__ == "__main__":
#     try:
#         # Set up collector
#         collector = BayAreaStartupDataCollector()
        
#         # Run collection process
#         startup_df = collector.collect_all_data()
        
#         # Display sample and stats if data was collected
#         if not startup_df.empty:
#             print("\nCollection complete! Dataset statistics:")
#             print(f"Total unique startups: {len(startup_df)}")
            
#             # Data source breakdown
#             if 'data_source' in startup_df.columns:
#                 print("\nData sources distribution:")
#                 for source, count in startup_df['data_source'].value_counts().items():
#                     print(f"- {source}: {count} records")
            
#             # Funding stage breakdown if available
#             if 'funding_stage' in startup_df.columns:
#                 print("\nFunding stage distribution:")
#                 for stage, count in startup_df['funding_stage'].value_counts().items():
#                     if isinstance(stage, str) and stage.strip():
#                         print(f"- {stage}: {count} startups")
            
#             # Show top companies by predictive score if available
#             if 'growth_potential_score' in startup_df.columns:
#                 print("\nTop 10 startups by growth potential score:")
#                 top_companies = startup_df.sort_values('growth_potential_score', ascending=False).head(10)
#                 for idx, row in top_companies.iterrows():
#                     print(f"- {row['name']}: {row['growth_potential_score']:.1f}/100")
        
#         # Clean up resources
#         collector.cleanup()
        
#     except Exception as e:
#         logger.error(f"Unhandled error in data collection: {e}", exc_info=True)
