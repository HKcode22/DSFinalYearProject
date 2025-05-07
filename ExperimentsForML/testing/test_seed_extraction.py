"""
Test script to verify that seed funding stages are correctly extracted 
from the topstartupio50.json file.
"""

import json
import pandas as pd
import re
import os
import sys
import logging
from pprint import pprint

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_funding_info(funding_str):
    """
    Extract funding information from a funding string.
    
    Args:
        funding_str (str): The funding string (e.g., "Y Combinator $2M Seed in 2020")
        
    Returns:
        tuple: (amount, stage, year)
    """
    if not funding_str or pd.isna(funding_str):
        return None, None, None
    
    amount = None
    stage = None
    year = None
    
    # Extract amount
    amount_match = re.search(r'\$(\d+(?:\.\d+)?[KMB]?)', funding_str)
    if amount_match:
        amount = amount_match.group(0)  # Keep the $ symbol
    
    # Extract stage with improved pattern matching for Seed
    stage_pattern = r'(Pre[-\s]?Seed|Seed|Angel|Series\s+[A-Z]|Venture[\s\-]+Series\s+Unknown|Initial\s+Coin\s+Offering|ICO|Private\s+Equity|Grant|Debt\s+Financing|Undisclosed|Post[-\s]?IPO)'
    stage_match = re.search(stage_pattern, funding_str, re.IGNORECASE)
    if stage_match:
        stage = stage_match.group(1)
    
    # Extract year
    year_match = re.search(r'in\s+(\d{4})', funding_str)
    if year_match:
        year = year_match.group(1)
    
    return amount, stage, year

def test_topstartup_seed_extraction(json_path='test_data/JSONFolder/topstartupio50.json'):
    """
    Test the extraction of seed funding stages from topstartupio50.json.
    
    Args:
        json_path (str): Path to the JSON file
        
    Returns:
        None: Outputs results to console
    """
    logger.info(f"Testing seed funding extraction from {json_path}")
    
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
        
        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame(data.get('startups', []))
        else:
            logger.error("Unexpected JSON format")
            return
        
        logger.info(f"Loaded {len(df)} records from {json_path}")
        
        # Check if funding column exists
        if 'funding' not in df.columns:
            logger.error("No 'funding' column found in the data")
            return
        
        # Apply extraction function
        funding_details = df['funding'].apply(extract_funding_info)
        
        # Create separate columns for extracted values
        df['funding_amount'] = funding_details.apply(lambda x: x[0] if x else None)
        df['funding_stage'] = funding_details.apply(lambda x: x[1] if x else None)
        df['funding_year'] = funding_details.apply(lambda x: x[2] if x else None)
        
        # Count unique funding stages
        stage_counts = df['funding_stage'].value_counts()
        logger.info("Funding stage counts:")
        print(stage_counts)
        
        # Focus on seed rounds
        seed_rounds = df[df['funding_stage'] == 'Seed']
        logger.info(f"Found {len(seed_rounds)} seed rounds")
        
        # Display seed round details
        if not seed_rounds.empty:
            print("\nSeed round details:")
            for i, row in seed_rounds.iterrows():
                print(f"Company: {row.get('name', 'Unknown')}")
                print(f"Funding: {row.get('funding', 'Unknown')}")
                print(f"Extracted Stage: {row.get('funding_stage', 'Unknown')}")
                print(f"Extracted Amount: {row.get('funding_amount', 'Unknown')}")
                print(f"Extracted Year: {row.get('funding_year', 'Unknown')}")
                print("-" * 50)
        else:
            print("No seed rounds found!")
            
        # Check for potential missed seed rounds by looking for 'seed' in funding text
        potential_missed = df[(df['funding_stage'] != 'Seed') & (df['funding'].str.lower().str.contains('seed', na=False))]
        if not potential_missed.empty:
            logger.warning(f"Found {len(potential_missed)} potential missed seed rounds")
            print("\nPotential missed seed rounds:")
            for i, row in potential_missed.iterrows():
                print(f"Company: {row.get('name', 'Unknown')}")
                print(f"Funding: {row.get('funding', 'Unknown')}")
                print(f"Extracted Stage: {row.get('funding_stage', 'Unknown')}")
                print("-" * 50)
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        
def main():
    # Path to JSON file
    json_file = 'JSONFolder/topstartupio50.json'
    
    # Check if file exists
    if not os.path.isfile(json_file):
        print(f"File not found: {json_file}")
        return
    
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Check if 'funding' column exists
    if 'funding' not in df.columns:
        print("No 'funding' column found in the data")
        return
    
    # Count total entries
    total_entries = len(df)
    print(f"Total entries: {total_entries}")
    
    # Extract funding information
    funding_info = df['funding'].apply(lambda x: extract_funding_info(x))
    df['amount'] = funding_info.apply(lambda x: x[0])
    df['stage'] = funding_info.apply(lambda x: x[1])
    df['funding_year'] = funding_info.apply(lambda x: x[2])
    
    # Count entries with Seed funding
    seed_entries = df[df['stage'].str.contains('Seed', case=False, na=False)]
    seed_count = len(seed_entries)
    print(f"Entries with Seed funding: {seed_count}")
    print(f"Percentage with Seed funding: {seed_count/total_entries*100:.2f}%")
    
    # Print first few Seed entries to verify
    print("\nSample Seed funding entries:")
    for idx, row in seed_entries.head(10).iterrows():
        print(f"{row['name']}: {row['funding']} -> Stage: {row['stage']}, Amount: {row['amount']}")
    
    # Check for specific patterns that might be missed
    print("\nChecking for potentially missed Seed funding entries:")
    potential_missed = df[df['funding'].str.contains('Seed', case=False, na=False) & ~df['stage'].str.contains('Seed', case=False, na=False)]
    for idx, row in potential_missed.head(10).iterrows():
        print(f"{row['name']}: {row['funding']} -> Stage: {row['stage']}")

if __name__ == "__main__":
    # Use command line argument for file path if provided, otherwise use default
    json_path = sys.argv[1] if len(sys.argv) > 1 else 'test_data/JSONFolder/topstartupio50.json'
    test_topstartup_seed_extraction(json_path)
    main() 