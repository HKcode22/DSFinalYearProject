import json
import pandas as pd
import re
import sys

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

def show_all_companies(json_path):
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Check if 'funding' column exists
    if 'funding' not in df.columns:
        print("No 'funding' column found in the data")
        return
    
    # Extract funding information
    funding_info = df['funding'].apply(lambda x: extract_funding_info(x))
    df['amount'] = funding_info.apply(lambda x: x[0])
    df['stage'] = funding_info.apply(lambda x: x[1])
    df['funding_year'] = funding_info.apply(lambda x: x[2])
    
    # Save all results to CSV
    result_df = df[['name', 'stage', 'funding']].copy()
    result_df.rename(columns={'name': 'Company', 'stage': 'Extracted_Stage', 'funding': 'Original_Funding_String'}, inplace=True)
    result_df.to_csv('all_funding_stages.csv', index=False)
    
    print(f"Saved all {len(result_df)} companies with their funding details to all_funding_stages.csv")
    
    # Print summary counts
    stage_counts = df['stage'].value_counts()
    print("\nFunding stage distribution:")
    for stage, count in stage_counts.items():
        if pd.notna(stage):
            print(f"{stage}: {count}")

if __name__ == "__main__":
    # Use command line argument for file path if provided, otherwise use default
    json_path = sys.argv[1] if len(sys.argv) > 1 else 'JSONFolder/topstartupio50.json'
    show_all_companies(json_path) 