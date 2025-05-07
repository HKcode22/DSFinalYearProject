import json
import re
import pandas as pd
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_funding_info(funding_str):
    """
    Enhanced function to extract funding information from various string formats.
    
    Args:
        funding_str (str): The funding information string
        
    Returns:
        tuple: (amount, stage, date)
    """
    if not funding_str or pd.isna(funding_str):
        return None, None, None
    
    # Common patterns: "Sequoia $100M Series D in 2025"
    # or "Andreessen Horowitz $10B Series J in 2024 $62.0B valuation"
    
    amount = None
    stage = None
    date = None
    
    # Extract amount
    amount_match = re.search(r'\$(\d+(?:\.\d+)?[KMB]?)', funding_str)
    if amount_match:
        amount = amount_match.group(0)  # Keep the $ symbol
    
    # Extract stage with improved pattern matching
    # Look for more funding stage patterns with case insensitivity
    stage_pattern = r'(Pre[-\s]?Seed|Seed|Angel|Series\s+[A-Z]|Venture[\s\-]+Series\s+Unknown|Initial\s+Coin\s+Offering|ICO|Private\s+Equity|Grant|Debt\s+Financing|Undisclosed|Post[-\s]?IPO)'
    stage_match = re.search(stage_pattern, funding_str, re.IGNORECASE)
    
    if stage_match:
        # Standardize the stage format
        raw_stage = stage_match.group(1)
        
        # Normalize stage name
        if re.match(r'pre[-\s]?seed', raw_stage, re.IGNORECASE):
            stage = 'Pre-Seed'
        elif re.match(r'seed', raw_stage, re.IGNORECASE):
            stage = 'Seed'
        elif re.match(r'angel', raw_stage, re.IGNORECASE):
            stage = 'Angel'
        elif re.match(r'series\s+([a-z])', raw_stage, re.IGNORECASE):
            # Ensure proper capitalization for series (e.g. "Series A")
            series_letter = re.match(r'series\s+([a-z])', raw_stage, re.IGNORECASE).group(1).upper()
            stage = f'Series {series_letter}'
        elif re.match(r'venture[-\s]+series[-\s]+unknown', raw_stage, re.IGNORECASE):
            stage = 'venture - series unknown'
        elif re.match(r'initial\s+coin\s+offering|ico', raw_stage, re.IGNORECASE):
            stage = 'initial coin offering'
        elif re.match(r'private\s+equity', raw_stage, re.IGNORECASE):
            stage = 'Private Equity'
        elif re.match(r'grant', raw_stage, re.IGNORECASE):
            stage = 'Grant'
        elif re.match(r'debt\s+financing', raw_stage, re.IGNORECASE):
            stage = 'debt financing'
        elif re.match(r'undisclosed', raw_stage, re.IGNORECASE):
            stage = 'undisclosed'
        elif re.match(r'post[-\s]?ipo', raw_stage, re.IGNORECASE):
            stage = 'Post-IPO'
        else:
            stage = raw_stage  # Use as-is if no specific match
    else:
        # If no explicit stage is found, try to infer from context
        # Check for common patterns in funding text
        funding_lower = funding_str.lower()
        
        if 'seed' in funding_lower and not stage:
            stage = 'Seed'
        elif 'angel' in funding_lower and not stage:
            stage = 'Angel'
        elif 'raised' in funding_lower and not stage:
            # For strings like "Raised $5M in 2019" without explicit stage
            if 'series' in funding_lower:
                # Try to extract series letter if mentioned
                series_match = re.search(r'series\s+([a-z])', funding_lower)
                if series_match:
                    stage = f'Series {series_match.group(1).upper()}'
                else:
                    stage = 'venture - series unknown'
            else:
                # Default to "Venture Funding" for generic raised amounts
                stage = 'venture - series unknown'
        elif 'valuation' in funding_lower and not stage:
            if 'post-ipo' in funding_lower or 'post ipo' in funding_lower:
                stage = 'Post-IPO'
            else:
                # Companies with just valuation mentioned but no explicit funding stage
                stage = 'venture - series unknown'
    
    # Extract date - usually has "in YYYY" format
    date_match = re.search(r'in (\d{4})', funding_str)
    if date_match:
        date = date_match.group(1)
    else:
        # Try to find just a year at the end of the string
        year_match = re.search(r'\b(20\d{2})\b', funding_str)
        if year_match:
            date = year_match.group(1)
    
    return amount, stage, date

def main():
    """
    Test the enhanced funding stage extraction on topstartupio50.json.
    """
    # Load the topstartupio50.json file
    json_path = '../JSONFolder/topstartupio50.json'
    
    if not os.path.exists(json_path):
        logger.error(f"File not found: {json_path}")
        return
    
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
        
        # Create a dataframe
        if isinstance(data, list):
            df = pd.DataFrame(data)
            
            # Test the extraction function
            logger.info(f"Processing {len(df)} records from topstartupio50.json")
            
            results = []
            for index, row in df.iterrows():
                if 'funding' in row and row['funding']:
                    amount, stage, date = extract_funding_info(row['funding'])
                    
                    results.append({
                        'company': row['name'],
                        'original_funding_text': row['funding'],
                        'extracted_amount': amount,
                        'extracted_stage': stage,
                        'extracted_date': date
                    })
            
            # Create a new DataFrame with the results
            results_df = pd.DataFrame(results)
            
            # Calculate stage distribution
            stage_counts = results_df['extracted_stage'].value_counts()
            
            # Print summary statistics
            logger.info(f"Successfully processed {len(results_df)} records")
            logger.info(f"Found {len(stage_counts)} unique funding stages")
            
            # Print the stage distribution
            for stage, count in stage_counts.items():
                percentage = (count / len(results_df)) * 100
                logger.info(f"{stage}: {count} records ({percentage:.1f}%)")
            
            # Check for missing stages (records with no stage identified)
            missing_stages = results_df[results_df['extracted_stage'].isna()]
            if not missing_stages.empty:
                logger.warning(f"Found {len(missing_stages)} records with no funding stage identified:")
                for _, row in missing_stages.iterrows():
                    logger.warning(f"Company: {row['company']}, Funding: {row['original_funding_text']}")
            
            # Save results to CSV for further analysis
            output_path = 'topstartup_funding_extraction.csv'
            results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
            
        else:
            logger.error("Unexpected JSON format - not a list")
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 