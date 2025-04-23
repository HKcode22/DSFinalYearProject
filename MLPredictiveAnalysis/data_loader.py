import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import List, Union, Dict, Any, Optional, Tuple
import re
from datetime import datetime
from .schema_validator import FundingDataValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FundingDataLoader:
    """
    A class for loading and processing startup funding data.
    Handles data validation, feature engineering, and merging of multiple data sources.
    """
    
    def __init__(self, data_directory: Union[str, Path]):
        """
        Initialize the FundingDataLoader with a data directory.
        
        Args:
            data_directory (Union[str, Path]): Path to directory containing funding data files
        """
        self.data_directory = Path(data_directory)
        self.validator = FundingDataValidator()
        self.funding_stage_order = {
            'Pre-Seed': 0,
            'Seed': 1,
            'Angel': 1,
            'Series A': 2,
            'Series B': 3,
            'Series C': 4,
            'Series D': 5,
            'Series E': 6,
            'Series F': 7,
            'Series G': 8,
            'Series H': 9,
            'IPO': 10
        }
    
    def load_json_file(self, filename: str) -> pd.DataFrame:
        """
        Load and validate a JSON file containing funding data.
        
        Args:
            filename (str): Name of the JSON file to load
            
        Returns:
            pd.DataFrame: Loaded and validated DataFrame
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        file_path = self.data_directory / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if filename == 'topstartupio50.json':
                df = pd.DataFrame(data)
            elif filename == 'fundraisestartup50.json':
                df = pd.DataFrame(data['companies'])
            elif filename == 'growthlistscrapper.json':
                df = pd.DataFrame(data)
            else:
                raise ValueError(f"Unsupported file format: {filename}")
            
            logger.info(f"Successfully loaded {len(df)} records from {filename}")
            
            # Transform data based on file type
            if filename == 'topstartupio50.json':
                df = self._transform_topstartup_data(df)
            elif filename == 'fundraisestartup50.json':
                df = self._transform_fundraise_data(df)
            elif filename == 'growthlistscrapper.json':
                df = self._transform_growthlist_data(df)
            
            # Validate data
            if not self.validator.validate_schema(df):
                errors = self.validator.get_validation_errors()
                logger.error(f"Schema validation failed for {filename}: {errors}")
                raise ValueError(f"Invalid schema in {filename}")
            
            if not self.validator.validate_content(df):
                errors = self.validator.get_validation_errors()
                logger.warning(f"Content validation issues in {filename}: {errors}")
            
            self.validator.clear_validation_errors()
            return df
            
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {filename}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            raise
    
    def _transform_topstartup_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform topstartup.io data format."""
        # Extract funding information
        df['funding_info'] = df['funding'].apply(self.extract_funding_info)
        df['funding_amount'] = df['funding_info'].apply(lambda x: x[0])
        df['funding_stage'] = df['funding_info'].apply(lambda x: x[1])
        df['funding_date'] = df['funding_info'].apply(lambda x: x[2])
        
        # Extract industry from description
        df['industry'] = df['description'].str.split('\n').str[-1]
        
        # Clean up employee count
        df['employees'] = df['employees'].apply(self._extract_employee_count)
        
        # Rename columns
        df = df.rename(columns={
            'name': 'company_name'
        })
        
        return df[['company_name', 'funding_stage', 'funding_amount', 'funding_date', 'industry', 'employees']]
    
    def _transform_fundraise_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform fundraise.com data format."""
        # Convert funding amount to millions
        df['funding_amount'] = pd.to_numeric(df['Funding_Amount_USD'], errors='coerce') / 1_000_000
        
        # Convert date
        df['funding_date'] = pd.to_datetime(df['Funding_Date'], format='%d-%b-%y')
        
        # Clean up employee count
        df['employees'] = pd.to_numeric(df['Total_Employees'], errors='coerce').fillna(0)
        
        # Standardize funding stage
        df['funding_stage'] = df['Funding_Type'].apply(self._standardize_funding_stage)
        
        # Rename columns
        df = df.rename(columns={
            'Company': 'company_name',
            'Industry': 'industry'
        })
        
        return df[['company_name', 'funding_stage', 'funding_amount', 'funding_date', 'industry', 'employees']]
    
    def _transform_growthlist_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform growthlist.com data format."""
        # Convert funding amount to millions
        df['funding_amount'] = df['funding_amount'].str.replace('$', '').str.replace(',', '')
        df.loc[df['funding_amount'] == '', 'funding_amount'] = '0'  # Replace empty strings with '0'
        df['funding_amount'] = df['funding_amount'].astype(float) / 1_000_000
        
        # Convert date
        df['funding_date'] = pd.to_datetime(df['last_funding_date'], format='%b %Y')
        
        # Set default employee count
        df['employees'] = 0
        
        # Standardize funding stage
        df['funding_stage'] = df['funding_type'].apply(self._standardize_funding_stage)
        
        # Rename columns
        df = df.rename(columns={
            'name': 'company_name'
        })
        
        return df[['company_name', 'funding_stage', 'funding_amount', 'funding_date', 'industry', 'employees']]
    
    def _extract_employee_count(self, employee_str: str) -> float:
        """
        Extract employee count from string.
        
        Args:
            employee_str (str): Employee string (e.g., "11-50 employees")
            
        Returns:
            float: Average employee count
        """
        try:
            # Extract numbers from string
            numbers = re.findall(r'\d+', employee_str)
            if len(numbers) == 2:
                # If range (e.g., "11-50"), take average
                return (float(numbers[0]) + float(numbers[1])) / 2
            elif len(numbers) == 1:
                # If single number
                return float(numbers[0])
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _standardize_funding_stage(self, stage: str) -> str:
        """
        Standardize funding stage string.
        
        Args:
            stage (str): Raw funding stage
            
        Returns:
            str: Standardized funding stage
        """
        stage = str(stage).strip().lower()
        
        # Define mapping of raw stages to standardized stages
        stage_mapping = {
            'series a': 'Series A',
            'series b': 'Series B',
            'series c': 'Series C',
            'series d': 'Series D',
            'series e': 'Series E',
            'series f': 'Series F',
            'series g': 'Series G',
            'seed': 'Seed',
            'angel': 'Angel',
            'pre-seed': 'Pre-Seed',
            'venture - series unknown': 'Venture - Series Unknown',
            'venture': 'Venture - Series Unknown',
            'grant': 'Unknown',
            'undisclosed': 'Unknown',
            'private equity': 'Unknown',
            'debt financing': 'Unknown',
            'initial coin offering': 'Unknown'
        }
        
        return stage_mapping.get(stage, 'Unknown')
    
    def standardize_funding_amount(self, amount_str: str) -> float:
        """
        Convert funding amount string to float value in millions.
        
        Args:
            amount_str (str): Funding amount string (e.g., "1M", "2.5B", "500K")
            
        Returns:
            float: Standardized amount in millions
        """
        if not isinstance(amount_str, str):
            return np.nan
            
        # Remove any currency symbols and whitespace
        amount_str = re.sub(r'[^\d.KMBkmb]', '', amount_str.strip())
        
        try:
            # Extract number and unit
            match = re.match(r'(\d+\.?\d*)([KMBkmb])?', amount_str)
            if not match:
                return np.nan
                
            number = float(match.group(1))
            unit = (match.group(2) or '').upper()
            
            # Convert to millions
            multipliers = {'K': 0.001, 'M': 1, 'B': 1000}
            return number * multipliers.get(unit, 1)
            
        except (ValueError, TypeError):
            return np.nan
    
    def extract_funding_info(self, funding_str: str) -> Tuple[float, str, datetime]:
        """
        Extract funding amount, stage, and date from funding string.
        
        Args:
            funding_str (str): Funding string (e.g., "Bessemer Sequoia $11M Series A in 2024")
            
        Returns:
            Tuple[float, str, datetime]: (amount, stage, date)
        """
        try:
            # Extract amount
            amount_match = re.search(r'\$(\d+(?:\.\d+)?)([MB])', funding_str)
            if amount_match:
                amount = float(amount_match.group(1))
                if amount_match.group(2) == 'B':
                    amount *= 1000
            else:
                amount = 0.0
            
            # Extract stage
            stage_patterns = [
                (r'Series\s+([A-Z])', lambda m: f'Series {m.group(1)}'),
                (r'\bSeed\b', lambda m: 'Seed'),
                (r'\bAngel\b', lambda m: 'Angel'),
                (r'\bPre-Seed\b', lambda m: 'Pre-Seed'),
                (r'Venture\s*-\s*Series\s*Unknown', lambda m: 'Venture - Series Unknown')
            ]
            
            stage = 'Unknown'
            for pattern, formatter in stage_patterns:
                match = re.search(pattern, funding_str, re.IGNORECASE)
                if match:
                    stage = formatter(match)
                    break
            
            # Extract date
            date_match = re.search(r'in\s+(\d{4})', funding_str)
            if date_match:
                year = int(date_match.group(1))
                date = datetime(year, 1, 1)  # Use January 1st as default
            else:
                date = datetime.now()
            
            return amount, stage, date
            
        except Exception as e:
            logger.warning(f"Error extracting funding info from '{funding_str}': {str(e)}")
            return 0.0, 'Unknown', datetime.now()
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features to the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with additional engineered features
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Convert funding_date to datetime if not already
        result['funding_date'] = pd.to_datetime(result['funding_date'])
        
        # Time-based features
        result['funding_year'] = result['funding_date'].dt.year
        result['funding_quarter'] = result['funding_date'].dt.quarter
        
        # Sort by company and date for sequential features
        result = result.sort_values(['company_name', 'funding_date'])
        
        # Calculate days since last funding
        result['days_since_last_funding'] = result.groupby('company_name')['funding_date'].diff().dt.days
        
        # Funding stage features
        result['funding_stage_number'] = result['funding_stage'].map(self.funding_stage_order)
        result['is_early_stage'] = result['funding_stage_number'].apply(lambda x: x <= 2 if pd.notnull(x) else None)
        
        # Amount-based features
        result['funding_amount'] = result['funding_amount'].apply(self.standardize_funding_amount)
        result['funding_amount_log'] = np.log1p(result['funding_amount'])
        result['cumulative_funding'] = result.groupby('company_name')['funding_amount'].cumsum()
        
        # Sequence features
        result['funding_sequence'] = result.groupby('company_name').cumcount() + 1
        result['funding_count'] = result.groupby('company_name')['funding_sequence'].transform('max')
        
        # Growth metrics
        result['funding_growth'] = result.groupby('company_name')['funding_amount'].pct_change()
        result['time_between_rounds'] = result.groupby('company_name')['funding_date'].diff().dt.days
        
        # Industry features
        result['industry_funding_mean'] = result.groupby('industry')['funding_amount'].transform('mean')
        
        return result
    
    def merge_dataframes(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple DataFrames with conflict resolution.
        
        Args:
            dfs (List[pd.DataFrame]): List of DataFrames to merge
            
        Returns:
            pd.DataFrame: Merged DataFrame
        """
        if not dfs:
            return pd.DataFrame()
            
        # Start with the first DataFrame
        result = dfs[0].copy()
        
        # Merge with remaining DataFrames
        for df in dfs[1:]:
            # Merge on company name and funding date
            result = pd.concat([result, df])
            
            # Remove duplicates, keeping the first occurrence
            result = result.drop_duplicates(
                subset=['company_name', 'funding_date'],
                keep='first'
            )
            
        # Sort by company name and funding date
        result = result.sort_values(['company_name', 'funding_date'])
        
        logger.info(f"Merged {len(dfs)} DataFrames, resulting in {len(result)} records")
        return result
    
    def load_and_process_data(self, filenames: List[str]) -> pd.DataFrame:
        """
        Load multiple JSON files, merge them, and engineer features.
        
        Args:
            filenames (List[str]): List of JSON filenames to process
            
        Returns:
            pd.DataFrame: Processed DataFrame ready for analysis
        """
        # Load all DataFrames
        dfs = []
        for filename in filenames:
            try:
                df = self.load_json_file(filename)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                continue
        
        if not dfs:
            logger.warning("No data was successfully loaded")
            return pd.DataFrame()
        
        # Merge DataFrames
        merged_df = self.merge_dataframes(dfs)
        
        # Engineer features
        processed_df = self.engineer_features(merged_df)
        
        logger.info(f"Successfully processed {len(processed_df)} records")
        return processed_df 