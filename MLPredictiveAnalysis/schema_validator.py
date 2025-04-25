from typing import Dict, Any, List
import pandas as pd
from datetime import datetime
import logging
import re

logger = logging.getLogger(__name__)

class FundingDataValidator:
    """Validates funding data schema and content."""
    
    REQUIRED_COLUMNS = {
        'company_name': str,
        'funding_stage': str,
        'funding_amount': float,
        'funding_date': 'datetime64[ns]',
        'industry': str,
        'employees': float
    }
    
    VALID_FUNDING_STAGES = {
        'Seed', 'Angel', 'Series A', 'Series B', 'Series C', 'Series D',
        'Series E', 'Series F', 'Series G', 'Unknown', 'Pre-Seed',
        'Venture - Series Unknown'
    }
    
    def __init__(self):
        """Initialize the validator."""
        self.validation_errors = []
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate the DataFrame schema matches required columns and types.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            bool: True if schema is valid, False otherwise
        """
        # Check required columns exist
        missing_cols = set(self.REQUIRED_COLUMNS.keys()) - set(df.columns)
        if missing_cols:
            self.validation_errors.append(f"Missing required columns: {missing_cols}")
            return False
            
        # Validate data types
        for col, expected_type in self.REQUIRED_COLUMNS.items():
            try:
                if expected_type == 'datetime64[ns]':
                    df[col] = pd.to_datetime(df[col])
                else:
                    df[col] = df[col].astype(expected_type)
            except Exception as e:
                self.validation_errors.append(f"Invalid data type for column {col}: {str(e)}")
                return False
        
        return True
    
    def validate_content(self, df: pd.DataFrame) -> bool:
        """
        Validate the content of the DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            bool: True if content is valid, False otherwise
        """
        # Check for missing values in required columns
        for col in self.REQUIRED_COLUMNS.keys():
            if df[col].isnull().any():
                self.validation_errors.append(f"Missing values in column {col}")
                return False
        
        # Validate funding stages
        invalid_stages = set(df['funding_stage'].unique()) - self.VALID_FUNDING_STAGES
        if invalid_stages:
            self.validation_errors.append(f"Invalid funding stages found: {invalid_stages}")
            return False
        
        # Validate funding amounts
        if (df['funding_amount'] < 0).any():
            self.validation_errors.append("Found negative funding amounts")
            return False
        
        # Validate dates
        min_date = pd.Timestamp('2000-01-01')
        max_date = pd.Timestamp.now()
        invalid_dates = df[
            (df['funding_date'] < min_date) | 
            (df['funding_date'] > max_date)
        ]
        if not invalid_dates.empty:
            self.validation_errors.append(
                f"Found {len(invalid_dates)} funding dates outside valid range"
            )
            return False
        
        # Validate employees
        if (df['employees'] < 0).any():
            self.validation_errors.append("Found negative employee counts")
            return False
        
        return True
    
    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors."""
        return self.validation_errors
    
    def clear_validation_errors(self):
        """Clear validation errors."""
        self.validation_errors = [] 