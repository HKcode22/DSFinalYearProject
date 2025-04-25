import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import xgboost as xgb
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
warnings.filterwarnings('ignore')

# Column standardization mappings
COLUMN_MAP = {
    'Company': 'company_name',
    'name': 'company_name',
    'Total_Employees': 'employees',
    'employee_count': 'employees',
    'Industry': 'industry',
    'industry': 'industry',
    'Funding_Date': 'funding_date',
    'last_funding_date': 'funding_date',
    'Funding_Type': 'funding_stage',
    'funding_type': 'funding_stage',
    'Funding_Amount_USD': 'funding_amount',
    'funding_amount': 'funding_amount'
}

def standardize_columns(df):
    """Standardize column names across different data sources"""
    return df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

def parse_funding(amount_str):
    """Parse funding amount from various formats"""
    if pd.isna(amount_str) or amount_str == "":
        return np.nan
    try:
        if isinstance(amount_str, str):
            amount_str = amount_str.replace('$', '').replace(',', '').strip().upper()
            if 'B' in amount_str:
                return float(amount_str.replace('B', '')) * 1e9
            elif 'M' in amount_str:
                return float(amount_str.replace('M', '')) * 1e6
            elif 'K' in amount_str:
                return float(amount_str.replace('K', '')) * 1e3
        return float(amount_str)
    except:
        return np.nan

def robust_parse_date(date_str):
    """Parse dates with multiple format support"""
    if pd.isna(date_str):
        return pd.NaT
    
    date_formats = ['%Y-%m-%d', '%b %Y', '%d-%b-%y', '%d-%b-%Y', '%Y-%m']
    
    for fmt in date_formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue
    
    try:
        return pd.to_datetime(date_str)
    except:
        return pd.NaT

class IndustryTrendAnalyzer:
    """
    Advanced industry trend analysis using STL decomposition and machine learning
    to identify emerging sectors and predict industry trajectories.
    """
    
    def __init__(self, data_dir="./JSONFolder", output_dir="./outputIndustryTrends"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize logging
        self.logger = self._setup_logging()
        
        # Initialize models and scalers
        self.stl_models = {}
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.growth_predictor = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        
        # Store computed metrics
        self.industry_metrics = {}
        self.emerging_industries = []
        self.saturated_industries = []
        self.industry_forecasts = {}
        self.employee_migration = {}
        
    def _setup_logging(self):
        """Configure logging for the analysis"""
        logger = logging.getLogger('IndustryTrendAnalysis')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        fh = logging.FileHandler(os.path.join(self.output_dir, 'industry_trends.log'))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger
        
    def load_and_merge_data(self):
        """Load and merge data with enhanced deduplication and validation"""
        try:
            # Load individual datasets
            fundraiser_data = self._load_fundraiser()
            growthlist_data = self._load_growthlist()
            topstartup_data = self._load_topstartup()

            # Standardize columns for all dataframes
            fundraiser_data = standardize_columns(fundraiser_data)
            growthlist_data = standardize_columns(growthlist_data)
            topstartup_data = standardize_columns(topstartup_data)

            # Add source column to track data origin
            fundraiser_data['source'] = 'fundraiser'
            growthlist_data['source'] = 'growthlist'
            topstartup_data['source'] = 'topstartup'

            # Convert dates before merging using robust parser
            for df in [fundraiser_data, growthlist_data, topstartup_data]:
                if 'funding_date' in df.columns:
                    df['funding_date'] = df['funding_date'].apply(robust_parse_date)

            # Parse funding amounts using robust parser
            for df in [fundraiser_data, growthlist_data, topstartup_data]:
                if 'funding_amount' in df.columns:
                    df['funding_amount'] = df['funding_amount'].apply(parse_funding)

            # Initialize employee count if missing
            for df in [fundraiser_data, growthlist_data, topstartup_data]:
                if 'employees' not in df.columns:
                    df['employees'] = np.nan

            # Merge with priority (keep most complete record)
            merged = pd.concat([fundraiser_data, growthlist_data, topstartup_data])
            merged['completeness'] = merged.notna().sum(axis=1)
            merged = merged.sort_values('completeness', ascending=False)
            merged = merged.drop_duplicates(
                subset=['company_name', 'funding_date'], 
                keep='first'
            )
            merged = merged.drop(columns=['completeness', 'source'])

            # Clean and calculate survival metrics
            processed = self._clean_data(merged)
            final_df = self._calculate_survival_metrics(processed)

            # Validate final dataset
            if len(final_df) == 0:
                self.logger.error("No valid records after processing")
                return pd.DataFrame()

            # Log data quality metrics
            missing_amounts = final_df['funding_amount'].isna().sum()
            missing_dates = final_df['funding_date'].isna().sum()
            
            if missing_amounts > 0:
                self.logger.warning(f"{missing_amounts} records still have missing funding amounts")
            if missing_dates > 0:
                self.logger.warning(f"{missing_dates} records still have missing dates")

            # Remove any remaining invalid records
            final_df = final_df.dropna(subset=['funding_amount', 'funding_date'])
            
            self.logger.info(f"Final dataset contains {len(final_df)} valid records")
            return final_df

        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def _calculate_growth_rate(self, values):
        """
        Calculate growth rate with improved accuracy and stability.
        
        Args:
            values (np.ndarray): Array of values to calculate growth rate from
            
        Returns:
            float: Calculated growth rate
        """
        # Input validation
        if not isinstance(values, np.ndarray) or len(values) < 2:
            return 0.0
        
        # Remove zeros and NaN values
        valid_values = values[values > 0]
        if len(valid_values) < 2:
            return 0.0
        
        # Calculate period-over-period growth rates
        growth_rates = np.diff(valid_values) / valid_values[:-1]
        
        # Calculate recent growth (last period)
        recent_growth = growth_rates[-1] if len(growth_rates) > 0 else 0
        
        # Calculate compound annual growth rate (CAGR)
        total_periods = len(valid_values) - 1
        if total_periods > 0:
            cagr = (valid_values[-1] / valid_values[0]) ** (1/total_periods) - 1
        else:
            cagr = 0
        
        # Weight recent growth more heavily for emerging trends
        # but maintain stability with CAGR
        weighted_growth = 0.6 * recent_growth + 0.4 * cagr
        
        # Apply more reasonable limits to growth rate
        # Max growth: 150% (2.5x), Min growth: -60% (0.4x)
        return np.clip(weighted_growth, -0.6, 1.5)

    def prepare_trend_features(self, df):
        """Create features for trend analysis"""
        self.logger.info("Preparing trend features...")
        
        try:
            # Create time-based features
            df['year'] = df['funding_date'].dt.year
            df['month'] = df['funding_date'].dt.month
            df['quarter'] = df['funding_date'].dt.quarter
            
            # Calculate funding metrics by industry
            industry_metrics = {}
            
            for industry in df['industry'].unique():
                if pd.isna(industry) or industry == '0':
                    continue
                    
                industry_data = df[df['industry'] == industry]
                
                # Monthly funding aggregation
                monthly_funding = industry_data.groupby(
                    [industry_data['funding_date'].dt.to_period('M')]
                )['funding_amount'].agg(['sum', 'count', 'mean'])
                
                # Calculate growth rates
                funding_growth = self._calculate_growth_rate(monthly_funding['sum'].tail(6))
                deal_growth = self._calculate_growth_rate(monthly_funding['count'].tail(6))
                
                # Calculate market concentration (HHI)
                total_funding = industry_data['funding_amount'].sum()
                company_shares = industry_data.groupby('company_name')['funding_amount'].sum() / total_funding
                hhi = (company_shares ** 2).sum()
                
                # Employee metrics
                emp_data = industry_data.copy()
                emp_data['employees'] = pd.to_numeric(emp_data['employees'], errors='coerce')
                emp_data = emp_data[emp_data['employees'].notna()]
                
                if len(emp_data) > 0:
                    emp_growth = self._calculate_growth_rate(emp_data.groupby('funding_date')['employees'].mean().tail(6))
                    emp_concentration = emp_data.groupby('company_name')['employees'].sum().std()
                else:
                    emp_growth = 0
                    emp_concentration = 0
                
                # Store metrics
                industry_metrics[industry] = {
                    'monthly_funding': monthly_funding,
                    'funding_growth': funding_growth,
                    'deal_growth': deal_growth,
                    'hhi': hhi,
                    'employee_growth': emp_growth,
                    'employee_concentration': emp_concentration,
                    'total_companies': len(industry_data['company_name'].unique()),
                    'total_funding': total_funding,
                    'avg_round_size': industry_data['funding_amount'].mean()
                }
            
            self.industry_metrics = industry_metrics
            return industry_metrics
            
        except Exception as e:
            self.logger.error(f"Error preparing trend features: {str(e)}")
            raise
            
    def analyze_industry_trends(self, df):
        """Analyze industry trends and identify emerging/saturated industries."""
        try:
            self.logger.info("Analyzing industry trends...")
            
            # Ensure funding_date is datetime and employees is numeric
            df['funding_date'] = pd.to_datetime(df['funding_date'])
            if 'employees' in df.columns:
                df['employees'] = pd.to_numeric(df['employees'], errors='coerce')
            
            # Expand multiple industries
            df_expanded = df.copy()
            df_expanded['industry'] = df_expanded['industry'].fillna('Unknown')
            df_expanded['industry'] = df_expanded['industry'].str.split(',')
            df_expanded = df_expanded.explode('industry')
            df_expanded['industry'] = df_expanded['industry'].str.strip()
            
            # Calculate total funding in last 12 months for market share
            last_date = df_expanded['funding_date'].max()
            recent_mask = df_expanded['funding_date'] >= (last_date - pd.DateOffset(months=12))
            total_recent_funding = df_expanded.loc[recent_mask, 'funding_amount'].sum()
            
            # Calculate industry maturity thresholds
            median_funding = df_expanded['funding_amount'].median()
            median_companies = df_expanded.groupby('industry')['company_name'].nunique().median()
            
            industry_metrics = {}
            emerging_industries = []
            saturated_industries = []
            
            # Get industry-level stats
            industry_totals = df_expanded.groupby('industry').agg({
                'funding_amount': 'sum',
                'company_name': 'nunique'
            }).sort_values('funding_amount', ascending=False)
            
            # Identify the top industries by funding (potential for saturation)
            top_industries = industry_totals.head(10).index.tolist()
            
            for industry in df_expanded['industry'].unique():
                if pd.isna(industry) or industry == '0' or industry == 'Unknown':
                    continue
                    
                industry_data = df_expanded[df_expanded['industry'] == industry].copy()
                if len(industry_data) < 5:  # Require minimum 5 companies for meaningful analysis
                    continue
                    
                # Calculate market metrics with improved accuracy
                industry_recent_mask = industry_data['funding_date'] >= (last_date - pd.DateOffset(months=12))
                recent_industry_data = industry_data[industry_recent_mask]
                
                # Calculate total funding and market share
                total_funding = industry_data['funding_amount'].sum()
                recent_funding = recent_industry_data['funding_amount'].sum()
                market_share = recent_funding / total_recent_funding if total_recent_funding > 0 else 0
                
                # Calculate normalized HHI (market concentration)
                if len(recent_industry_data) > 0:
                    company_shares = recent_industry_data.groupby('company_name')['funding_amount'].sum()
                    total_industry_funding = company_shares.sum()
                    if total_industry_funding > 0:
                        company_shares = company_shares / total_industry_funding
                        # Normalize HHI to 0-1 range based on number of companies
                        n_companies = len(company_shares)
                        min_hhi = 1 / n_companies if n_companies > 0 else 0  # Perfect competition
                        hhi = (company_shares ** 2).sum()
                        normalized_hhi = (hhi - min_hhi) / (1 - min_hhi) if (1 - min_hhi) > 0 else 0  # Scale to 0-1
                        concentration = normalized_hhi
                    else:
                        concentration = 0
                else:
                    concentration = 0
                
                # Calculate quarterly metrics with improved accuracy
                quarters = pd.date_range(end=last_date, periods=4, freq='Q')
                quarterly_metrics = []
                
                for i in range(len(quarters)-1):
                    quarter_data = industry_data[
                        (industry_data['funding_date'] > quarters[i]) & 
                        (industry_data['funding_date'] <= quarters[i+1])
                    ]
                    
                    metrics = {
                        'funding': quarter_data['funding_amount'].sum(),
                        'deals': len(quarter_data),
                        'employees': quarter_data['employees'].mean() if 'employees' in quarter_data.columns else 0,
                        'avg_deal_size': (quarter_data['funding_amount'].sum() / len(quarter_data)) if len(quarter_data) > 0 else 0
                    }
                    quarterly_metrics.append(metrics)
                
                # Extract time series for each metric
                funding_series = np.array([m['funding'] for m in quarterly_metrics])
                deals_series = np.array([m['deals'] for m in quarterly_metrics])
                employee_series = np.array([m['employees'] for m in quarterly_metrics])
                deal_size_series = np.array([m['avg_deal_size'] for m in quarterly_metrics])
                
                # Calculate growth rates with improved method
                funding_growth = self._calculate_growth_rate(funding_series)
                deal_growth = self._calculate_growth_rate(deals_series)
                emp_growth = self._calculate_growth_rate(employee_series)
                deal_size_growth = self._calculate_growth_rate(deal_size_series)
                
                # Calculate weighted growth score with balanced weights
                growth_score = (
                    0.4 * funding_growth +    # Increased weight on funding growth
                    0.3 * deal_growth +       # Deal volume is important
                    0.2 * emp_growth +        # Employee growth indicates scaling
                    0.1 * deal_size_growth    # Deal size growth shows maturity
                )
                
                # Calculate momentum using exponential moving average
                if len(funding_series) >= 2:
                    # Calculate exponential weights
                    alpha = 0.7  # Decay factor
                    weights = alpha * (1 - alpha) ** np.arange(len(funding_series))
                    weights = weights / weights.sum()  # Normalize weights
                    
                    # Calculate weighted average growth
                    weighted_funding = np.sum(funding_series * weights)
                    momentum = (funding_series[-1] / weighted_funding - 1) if weighted_funding > 0 else 0
                    
                    # Normalize momentum to reasonable range
                    momentum = np.clip(momentum, -0.5, 1.0)
                else:
                    momentum = 0
                
                # Calculate maturity score with improved balance
                num_companies = len(recent_industry_data['company_name'].unique())
                avg_funding = total_funding / len(industry_data) if len(industry_data) > 0 else 0
                
                # Normalize components to 0-1 range
                market_share_norm = market_share / 0.25  # Normalize to 25% market share
                concentration_norm = concentration  # Already normalized
                companies_norm = min(num_companies / (2 * median_companies), 1)  # Cap at 2x median
                funding_norm = min(avg_funding / (2 * median_funding), 1)  # Cap at 2x median
                
                maturity = (
                    0.35 * market_share_norm +     # Market presence
                    0.25 * concentration_norm +    # Market concentration
                    0.25 * companies_norm +        # Industry scale
                    0.15 * funding_norm            # Funding scale
                )
                
                # Store metrics
                metrics = {
                    'quarterly_metrics': quarterly_metrics,
                    'funding_growth': funding_growth,
                    'deal_growth': deal_growth,
                    'employee_growth': emp_growth,
                    'deal_size_growth': deal_size_growth,
                    'concentration': concentration,
                    'total_companies': num_companies,
                    'total_funding': total_funding,
                    'avg_round_size': avg_funding,
                    'growth_score': growth_score,
                    'momentum': momentum,
                    'maturity': maturity,
                    'market_share': market_share
                }
                
                industry_metrics[industry] = metrics
                
                # Identify emerging industries with improved criteria
                quarterly_growth = funding_series
                if len(quarterly_growth) >= 2:
                    recent_quarterly_growth = (quarterly_growth[-1] / quarterly_growth[0] - 1) if quarterly_growth[0] > 0 else 0
                else:
                    recent_quarterly_growth = 0

                # Calculate average quarterly metrics
                avg_quarterly_funding = np.mean(funding_series) if len(funding_series) > 0 else 0
                avg_quarterly_deals = np.mean(deals_series) if len(deals_series) > 0 else 0
                
                # Identify emerging industries with refined criteria
                is_emerging = (
                    num_companies >= 3 and                    # At least 3 companies
                    total_funding > 5e5 and                   # At least $500K total funding
                    maturity < 0.5 and                        # Not yet mature (normalized)
                    market_share < 0.1 and                    # Less than 10% market share
                    (
                        recent_quarterly_growth > 0.3 or      # 30% quarterly growth
                        emp_growth > 0.5 or                   # 50% employee growth
                        (
                            avg_quarterly_funding > 1e6 and   # Average $1M per quarter
                            avg_quarterly_deals >= 2          # At least 2 deals per quarter
                        )
                    )
                )
                
                if is_emerging:
                    emerging_industries.append({
                        'industry': industry,
                        'funding_growth': recent_quarterly_growth * 100,  # Convert to percentage
                        'employee_growth': emp_growth * 100,              # Convert to percentage
                        'deal_growth': deal_growth * 100,                 # Convert to percentage
                        'market_share': market_share * 100,               # Convert to percentage
                        'companies': num_companies,
                        'total_funding': total_funding,
                        'avg_quarterly_funding': avg_quarterly_funding,
                        'avg_quarterly_deals': avg_quarterly_deals
                    })
                
                # Identify saturated industries with improved criteria
                # We'll use multiple approaches to identify saturation
                
                # 1. Major industries with high market share and concentration
                is_major_player = (
                    market_share > 0.05 and         # At least 5% market share
                    concentration > 0.5 and         # High concentration
                    total_funding > 50e6            # At least $50M funding
                )
                
                # 2. Slowing growth in established sectors
                is_slowing_established = (
                    industry in top_industries and  # Top funded industry
                    num_companies >= 10 and         # Established player count
                    funding_growth < 0.1 and        # Slow funding growth
                    maturity > 0.4                  # Moderate to high maturity
                )
                
                # 3. Industries reaching saturation point
                is_at_saturation = (
                    maturity > 0.6 and              # High maturity score
                    momentum < 0 and                # Negative momentum
                    num_companies >= 5 and          # Reasonable number of players
                    total_funding > 10e6            # Significant total funding
                )
                
                # Combine criteria (any can qualify)
                if is_major_player or is_slowing_established or is_at_saturation:
                    saturated_industries.append({
                        'industry': industry,
                        'maturity': maturity,
                        'momentum': momentum,
                        'market_share': market_share * 100,  # Convert to percentage
                        'concentration': concentration,
                        'funding_growth': funding_growth * 100,  # Convert to percentage
                        'total_funding': total_funding,
                        'is_major_player': is_major_player,
                        'is_slowing_established': is_slowing_established,
                        'is_at_saturation': is_at_saturation,
                        'companies': num_companies
                    })
            
            # Sort results
            self.emerging_industries = sorted(emerging_industries, key=lambda x: x['funding_growth'], reverse=True)
            self.saturated_industries = sorted(saturated_industries, key=lambda x: x['maturity'], reverse=True)
            self.industry_metrics = industry_metrics
            
            # Log summary
            self.logger.info("\nIndustry Analysis Summary:")
            self.logger.info(f"Total Industries Analyzed: {len(industry_metrics)}")
            self.logger.info(f"Emerging Industries Found: {len(emerging_industries)}")
            self.logger.info(f"Saturated Industries Found: {len(saturated_industries)}")
            self.logger.info(f"Industries with Forecasts: {len(industry_metrics)}")
            
            # Log top emerging industries
            self.logger.info("\nTop 5 Emerging Industries:")
            for industry in self.emerging_industries[:5]:
                self.logger.info(f"{industry['industry']}:")
                self.logger.info(f"  - Quarterly Funding Growth: {industry['funding_growth']:.1f}%")
                self.logger.info(f"  - Employee Growth: {industry['employee_growth']:.1f}%")
                self.logger.info(f"  - Deal Growth: {industry['deal_growth']:.1f}%")
                self.logger.info(f"  - Market Share: {industry['market_share']:.2f}%")
                self.logger.info(f"  - Active Companies: {industry['companies']}")
                self.logger.info(f"  - Avg Quarterly Funding: ${industry['avg_quarterly_funding']:.2f}")
                self.logger.info(f"  - Avg Quarterly Deals: {industry['avg_quarterly_deals']}")
                self.logger.info(f"  - Total Funding: ${industry['total_funding']:.2f}")
            
            # Log top saturated industries
            self.logger.info("\nTop 5 Saturated Industries by Maturity:")
            for industry in self.saturated_industries[:5]:
                self.logger.info(f"{industry['industry']}:")
                self.logger.info(f"  - Maturity: {industry['maturity']:.2f}")
                self.logger.info(f"  - Market Share: {industry['market_share']:.2f}%")
                self.logger.info(f"  - Concentration: {industry['concentration']:.2f}")
                self.logger.info(f"  - Funding Growth: {industry['funding_growth']:.1f}%")
                self.logger.info(f"  - Momentum: {industry['momentum']*100:.1f}%")
                self.logger.info(f"  - Companies: {industry['companies']}")
                self.logger.info(f"  - Total Funding: ${industry['total_funding']:.2f}")
                saturation_reasons = []
                if industry['is_major_player']:
                    saturation_reasons.append("Major Market Player")
                if industry['is_slowing_established']:
                    saturation_reasons.append("Slowing Established Sector")
                if industry['is_at_saturation']:
                    saturation_reasons.append("Reached Saturation Point")
                self.logger.info(f"  - Saturation Indicators: {', '.join(saturation_reasons)}")
            
            return industry_metrics, emerging_industries, saturated_industries
            
        except Exception as e:
            self.logger.error(f"Error analyzing industry trends: {str(e)}")
            raise
            
    def analyze_employee_migration(self, df):
        """Analyze employee movement patterns between industries with improved accuracy"""
        self.logger.info("Analyzing employee migration patterns...")
        
        try:
            # Clean and prepare data
            df_clean = df.copy()
            
            # Convert employees to numeric and handle missing values
            df_clean['employees'] = pd.to_numeric(df_clean['employees'], errors='coerce')
            df_clean = df_clean[df_clean['employees'].notna() & (df_clean['employees'] > 0)]
            
            # Calculate employee changes over time by industry
            df_clean['month'] = df_clean['funding_date'].dt.to_period('M')
            
            # Get the last 12 months of data
            last_date = df_clean['funding_date'].max()
            cutoff_date = last_date - pd.DateOffset(months=12)
            recent_data = df_clean[df_clean['funding_date'] >= cutoff_date]
            
            # Store results
            migration_patterns = {}
            
            # Process each industry
            for industry in df_clean['industry'].unique():
                if pd.isna(industry) or industry == '0':
                    continue
                    
                industry_data = recent_data[recent_data['industry'] == industry]
                if len(industry_data) < 3:  # Need minimum data points
                    continue
                
                # Calculate monthly employee counts using proper aggregation
                monthly_data = industry_data.groupby('month').agg({
                    'employees': ['mean', 'count', 'std'],
                    'company_name': 'nunique'
                })
                
                # Ensure we have enough monthly data and companies
                if len(monthly_data) < 2 or monthly_data['company_name']['nunique'].max() < 1:
                    continue
                    
                # Calculate metrics with better handling of missing values
                avg_employees = monthly_data['employees']['mean'].mean()
                
                # Default values if data is insufficient
                employee_volatility = 0.0
                size_inequality = 0.0
                
                # Calculate employee volatility only if we have sufficient data
                if len(monthly_data) >= 2 and not np.isnan(monthly_data['employees']['std'].mean()):
                    employee_volatility = monthly_data['employees']['std'].mean() / max(avg_employees, 1)
                
                # Calculate company size distribution if we have multiple companies
                company_size_distribution = industry_data.groupby('company_name')['employees'].mean()
                if len(company_size_distribution) > 1 and company_size_distribution.mean() > 0:
                    size_inequality = company_size_distribution.std() / company_size_distribution.mean()
                
                # Calculate growth rate using the improved method
                monthly_employees = monthly_data['employees']['mean'].values
                if len(monthly_employees) >= 2:
                    growth_rate = self._calculate_growth_rate(monthly_employees)
                    
                    # Store comprehensive metrics
                    migration_patterns[industry] = {
                        'growth_rate': growth_rate * 100,  # Convert to percentage
                        'avg_employees': avg_employees,
                        'total_companies': monthly_data['company_name']['nunique'].max(),
                        'employee_volatility': employee_volatility,
                        'size_inequality': size_inequality,
                        'monthly_data_points': len(monthly_data)
                    }
            
            # Store results
            self.employee_migration = migration_patterns
            
            # Sort industries by growth rate
            sorted_industries = sorted(
                migration_patterns.items(),
                key=lambda x: x[1]['growth_rate'],
                reverse=True
            )
            
            # Calculate summary statistics
            growing_industries = sum(1 for _, v in migration_patterns.items() if v['growth_rate'] > 10)
            moderate_growth = sum(1 for _, v in migration_patterns.items() if 0 < v['growth_rate'] <= 10)
            stable_industries = sum(1 for _, v in migration_patterns.items() if abs(v['growth_rate']) <= 5)
            shrinking_industries = sum(1 for _, v in migration_patterns.items() if v['growth_rate'] < 0)
            
            # Log enhanced summary statistics
            self.logger.info("\nEmployee migration analysis summary:")
            self.logger.info(f"Total industries analyzed: {len(migration_patterns)}")
            self.logger.info(f"High growth industries (>10%): {growing_industries}")
            self.logger.info(f"Moderate growth industries (0-10%): {moderate_growth}")
            self.logger.info(f"Stable industries (±5% growth): {stable_industries}")
            self.logger.info(f"Shrinking industries: {shrinking_industries}")
            
            # Log detailed metrics for top growing industries
            self.logger.info("\nTop 5 fastest growing industries:")
            for industry, metrics in sorted_industries[:5]:
                self.logger.info(
                    f"{industry}:\n"
                    f"  - Growth Rate: {metrics['growth_rate']:.1f}%\n"
                    f"  - Avg Employees: {metrics['avg_employees']:.0f}\n"
                    f"  - Companies: {metrics['total_companies']}\n"
                    f"  - Employee Volatility: {metrics['employee_volatility']:.2f}\n"
                    f"  - Size Inequality: {metrics['size_inequality']:.2f}\n"
                    f"  - Monthly Data Points: {metrics['monthly_data_points']}"
                )
            
            # Log detailed metrics for fastest shrinking industries
            self.logger.info("\nTop 5 fastest shrinking industries:")
            for industry, metrics in sorted_industries[-min(5, len(sorted_industries)):]:
                self.logger.info(
                    f"{industry}:\n"
                    f"  - Growth Rate: {metrics['growth_rate']:.1f}%\n"
                    f"  - Avg Employees: {metrics['avg_employees']:.0f}\n"
                    f"  - Companies: {metrics['total_companies']}\n"
                    f"  - Employee Volatility: {metrics['employee_volatility']:.2f}\n"
                    f"  - Size Inequality: {metrics['size_inequality']:.2f}\n"
                    f"  - Monthly Data Points: {metrics['monthly_data_points']}"
                )
            
            return migration_patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing employee migration: {str(e)}")
            raise
            
    def generate_report(self):
        """Generate comprehensive industry trend report"""
        self.logger.info("Generating industry trend report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'emerging_industries': self.emerging_industries[:10],  # Top 10
            'saturated_industries': self.saturated_industries[:10],  # Top 10
            'industry_metrics': self.industry_metrics,
            'forecasts': self.industry_forecasts,
            'employee_migration': self.employee_migration
        }
        
        # Save report
        report_path = os.path.join(self.output_dir, 'industry_trend_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        self.logger.info(f"Report saved to {report_path}")
        return report
        
    def visualize_trends(self):
        """Create comprehensive visualizations of industry trends"""
        self.logger.info("Creating trend visualizations...")
        
        try:
            # Set style
            plt.style.use('default')  # Use default style instead of seaborn
            
            # 1. Emerging Industries Plot
            plt.figure(figsize=(15, 8))
            industries = [x['industry'] for x in self.emerging_industries[:10]]
            funding_growth = [x['funding_growth'] for x in self.emerging_industries[:10]]
            employee_growth = [x['employee_growth'] for x in self.emerging_industries[:10]]
            market_share = [x['market_share'] for x in self.emerging_industries[:10]]
            
            if industries:  # Only create plot if we have data
                # Create subplot for growth metrics
                plt.subplot(1, 2, 1)
                bars = plt.barh(industries, funding_growth, alpha=0.8, color='skyblue')
                plt.title('Top 10 Emerging Industries', fontsize=12, pad=20)
                plt.xlabel('Funding Growth (%)')
                
                # Add value labels
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    plt.text(width, bar.get_y() + bar.get_height()/2,
                            f'{funding_growth[i]:.1f}%',
                            ha='left', va='center', fontsize=10)
                
                # Add employee growth and market share
                plt.subplot(1, 2, 2)
                x = np.arange(len(industries))
                width = 0.35
                
                plt.bar(x - width/2, employee_growth, width, label='Employee Growth (%)', alpha=0.8, color='lightgreen')
                plt.bar(x + width/2, market_share, width, label='Market Share (%)', alpha=0.8, color='coral')
                plt.xticks(x, industries, rotation=45, ha='right')
                plt.title('Growth Metrics by Industry', fontsize=12, pad=20)
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'emerging_industries.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            # 2. Saturated Industries Plot
            plt.figure(figsize=(15, 8))
            if self.saturated_industries:  # Only create plot if we have data
                industries = [x['industry'] for x in self.saturated_industries[:10]]
                maturity = [x['maturity'] for x in self.saturated_industries[:10]]
                momentum = [x['momentum'] for x in self.saturated_industries[:10]]
                market_share = [x['market_share'] for x in self.saturated_industries[:10]]
                concentration = [x['concentration'] for x in self.saturated_industries[:10]]
                
                # Create subplot for maturity metrics
                plt.subplot(1, 2, 1)
                bars = plt.barh(industries, maturity, alpha=0.8, color='coral')
                plt.title('Top 10 Saturated Industries', fontsize=12, pad=20)
                plt.xlabel('Maturity Score')
                
                # Add value labels
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    plt.text(width, bar.get_y() + bar.get_height()/2,
                            f'{maturity[i]:.2f}',
                            ha='left', va='center', fontsize=10)
                
                # Add momentum and concentration
                plt.subplot(1, 2, 2)
                x = np.arange(len(industries))
                width = 0.35
                
                plt.bar(x - width/2, [m*100 for m in momentum], width, label='Momentum (%)', alpha=0.8, color='lightblue')
                plt.bar(x + width/2, [c*100 for c in concentration], width, label='Concentration (%)', alpha=0.8, color='lightgreen')
                plt.xticks(x, industries, rotation=45, ha='right')
                plt.title('Industry Metrics', fontsize=12, pad=20)
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'saturated_industries.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            # 3. Industry Growth Matrix
            plt.figure(figsize=(12, 8))
            
            if self.emerging_industries:  # Only create plot if we have data
                # Create scatter plot of funding vs employee growth
                funding_growth = [x['funding_growth'] for x in self.emerging_industries[:10]]
                employee_growth = [x['employee_growth'] for x in self.emerging_industries[:10]]
                industries = [x['industry'] for x in self.emerging_industries[:10]]
                market_shares = [x['market_share'] for x in self.emerging_industries[:10]]
                
                # Scale market share for visibility in scatter plot
                sizes = [ms * 100 for ms in market_shares]  # Adjust multiplier for better visibility
                
                scatter = plt.scatter(funding_growth, employee_growth, s=sizes, alpha=0.6, c='skyblue')
                
                # Add labels for top industries
                for i, industry in enumerate(industries[:5]):
                    plt.annotate(industry, (funding_growth[i], employee_growth[i]))
                
                plt.title('Industry Growth Matrix', fontsize=12, pad=20)
                plt.xlabel('Funding Growth (%)')
                plt.ylabel('Employee Growth (%)')
                plt.grid(True, alpha=0.3)
                
                # Add size legend
                handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6,
                                                        num=4, func=lambda s: s/100)
                plt.legend(handles, labels, loc="upper right", title="Market Share (%)")
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'industry_matrix.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            self.logger.info("Trend visualizations created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
            plt.close('all')

    def _load_fundraiser(self):
        """Load fundraiser data with error handling"""
        try:
            file_path = os.path.join(self.data_dir, 'fundraisestartup50.json')
            with open(file_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data.get('companies', []))
            self.logger.info(f"Loaded {len(df)} records from fundraiser data")
            return df
        except Exception as e:
            self.logger.error(f"Error loading fundraiser data: {str(e)}")
            return pd.DataFrame()

    def _load_growthlist(self):
        """Load growthlist data with error handling"""
        try:
            file_path = os.path.join(self.data_dir, 'growthlistscrapper.json')
            with open(file_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            self.logger.info(f"Loaded {len(df)} records from growthlist data")
            return df
        except Exception as e:
            self.logger.error(f"Error loading growthlist data: {str(e)}")
            return pd.DataFrame()

    def _load_topstartup(self):
        """Load topstartup data with proper date handling"""
        try:
            file_path = os.path.join(self.data_dir, 'topstartupio50.json')
            with open(file_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            
            # Convert founding_year to datetime - handle it properly
            df['funding_date'] = pd.to_datetime(df['founding_year'].astype(str), format='%Y')
            
            # Map columns to standard names
            df = df.rename(columns={
                'name': 'company_name',
                'category': 'industry'
            })
            
            # Extract funding amount and stage from funding string
            def extract_funding_info(funding_str):
                if pd.isna(funding_str):
                    return pd.NA, pd.NA
                
                # Extract amount
                amount_match = re.search(r'\$(\d+(?:\.\d+)?[BMK]?)', str(funding_str))
                amount = amount_match.group(1) if amount_match else pd.NA
                
                # Extract stage
                stage_patterns = {
                    'Series A': r'Series A',
                    'Series B': r'Series B',
                    'Series C': r'Series C',
                    'Series D': r'Series D',
                    'Series E': r'Series E',
                    'Seed': r'Seed',
                    'Pre-Seed': r'Pre-Seed',
                    'IPO': r'IPO|Post-IPO'
                }
                
                stage = 'Unknown'
                for s, pattern in stage_patterns.items():
                    if re.search(pattern, str(funding_str), re.IGNORECASE):
                        stage = s
                        break
                
                return amount, stage
            
            # Apply funding extraction
            df[['funding_amount', 'funding_stage']] = df.apply(
                lambda x: pd.Series(extract_funding_info(x['funding'])),
                axis=1
            )
            
            # Clean employee count - extract first number from range
            df['employees'] = df['employees'].str.extract(r'(\d+)').astype(float)
            
            self.logger.info(f"Loaded {len(df)} records from topstartup data")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading topstartup data: {str(e)}")
            return pd.DataFrame()

    def _clean_data(self, df):
        """Clean and standardize the data"""
        try:
            # Make a copy to avoid modifying the original
            data = df.copy()
            
            # Ensure required columns exist
            required_cols = ['company_name', 'funding_date', 'funding_amount', 'funding_stage', 'industry']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                self.logger.warning(f"Missing columns: {missing_cols}")
                for col in missing_cols:
                    data[col] = np.nan
            
            # Clean funding amounts
            if 'funding_amount' in data.columns:
                # Convert to numeric, handling various formats
                data['funding_amount'] = data['funding_amount'].apply(parse_funding)
                
                # Calculate industry medians for imputation
                industry_medians = data.groupby('industry')['funding_amount'].transform('median')
                overall_median = data['funding_amount'].median()
                
                # Fill missing values with industry median, then overall median
                data['funding_amount'] = data['funding_amount'].fillna(industry_medians)
                data['funding_amount'] = data['funding_amount'].fillna(overall_median)
                
                # Log warning about imputed values
                imputed_count = data['funding_amount'].isna().sum()
                if imputed_count > 0:
                    self.logger.warning(f"Imputed {imputed_count} missing funding amounts")
            
            # Clean dates
            if 'funding_date' in data.columns:
                data['funding_date'] = pd.to_datetime(data['funding_date'], errors='coerce')
                invalid_dates = data['funding_date'].isna().sum()
                if invalid_dates > 0:
                    self.logger.warning(f"Removed {invalid_dates} rows with invalid dates")
                data = data.dropna(subset=['funding_date'])
            
            # Standardize funding stages
            if 'funding_stage' in data.columns:
                stage_mapping = {
                    'series a': 'Series A',
                    'series b': 'Series B',
                    'series c': 'Series C',
                    'series d': 'Series D',
                    'series e': 'Series E',
                    'series f': 'Series F',
                    'seed': 'Seed',
                    'angel': 'Angel',
                    'pre-seed': 'Pre-Seed',
                    'ipo': 'IPO',
                    'venture': 'Venture',
                    'private equity': 'Private Equity'
                }
                
                data['funding_stage'] = data['funding_stage'].fillna('Unknown')
                data['funding_stage'] = data['funding_stage'].str.lower().map(
                    lambda x: next((v for k, v in stage_mapping.items() if str(x).lower().startswith(k)), 'Other')
                )
            
            # Clean industry categories
            if 'industry' in data.columns:
                # Handle multiple industries
                data['industry'] = data['industry'].fillna('Unknown')
                # Split and take primary industry
                data['industry'] = data['industry'].str.split(',').str[0]
                data['industry'] = data['industry'].str.strip()
                
                # Standardize common variations
                industry_mapping = {
                    'ai': 'Artificial Intelligence',
                    'artificial intelligence': 'Artificial Intelligence',
                    'machine learning': 'Artificial Intelligence',
                    'fintech': 'Financial Technology',
                    'financial technology': 'Financial Technology',
                    'biotech': 'Biotechnology',
                    'biotechnology': 'Biotechnology',
                    'saas': 'Software',
                    'software': 'Software'
                }
                
                data['industry'] = data['industry'].str.lower().map(
                    lambda x: next((v for k, v in industry_mapping.items() if str(x).lower().startswith(k)), x)
                )
                data['industry'] = data['industry'].str.title()
            
            # Sort by company and date
            data = data.sort_values(['company_name', 'funding_date'])
            
            # Remove duplicates
            dups = data.duplicated(subset=['company_name', 'funding_date'], keep='first').sum()
            if dups > 0:
                self.logger.info(f"Removed {dups} duplicate entries")
                data = data.drop_duplicates(subset=['company_name', 'funding_date'], keep='first')
            
            # Log summary statistics
            self.logger.info(f"Cleaned data summary:")
            self.logger.info(f"Total companies: {data['company_name'].nunique()}")
            self.logger.info(f"Total funding rounds: {len(data)}")
            self.logger.info(f"Date range: {data['funding_date'].min():%Y-%m-%d} to {data['funding_date'].max():%Y-%m-%d}")
            self.logger.info(f"Total funding amount: ${data['funding_amount'].sum():,.2f}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error in data cleaning: {str(e)}")
            raise

    def _calculate_survival_metrics(self, df):
        """Calculate funding survival metrics"""
        try:
            data = df.copy()
            
            # Calculate next funding date for each company
            data['next_funding_date'] = data.groupby('company_name')['funding_date'].shift(-1)
            
            # Calculate time between funding rounds
            data['duration'] = (data['next_funding_date'] - data['funding_date']).dt.days
            
            # Mark if company received next funding (event=1) or not (event=0)
            data['event'] = data['next_funding_date'].notna().astype(int)
            
            # For companies without next funding, calculate duration to end of observation period
            max_date = data['funding_date'].max()
            mask = data['event'] == 0
            data.loc[mask, 'duration'] = (max_date - data.loc[mask, 'funding_date']).dt.days
            
            # Remove invalid durations
            data = data[data['duration'] >= 0]
            
            # Calculate additional metrics
            data['funding_growth'] = data.groupby('company_name')['funding_amount'].pct_change()
            data['days_since_first'] = data.groupby('company_name')['funding_date'].transform(
                lambda x: (x - x.min()).dt.days
            )
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating survival metrics: {str(e)}")
            raise

    def validate_and_calibrate(self, df_expanded, industry_metrics, emerging_industries, saturated_industries):
        """Validate trend analysis and create calibration plots"""
        try:
            # Split data into training and validation sets
            cutoff_date = df_expanded['funding_date'].max() - pd.DateOffset(months=3)
            train_data = df_expanded[df_expanded['funding_date'] < cutoff_date]
            val_data = df_expanded[df_expanded['funding_date'] >= cutoff_date]
            
            # Calculate validation metrics
            validation_metrics = {
                'total_samples': len(df_expanded),
                'train_samples': len(train_data),
                'val_samples': len(val_data),
                'industry_coverage': len(industry_metrics) / len(df_expanded['industry'].unique()),
                'metrics': {}
            }
            
            # Validate growth predictions
            predicted_growth = {}
            actual_growth = {}
            
            for industry in industry_metrics:
                # Get predicted growth from training data
                train_industry = train_data[train_data['industry'] == industry]
                if len(train_industry) < 5:
                    continue
                
                # Calculate predicted growth based on training data
                train_recent = train_industry[train_industry['funding_date'] >= (cutoff_date - pd.DateOffset(months=3))]
                train_prev = train_industry[train_industry['funding_date'] < (cutoff_date - pd.DateOffset(months=3))]
                
                if len(train_recent) == 0 or len(train_prev) == 0:
                    continue
                
                predicted_growth[industry] = {
                    'funding': (train_recent['funding_amount'].mean() / train_prev['funding_amount'].mean() - 1),
                    'deals': (len(train_recent) / len(train_prev) - 1),
                    'employees': (train_recent['employees'].mean() / train_prev['employees'].mean() - 1) if 'employees' in train_industry.columns else 0
                }
                
                # Calculate actual growth from validation data
                val_industry = val_data[val_data['industry'] == industry]
                if len(val_industry) == 0:
                    continue
                    
                actual_growth[industry] = {
                    'funding': (val_industry['funding_amount'].mean() / train_recent['funding_amount'].mean() - 1),
                    'deals': (len(val_industry) / len(train_recent) - 1),
                    'employees': (val_industry['employees'].mean() / train_recent['employees'].mean() - 1) if 'employees' in val_industry.columns else 0
                }
            
            # Calculate prediction accuracy
            funding_rmse = np.sqrt(np.mean([
                (predicted_growth[ind]['funding'] - actual_growth[ind]['funding'])**2
                for ind in predicted_growth if ind in actual_growth
            ]))
            
            deals_rmse = np.sqrt(np.mean([
                (predicted_growth[ind]['deals'] - actual_growth[ind]['deals'])**2
                for ind in predicted_growth if ind in actual_growth
            ]))
            
            employees_rmse = np.sqrt(np.mean([
                (predicted_growth[ind]['employees'] - actual_growth[ind]['employees'])**2
                for ind in predicted_growth if ind in actual_growth
            ]))
            
            # Calculate emerging industry precision
            emerging_correct = sum(
                1 for ind in emerging_industries 
                if ind['industry'] in actual_growth and 
                actual_growth[ind['industry']]['funding'] > 0 and
                actual_growth[ind['industry']]['deals'] > 0
            )
            emerging_precision = emerging_correct / len(emerging_industries) if emerging_industries else 0
            
            # Calculate saturated industry precision
            saturated_correct = sum(
                1 for ind in saturated_industries
                if ind['industry'] in actual_growth and
                actual_growth[ind['industry']]['funding'] < 0
            )
            saturated_precision = saturated_correct / len(saturated_industries) if saturated_industries else 0
            
            validation_metrics['metrics'] = {
                'funding_growth_rmse': funding_rmse,
                'deals_growth_rmse': deals_rmse,
                'employees_growth_rmse': employees_rmse,
                'emerging_precision': emerging_precision,
                'saturated_precision': saturated_precision
            }
            
            # Create calibration plots
            self._create_calibration_plots(predicted_growth, actual_growth)
            
            # Log validation results
            self.logger.info("\nValidation Metrics:")
            self.logger.info(f"Samples: Total={validation_metrics['total_samples']}, Train={validation_metrics['train_samples']}, Val={validation_metrics['val_samples']}")
            self.logger.info(f"Industry Coverage: {validation_metrics['industry_coverage']*100:.1f}%")
            self.logger.info(f"Funding Growth RMSE: {funding_rmse:.3f}")
            self.logger.info(f"Deals Growth RMSE: {deals_rmse:.3f}")
            self.logger.info(f"Employees Growth RMSE: {employees_rmse:.3f}")
            self.logger.info(f"Emerging Industries Precision: {emerging_precision*100:.1f}%")
            self.logger.info(f"Saturated Industries Precision: {saturated_precision*100:.1f}%")
            
            return validation_metrics
            
        except Exception as e:
            self.logger.error(f"Error in validation: {str(e)}")
            return None
            
    def _create_calibration_plots(self, predicted_growth, actual_growth):
        """Create calibration plots for growth predictions"""
        try:
            plt.figure(figsize=(15, 5))
            
            # Funding growth calibration
            plt.subplot(131)
            x = [predicted_growth[ind]['funding'] for ind in predicted_growth if ind in actual_growth]
            y = [actual_growth[ind]['funding'] for ind in predicted_growth if ind in actual_growth]
            plt.scatter(x, y, alpha=0.5)
            plt.plot([-1, 2], [-1, 2], 'r--')  # Perfect calibration line
            plt.xlabel('Predicted Funding Growth')
            plt.ylabel('Actual Funding Growth')
            plt.title('Funding Growth Calibration')
            plt.grid(True, alpha=0.3)
            
            # Deals growth calibration
            plt.subplot(132)
            x = [predicted_growth[ind]['deals'] for ind in predicted_growth if ind in actual_growth]
            y = [actual_growth[ind]['deals'] for ind in predicted_growth if ind in actual_growth]
            plt.scatter(x, y, alpha=0.5)
            plt.plot([-1, 2], [-1, 2], 'r--')
            plt.xlabel('Predicted Deals Growth')
            plt.ylabel('Actual Deals Growth')
            plt.title('Deals Growth Calibration')
            plt.grid(True, alpha=0.3)
            
            # Employee growth calibration
            plt.subplot(133)
            x = [predicted_growth[ind]['employees'] for ind in predicted_growth if ind in actual_growth]
            y = [actual_growth[ind]['employees'] for ind in predicted_growth if ind in actual_growth]
            plt.scatter(x, y, alpha=0.5)
            plt.plot([-1, 2], [-1, 2], 'r--')
            plt.xlabel('Predicted Employee Growth')
            plt.ylabel('Actual Employee Growth')
            plt.title('Employee Growth Calibration')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'calibration_plots.png'))
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating calibration plots: {str(e)}")

    def calculate_growth_score(self, industry_data):
        """Calculate a comprehensive growth score based on multiple metrics and timeframes."""
        if len(industry_data) < 4:  # Need at least 4 quarters of data
            return 0.0
        
        # Sort data by date
        industry_data = industry_data.sort_values('date')
        
        # Define timeframes (in quarters)
        timeframes = {
            'recent': 4,
            'mid': 8,
            'long': 12
        }
        
        # Calculate weights for different timeframes
        weights = {
            'recent': 0.5,
            'mid': 0.3,
            'long': 0.2
        }
        
        # Metric weights
        metric_weights = {
            'funding': 0.35,
            'deals': 0.25,
            'employees': 0.25,
            'deal_size': 0.15
        }
        
        total_score = 0
        
        for timeframe_name, quarters in timeframes.items():
            if len(industry_data) >= quarters:
                # Calculate growth rates for each metric
                funding_growth = (industry_data['funding_amount'].iloc[-1] / 
                                max(industry_data['funding_amount'].iloc[-quarters], 1)) - 1
                
                deals_growth = (industry_data['num_deals'].iloc[-1] / 
                              max(industry_data['num_deals'].iloc[-quarters], 1)) - 1
                
                employee_growth = (industry_data['employee_count'].iloc[-1] / 
                                 max(industry_data['employee_count'].iloc[-quarters], 1)) - 1
                
                avg_deal_size_growth = (
                    (industry_data['funding_amount'].iloc[-1] / max(industry_data['num_deals'].iloc[-1], 1)) /
                    (industry_data['funding_amount'].iloc[-quarters] / max(industry_data['num_deals'].iloc[-quarters], 1)) - 1
                )
                
                # Normalize growth rates (clip to reasonable ranges)
                funding_growth = np.clip(funding_growth, -1, 5)
                deals_growth = np.clip(deals_growth, -1, 5)
                employee_growth = np.clip(employee_growth, -1, 5)
                avg_deal_size_growth = np.clip(avg_deal_size_growth, -1, 5)
                
                # Calculate weighted score for this timeframe
                timeframe_score = (
                    funding_growth * metric_weights['funding'] +
                    deals_growth * metric_weights['deals'] +
                    employee_growth * metric_weights['employees'] +
                    avg_deal_size_growth * metric_weights['deal_size']
                )
                
                total_score += timeframe_score * weights[timeframe_name]
        
        # Normalize final score to 0-1 range
        normalized_score = (total_score + 1) / 6  # Assuming max growth of 5x
        return np.clip(normalized_score, 0, 1)

    def identify_emerging_industries(self, df):
        """Identify emerging industries based on growth patterns and other criteria."""
        emerging_industries = []
        
        for industry in df['industry'].unique():
            industry_data = df[df['industry'] == industry].copy()
            
            if len(industry_data) < 4:  # Skip if less than 4 quarters of data
                continue
            
            # Calculate key metrics
            growth_score = self.calculate_growth_score(industry_data)
            company_count = len(industry_data['company_name'].unique())
            avg_company_age = industry_data['company_age'].mean()
            market_share = (industry_data['funding_amount'].sum() / 
                           df['funding_amount'].sum())
            
            # Criteria for emerging industries
            is_emerging = (
                growth_score > 0.2 and  # Lowered threshold
                3 <= company_count <= 25 and  # Slightly increased upper limit
                avg_company_age < 7 and  # Increased age limit
                market_share < 0.08  # Increased market share limit
            )
            
            if is_emerging:
                growth_rate = ((industry_data['funding_amount'].iloc[-1] / 
                              industry_data['funding_amount'].iloc[0]) - 1) * 100
                employee_growth = ((industry_data['employee_count'].iloc[-1] / 
                                 industry_data['employee_count'].iloc[0]) - 1) * 100
                
                emerging_industries.append({
                    'industry': industry,
                    'growth_score': growth_score,
                    'growth_rate': growth_rate,
                    'market_share': market_share * 100,
                    'company_count': company_count,
                    'employee_growth': employee_growth
                })
        
        # Sort by growth score
        emerging_industries.sort(key=lambda x: x['growth_score'], reverse=True)
        return emerging_industries

def main():
    """Run the complete industry trend analysis pipeline"""
    try:
        # Initialize analyzer
        analyzer = IndustryTrendAnalyzer()
        
        # Load and process data
        df = analyzer.load_and_merge_data()
        
        # Perform analysis
        industry_metrics, emerging_industries, saturated_industries = analyzer.analyze_industry_trends(df)
        
        # Analyze employee migration
        migration = analyzer.analyze_employee_migration(df)
        
        # Generate report and visualizations
        report = analyzer.generate_report()
        analyzer.visualize_trends()
        
        print("\nAnalysis completed successfully!")
        print("\nTop 5 Emerging Industries:")
        for industry in emerging_industries[:5]:
            print(f"- {industry['industry']}:")
            print(f"  - Funding Growth: {industry['funding_growth']:.1f}%")
            print(f"  - Employee Growth: {industry['employee_growth']:.1f}%")
            print(f"  - Deal Growth: {industry['deal_growth']:.1f}%")
            print(f"  - Market Share: {industry['market_share']:.2f}%")
            print(f"  - Companies: {industry['companies']}")
            print(f"  - Avg Quarterly Funding: ${industry['avg_quarterly_funding']:,.2f}")
            print(f"  - Total Funding: ${industry['total_funding']:,.2f}")
            
        print("\nTop 5 Saturated Industries:")
        for industry in saturated_industries[:5]:
            print(f"- {industry['industry']}:")
            print(f"  - Maturity Score: {industry['maturity']:.2f}")
            print(f"  - Market Share: {industry['market_share']:.2f}%")
            print(f"  - Concentration: {industry['concentration']:.2f}")
            print(f"  - Funding Growth: {industry['funding_growth']:.1f}%")
            print(f"  - Momentum: {industry['momentum']*100:.1f}%")
            print(f"  - Companies: {industry['companies']}")
            print(f"  - Total Funding: ${industry['total_funding']:,.2f}")
            
    except Exception as e:
        print(f"Error in analysis pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
