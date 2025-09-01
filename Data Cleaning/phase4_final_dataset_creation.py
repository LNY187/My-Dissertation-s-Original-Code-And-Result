#!/usr/bin/env python3
"""
Phase 4: Final Dataset Creation Pipeline
=======================================

This script consolidates all final dataset creation operations:
1. Load crime and socioeconomic datasets
2. Merge datasets on province-year keys
3. Calculate crime rates and additional metrics
4. Perform data quality validation
5. Generate analysis-ready panel dataset
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

class FinalDatasetCreator:
    """Main class for creating the final analysis-ready dataset."""
    
    def __init__(self):
        self.crime_df = None
        self.socio_df = None
        self.final_df = None
        
    def load_datasets(self, crime_file='output/aggregated_crime_data.csv', 
                     socio_file='output/merged_socioeconomic_data.csv'):
        """
        Load crime and socioeconomic datasets.
        
        Args:
            crime_file (str): Path to aggregated crime data
            socio_file (str): Path to merged socioeconomic data
            
        Returns:
            bool: Success status
        """
        print("=== Step 1: Loading Datasets ===")
        
        # Check if required files exist
        if not os.path.exists(crime_file):
            print(f"Crime data file not found: {crime_file}")
            print("Please run Phase 2 (Crime Processing) first.")
            return False
        
        if not os.path.exists(socio_file):
            print(f"Socioeconomic data file not found: {socio_file}")
            print("Please run Phase 3 (Socioeconomic Processing) first.")
            return False
        
        try:
            # Load crime data
            print(f"Loading crime data from {crime_file}...")
            self.crime_df = pd.read_csv(crime_file, encoding='utf-8-sig')
            print(f"   Crime data shape: {self.crime_df.shape}")
            
            # Load socioeconomic data
            print(f"Loading socioeconomic data from {socio_file}...")
            self.socio_df = pd.read_csv(socio_file, encoding='utf-8-sig')
            print(f"   Socioeconomic data shape: {self.socio_df.shape}")
            
            # Basic validation
            if self.crime_df.empty or self.socio_df.empty:
                print("One or both datasets are empty!")
                return False
            
            print("Datasets loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading datasets: {e}")
            return False
    
    def validate_datasets(self):
        """
        Validate dataset structure and key columns.
        
        Returns:
            bool: Validation success status
        """
        print("\n=== Step 2: Validating Dataset Structure ===")
        
        if self.crime_df is None or self.socio_df is None:
            print("Datasets not loaded!")
            return False
        
        # Check for required columns in crime data
        crime_required_cols = ['Province', 'Year']
        crime_missing = [col for col in crime_required_cols if col not in self.crime_df.columns]
        
        if crime_missing:
            print(f"Missing required columns in crime data: {crime_missing}")
            return False
        
        # Check for required columns in socioeconomic data
        socio_required_cols = ['Province', 'Year']
        socio_missing = [col for col in socio_required_cols if col not in self.socio_df.columns]
        
        if socio_missing:
            print(f"Missing required columns in socioeconomic data: {socio_missing}")
            return False
        
        # Check year coverage
        crime_years = set(self.crime_df['Year'].unique())
        socio_years = set(self.socio_df['Year'].unique())
        common_years = crime_years.intersection(socio_years)
        
        print(f"Year Coverage:")
        print(f"   Crime data years: {sorted(crime_years)}")
        print(f"   Socioeconomic data years: {sorted(socio_years)}")
        print(f"   Common years: {sorted(common_years)}")
        
        if not common_years:
            print("No common years between datasets!")
            return False
        
        # Check province coverage
        crime_provinces = set(self.crime_df['Province'].unique())
        socio_provinces = set(self.socio_df['Province'].unique())
        common_provinces = crime_provinces.intersection(socio_provinces)
        
        print(f"Province Coverage:")
        print(f"   Crime data provinces: {len(crime_provinces)}")
        print(f"   Socioeconomic data provinces: {len(socio_provinces)}")
        print(f"   Common provinces: {len(common_provinces)}")
        
        if len(common_provinces) < 10:  # Minimum threshold
            print("Warning: Very few common provinces between datasets!")
        
        print("Dataset validation completed")
        return True
    
    def merge_datasets(self):
        """
        Merge crime and socioeconomic datasets.
        
        Returns:
            bool: Merge success status
        """
        print("\n=== Step 3: Merging Datasets ===")
        
        if self.crime_df is None or self.socio_df is None:
            print("Datasets not loaded!")
            return False
        
        try:
            print("Merging crime and socioeconomic data...")
            
            # Merge on Province and Year using inner join to keep only matching records
            self.final_df = pd.merge(self.crime_df, self.socio_df, 
                                   on=['Province', 'Year'], how='inner')
            
            print(f"Merge completed!")
            print(f"   Crime data: {self.crime_df.shape}")
            print(f"   Socioeconomic data: {self.socio_df.shape}")
            print(f"   Final merged dataset: {self.final_df.shape}")
            
            if self.final_df.empty:
                print("Merged dataset is empty!")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error merging datasets: {e}")
            return False
    
    def calculate_crime_rates_and_metrics(self):
        """
        Calculate crime rates per 100,000 population and additional metrics.
        
        Returns:
            bool: Calculation success status
        """
        print("\n=== Step 4: Calculating Crime Rates and Additional Metrics ===")
        
        if self.final_df is None:
            print("No merged dataset available!")
            return False
        
        try:
            df = self.final_df.copy()
            
            # 1. Calculate crime rates per 100,000 population
            print("Calculating crime rates per 100,000 population...")
            
            crime_count_columns = [col for col in df.columns if col.endswith('_Crime_Count')]
            
            # Check if we have population data
            population_cols = [col for col in df.columns if 'Population' in col]
            if not population_cols:
                print("No population column found, cannot calculate crime rates")
            else:
                # Try to find the best population column
                pop_col = None
                for col in ['Total_Population', 'Population', 'total_population']:
                    if col in df.columns:
                        pop_col = col
                        break
                
                if pop_col:
                    print(f"   Using population column: {pop_col}")
                    
                    for crime_col in crime_count_columns:
                        if crime_col in df.columns:
                            rate_col = crime_col.replace('_Count', '_Rate')
                            # Population is typically in 10,000s, so multiply by 10,000 then divide by 100,000
                            df[rate_col] = (df[crime_col] / (df[pop_col] * 10000)) * 100000
                            print(f"   ✓ {rate_col} calculated")
                    
                    # Calculate total crime rate
                    if crime_count_columns:
                        df['Total_Crime_Count'] = df[crime_count_columns].sum(axis=1)
                        df['Total_Crime_Rate'] = (df['Total_Crime_Count'] / (df[pop_col] * 10000)) * 100000
                        print("   ✓ Total crime metrics calculated")
            
            # 2. Add time trend variable
            print("Adding time trend variable...")
            df['Time_Trend'] = df['Year'] - df['Year'].min()
            print("   ✓ Time trend variable added")
            
            # 3. Calculate log transformations for key economic variables
            print("Calculating log transformations...")
            economic_vars_for_log = []
            
            # Find suitable economic variables for log transformation
            for col in df.columns:
                if any(term in col for term in ['GDP', 'Income', 'Population']) and df[col].dtype in ['int64', 'float64']:
                    if (df[col] > 0).all():  # Ensure all values are positive
                        economic_vars_for_log.append(col)
            
            for var in economic_vars_for_log[:5]:  # Limit to top 5 to avoid clutter
                log_var = f'Log_{var}'
                df[log_var] = np.log(df[var] + 1)  # Add 1 to avoid log(0)
                print(f"   ✓ {log_var} calculated")
            
            # 4. Calculate additional economic indicators
            print("Calculating additional economic indicators...")
            
            # Economic development level (GDP per capita categories)
            if any('GDP_Per_Capita' in col for col in df.columns):
                gdp_per_capita_col = None
                for col in df.columns:
                    if 'GDP_Per_Capita' in col:
                        gdp_per_capita_col = col
                        break
                
                if gdp_per_capita_col:
                    gdp_median = df[gdp_per_capita_col].median()
                    df['Economic_Development_Level'] = df[gdp_per_capita_col].apply(
                        lambda x: 'High' if x > gdp_median else 'Low'
                    )
                    print("   ✓ Economic development level categorized")
            
            # 5. Calculate regional crime intensity
            print("Calculating regional crime metrics...")
            if 'Total_Crime_Rate' in df.columns:
                # Crime intensity categories
                crime_rate_median = df['Total_Crime_Rate'].median()
                df['Crime_Intensity'] = df['Total_Crime_Rate'].apply(
                    lambda x: 'High' if x > crime_rate_median else 'Low'
                )
                print("   ✓ Crime intensity categorized")
                
                # Year-over-year crime rate change
                df = df.sort_values(['Province', 'Year'])
                df['Crime_Rate_Change'] = df.groupby('Province')['Total_Crime_Rate'].pct_change() * 100
                print("   ✓ Crime rate change calculated")
            
            # 6. Add regional dummy variables (if needed for analysis)
            print("Adding regional classifications...")
            
            # Define major regions (can be customized based on analysis needs)
            eastern_provinces = ['北京市', '天津市', '河北省', '上海市', '江苏省', '浙江省', 
                               '福建省', '山东省', '广东省', '海南省']
            central_provinces = ['山西省', '安徽省', '江西省', '河南省', '湖北省', '湖南省']
            western_provinces = ['内蒙古自治区', '广西壮族自治区', '重庆市', '四川省', '贵州省', 
                               '云南省', '西藏自治区', '陕西省', '甘肃省', '青海省', '宁夏回族自治区', '新疆维吾尔自治区']
            northeastern_provinces = ['辽宁省', '吉林省', '黑龙江省']
            
            def classify_region(province):
                if province in eastern_provinces:
                    return 'Eastern'
                elif province in central_provinces:
                    return 'Central'
                elif province in western_provinces:
                    return 'Western'
                elif province in northeastern_provinces:
                    return 'Northeastern'
                else:
                    return 'Other'
            
            df['Region'] = df['Province'].apply(classify_region)
            print("   ✓ Regional classifications added")
            
            self.final_df = df
            print("Crime rates and additional metrics calculated successfully")
            return True
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return False
    
    def perform_data_quality_checks(self):
        """
        Perform comprehensive data quality validation.
        
        Returns:
            bool: Data quality check success status
        """
        print("\n=== Step 5: Data Quality Validation ===")
        
        if self.final_df is None:
            print("No final dataset available!")
            return False
        
        df = self.final_df
        
        # 1. Check for missing values
        print("Checking for missing values...")
        missing_stats = df.isnull().sum()
        variables_with_missing = missing_stats[missing_stats > 0]
        
        if not variables_with_missing.empty:
            print(f"   Variables with missing values:")
            for var, count in variables_with_missing.items():
                percentage = (count / len(df)) * 100
                print(f"   - {var}: {count} ({percentage:.1f}%)")
        else:
            print("   ✓ No missing values found")
        
        # 2. Check for duplicate records
        print("Checking for duplicate records...")
        duplicates = df.duplicated(subset=['Province', 'Year']).sum()
        if duplicates > 0:
            print(f"   Found {duplicates} duplicate province-year combinations")
        else:
            print("   ✓ No duplicate records found")
        
        # 3. Check data ranges and outliers
        print("Checking for data anomalies...")
        
        # Check crime rates for negative values
        crime_rate_cols = [col for col in df.columns if col.endswith('_Rate')]
        for col in crime_rate_cols:
            negative_values = (df[col] < 0).sum()
            if negative_values > 0:
                print(f"   {col} has {negative_values} negative values")
        
        # Check for extremely high crime rates (potential data errors)
        if 'Total_Crime_Rate' in df.columns:
            high_crime_threshold = df['Total_Crime_Rate'].quantile(0.95)
            high_crime_cases = (df['Total_Crime_Rate'] > high_crime_threshold * 2).sum()
            if high_crime_cases > 0:
                print(f"   {high_crime_cases} cases with extremely high crime rates")
        
        # 4. Check temporal consistency
        print("Checking temporal consistency...")
        year_coverage = df.groupby('Province')['Year'].agg(['min', 'max', 'count'])
        incomplete_coverage = year_coverage[year_coverage['count'] < 7]  # Less than 7 years
        
        if not incomplete_coverage.empty:
            print(f"   {len(incomplete_coverage)} provinces with incomplete year coverage")
        else:
            print("   ✓ All provinces have complete temporal coverage")
        
        print("Data quality validation completed")
        return True
    
    def generate_summary_statistics(self):
        """
        Generate comprehensive summary statistics for the final dataset.
        """
        print("\n=== Step 6: Generating Summary Statistics ===")
        
        if self.final_df is None:
            print("No final dataset available!")
            return False
        
        df = self.final_df
        
        # Basic dataset information
        print("Dataset Summary:")
        print(f"   • Total observations: {len(df):,}")
        print(f"   • Number of provinces: {df['Province'].nunique()}")
        print(f"   • Years covered: {df['Year'].min()} - {df['Year'].max()}")
        print(f"   • Total variables: {len(df.columns)}")
        
        # Crime statistics
        print("\nCrime Statistics:")
        crime_rate_vars = [col for col in df.columns if col.endswith('_Rate')]
        if crime_rate_vars:
            for var in crime_rate_vars:
                mean_rate = df[var].mean()
                std_rate = df[var].std()
                print(f"   • {var}: Mean={mean_rate:.2f}, Std={std_rate:.2f}")
        
        # Economic indicators
        print("\nEconomic Indicators:")
        economic_vars = [col for col in df.columns if any(term in col for term in ['GDP', 'Income'])]
        for var in economic_vars[:5]:  # Show top 5
            if df[var].dtype in ['int64', 'float64']:
                mean_val = df[var].mean()
                std_val = df[var].std()
                print(f"   • {var}: Mean={mean_val:.2f}, Std={std_val:.2f}")
        
        # Regional distribution
        if 'Region' in df.columns:
            print("\nRegional Distribution:")
            region_counts = df['Region'].value_counts()
            for region, count in region_counts.items():
                percentage = (count / len(df)) * 100
                print(f"   • {region}: {count} ({percentage:.1f}%)")
        
        # Temporal distribution
        print("\nTemporal Distribution:")
        year_counts = df['Year'].value_counts().sort_index()
        for year, count in year_counts.items():
            print(f"   • {year}: {count} observations")
        
        print("Summary statistics generated")
        return True
    
    def save_final_dataset(self, output_file='final_panel_dataset.csv'):
        """
        Save the final analysis-ready dataset.
        
        Args:
            output_file (str): Output filename
            
        Returns:
            bool: Save success status
        """
        print(f"\n=== Step 7: Saving Final Dataset ===")
        
        if self.final_df is None:
            print("No final dataset to save!")
            return False
        
        try:
            # Create output directory
            os.makedirs('output', exist_ok=True)
            output_path = f'output/{output_file}'
            
            # Save the final dataset
            self.final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            print(f"Final dataset saved to: {output_path}")
            print(f"   • Shape: {self.final_df.shape}")
            print(f"   • Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
            
            # Create a data dictionary
            dict_path = 'output/data_dictionary.txt'
            with open(dict_path, 'w', encoding='utf-8') as f:
                f.write("Data Dictionary for Final Panel Dataset\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Dataset created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total observations: {len(self.final_df):,}\n")
                f.write(f"Total variables: {len(self.final_df.columns)}\n\n")
                
                f.write("Variables:\n")
                for i, col in enumerate(self.final_df.columns, 1):
                    f.write(f"{i:3d}. {col}\n")
            
            print(f"Data dictionary saved to: {dict_path}")
            
            return True
            
        except Exception as e:
            print(f"Error saving final dataset: {e}")
            return False


def main():
    """Main execution function for Phase 4: Final Dataset Creation."""
    print("Phase 4: Final Dataset Creation Pipeline")
    print("=" * 50)
    print(f"Processing started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    creator = FinalDatasetCreator()
    
    # Step 1: Load datasets
    if not creator.load_datasets():
        print("Failed at Step 1: Load datasets")
        return False
    
    # Step 2: Validate datasets
    if not creator.validate_datasets():
        print("Failed at Step 2: Validate datasets")
        return False
    
    # Step 3: Merge datasets
    if not creator.merge_datasets():
        print("Failed at Step 3: Merge datasets")
        return False
    
    # Step 4: Calculate crime rates and metrics
    if not creator.calculate_crime_rates_and_metrics():
        print("Failed at Step 4: Calculate metrics")
        return False
    
    # Step 5: Perform data quality checks
    if not creator.perform_data_quality_checks():
        print("Failed at Step 5: Data quality checks")
        return False
    
    # Step 6: Generate summary statistics
    if not creator.generate_summary_statistics():
        print("Failed at Step 6: Generate summary statistics")
        return False
    
    # Step 7: Save final dataset
    if not creator.save_final_dataset():
        print("Failed at Step 7: Save final dataset")
        return False
    
    print(f"\nPhase 4: Final Dataset Creation completed successfully!")
    print(f"   Output: output/final_panel_dataset.csv")
    print(f"   Output: output/data_dictionary.txt")
    print(f"   Processing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
