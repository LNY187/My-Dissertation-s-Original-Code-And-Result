#!/usr/bin/env python3
"""
Phase 3: Socioeconomic Data Processing Pipeline
==============================================

This script consolidates all socioeconomic data processing operations:
1. Load and standardize multiple socioeconomic CSV files
2. Convert wide-format data to long-format (province-year structure)
3. Handle special data formats and encoding issues
4. Calculate derived variables and indicators
5. Merge all indicators into comprehensive dataset
"""

import pandas as pd
import glob
import re
import json
import os
import sys
from functools import reduce

class SocioeconomicProcessor:
    """Main class for socioeconomic data processing."""
    
    def __init__(self):
        self.processed_dataframes = []
        self.merged_data = None
        
        # Define comprehensive file mapping
        self.files_to_process = {
            # Basic economic indicators
            'data/2013-2019 地区生产总值(亿元).csv': 'GDP_100M_Yuan',
            'data/人均地区生产总值.csv': 'GDP_Per_Capita',
            'data/地区生产总值指数（上年=100）.csv': 'GDP_Index',
            
            # Industry value added
            'data/第一产业增加值.csv': 'Primary_Industry_Value_Added',
            'data/第二产业增加值.csv': 'Secondary_Industry_Value_Added', 
            'data/第三产业增加值.csv': 'Tertiary_Industry_Value_Added',
            'data/农林牧渔业增加值.csv': 'Agriculture_Forestry_Livestock_Fishery_Value_Added',
            'data/工业增加值（亿元）.csv': 'Industrial_Value_Added',
            'data/建筑业增加值（亿元）.csv': 'Construction_Value_Added',
            'data/批发和零售业增加值（亿元）.csv': 'Wholesale_Retail_Value_Added',
            'data/房地产业增加值（亿元）.csv': 'Real_Estate_Value_Added',
            'data/住宿和餐饮增加值（亿元）.csv': 'Accommodation_Catering_Value_Added',
            'data/其他行业增加值（亿元）.csv': 'Other_Industries_Value_Added',
            
            # Population indicators
            'data/地区及主要城市人口 2013-2019.csv': 'Total_Population',
            'data/2013-2019 性别比（女=100）.csv': 'Gender_Ratio_Female_100',
            
            # Income indicators
            'data/全体居民人均可支配收入.csv': 'All_Residents_Per_Capita_Disposable_Income',
            'data/城镇居民人均可支配收入.csv': 'Urban_Per_Capita_Disposable_Income',
            'data/农村居民人均可支配收入.csv': 'Rural_Per_Capita_Disposable_Income',
            
            # Consumption indicators
            'data/全体居民人均消费支出（元）.csv': 'All_Residents_Per_Capita_Consumption_Expenditure',
            'data/城镇居民人均消费支出（元）.csv': 'Urban_Per_Capita_Consumption_Expenditure',
            'data/农村居民人均消费支出（元）.csv': 'Rural_Per_Capita_Consumption_Expenditure',
            'data/居民消费（亿元）.csv': 'Resident_Consumption_100M_Yuan',
            'data/城镇居民消费（亿元）.csv': 'Urban_Resident_Consumption_100M_Yuan',
            'data/农村居民消费（亿元）.csv': 'Rural_Resident_Consumption_100M_Yuan',
            'data/居民消费水平（元）.csv': 'Resident_Consumption_Level_Yuan',
            'data/城镇居民消费水平（元）.csv': 'Urban_Resident_Consumption_Level_Yuan',
            'data/农村居民消费水平（元）.csv': 'Rural_Resident_Consumption_Level_Yuan',
            
            # Employment indicators
            'data/城镇登记失业人数（万人）.csv': 'Urban_Registered_Unemployment_10K_People',
            'data/城镇登记失业率.csv': 'Urban_Registered_Unemployment_Rate',
            
            # Price indicators
            'data/居民消费价格指数.csv': 'Consumer_Price_Index',
            'data/居民消费水平指数（上年=100）.csv': 'Resident_Consumption_Level_Index',
            'data/城镇居民消费水平指数（上年=100）.csv': 'Urban_Consumption_Level_Index',
            'data/农村居民消费水平指数（上年=100）.csv': 'Rural_Consumption_Level_Index',
            
            # Government finance
            'data/地方财政预算收入（亿元）.csv': 'Local_Government_Budget_Revenue',
            'data/地方财政一般预算支出（亿元）.csv': 'Local_Government_Budget_Expenditure',
            
            # Production method GDP
            'data/支出法生产总值（亿元）.csv': 'Expenditure_Method_GDP',
            'data/收入法生产总值（亿元）.csv': 'Income_Method_GDP',
            
            # GDP components
            'data/最终消费（亿元）.csv': 'Final_Consumption',
            'data/固定资本形成总额（亿元）.csv': 'Gross_Fixed_Capital_Formation',
            'data/最终消费率（%）.csv': 'Final_Consumption_Rate',
            'data/政府消费（亿元）.csv': 'Government_Consumption',
            'data/居民消费（亿元）.csv': 'Resident_Consumption',
            
            # Other economic indicators
            'data/劳动者报酬（亿元）.csv': 'Labor_Compensation',
            'data/固定资产折旧（亿元）.csv': 'Fixed_Asset_Depreciation',
            'data/生产税净额（亿元）.csv': 'Net_Production_Taxes',
            'data/存货增加（亿元）.csv': 'Inventory_Increase',
        }
    
    def process_socioeconomic_data(self, file_path, value_name):
        """
        Read and process a single socioeconomic CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            value_name (str): Name for the value column
            
        Returns:
            DataFrame: Processed data in long format
        """
        try:
            # Special handling for area data (static across years)
            if 'provincial_area.csv' in file_path:
                df = pd.read_csv(file_path, encoding='utf-8-sig')
                # Create records for each year (2013-2019)
                years = range(2013, 2020)
                expanded_data = []
                for _, row in df.iterrows():
                    for year in years:
                        expanded_data.append({
                            'Province': row['Province'],
                            'Year': year,
                            value_name: row['Area_km2']
                        })
                return pd.DataFrame(expanded_data)
            
            # Try multiple encodings for Chinese data files
            encodings = ['gbk', 'utf-8-sig', 'gb18030', 'utf-8']
            df = None
            
            for encoding in encodings:
                try:
                    # Read the CSV file, skipping the first 3 rows of metadata
                    df = pd.read_csv(file_path, encoding=encoding, skiprows=3)
                    break
                except (UnicodeDecodeError, FileNotFoundError):
                    continue
            
            if df is None:
                print(f"Could not read {file_path} with any encoding")
                return None
            
            # Rename the first column to 'Province'
            df.rename(columns={df.columns[0]: 'Province'}, inplace=True)
            
            # Clean province names (remove extra characters)
            df['Province'] = df['Province'].astype(str).str.strip()
            
            # Melt the dataframe to transform from wide to long format
            df_long = df.melt(id_vars=['Province'], var_name='Year', value_name=value_name)
            
            # Extract the year from the 'Year' column (e.g., '2019年' -> 2019)
            df_long['Year'] = df_long['Year'].astype(str).str.extract(r'(\d{4})').astype(float)
            
            # Remove rows with missing years or values
            df_long = df_long.dropna(subset=['Year', value_name])
            df_long['Year'] = df_long['Year'].astype(int)
            
            # Filter for study period (2013-2019)
            df_long = df_long[(df_long['Year'] >= 2013) & (df_long['Year'] <= 2019)]
            
            print(f"✓ Processed {file_path}: {len(df_long)} records")
            return df_long
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None
    
    def load_and_process_all_files(self):
        """
        Load and process all socioeconomic data files.
        
        Returns:
            bool: Success status
        """
        print("=== Step 1: Loading and Processing Socioeconomic Data Files ===")
        
        processed_dataframes = []
        
        for file_path, value_name in self.files_to_process.items():
            if os.path.exists(file_path):
                print(f"Processing {file_path}...")
                processed_df = self.process_socioeconomic_data(file_path, value_name)
                if processed_df is not None:
                    processed_dataframes.append(processed_df)
            else:
                print(f"File not found: {file_path}")
        
        if not processed_dataframes:
            print("No data files processed successfully!")
            return False
        
        self.processed_dataframes = processed_dataframes
        print(f"Successfully processed {len(processed_dataframes)} data files")
        return True
    
    def merge_all_datasets(self):
        """
        Merge all processed datasets into a single comprehensive dataset.
        
        Returns:
            bool: Success status
        """
        print("\n=== Step 2: Merging All Datasets ===")
        
        if not self.processed_dataframes:
            print("No processed dataframes to merge!")
            return False
        
        try:
            # Merge all dataframes on Province and Year
            print("Merging dataframes...")
            merged_df = reduce(
                lambda left, right: pd.merge(left, right, on=['Province', 'Year'], how='outer'),
                self.processed_dataframes
            )
            
            print(f"Merged dataset shape: {merged_df.shape}")
            print(f"   Years covered: {merged_df['Year'].min()} - {merged_df['Year'].max()}")
            print(f"   Provinces: {merged_df['Province'].nunique()}")
            
            self.merged_data = merged_df
            return True
            
        except Exception as e:
            print(f"Error merging datasets: {e}")
            return False
    
    def calculate_derived_variables(self):
        """
        Calculate derived variables and economic indicators.
        
        Returns:
            bool: Success status
        """
        print("\n=== Step 3: Calculating Derived Variables ===")
        
        if self.merged_data is None:
            print("No merged data available!")
            return False
        
        df = self.merged_data.copy()
        
        try:
            # 1. Urbanization rate (if urban and total population available)
            # Note: Population data is typically in units of 10,000 people
            if any('Urban' in col and 'Population' in col for col in df.columns) and 'Total_Population' in df.columns:
                # Try to find urban population column
                urban_col = None
                for col in df.columns:
                    if 'urban' in col.lower() and 'population' in col.lower():
                        urban_col = col
                        break
                
                if urban_col:
                    df['Urbanization_Rate'] = (df[urban_col] / df['Total_Population']) * 100
                    print("✓ Urbanization rate calculated")
            
            # 2. Industrial structure ratios
            if all(col in df.columns for col in ['Primary_Industry_Value_Added', 'Secondary_Industry_Value_Added', 'Tertiary_Industry_Value_Added']):
                total_industry = (df['Primary_Industry_Value_Added'] + 
                                df['Secondary_Industry_Value_Added'] + 
                                df['Tertiary_Industry_Value_Added'])
                
                df['Primary_Industry_Share'] = (df['Primary_Industry_Value_Added'] / total_industry) * 100
                df['Secondary_Industry_Share'] = (df['Secondary_Industry_Value_Added'] / total_industry) * 100
                df['Tertiary_Industry_Share'] = (df['Tertiary_Industry_Value_Added'] / total_industry) * 100
                print("✓ Industrial structure shares calculated")
            
            # 3. Income inequality (Urban-Rural income ratio)
            if 'Urban_Per_Capita_Disposable_Income' in df.columns and 'Rural_Per_Capita_Disposable_Income' in df.columns:
                df['Urban_Rural_Income_Ratio'] = (df['Urban_Per_Capita_Disposable_Income'] / 
                                                df['Rural_Per_Capita_Disposable_Income'])
                print("✓ Urban-rural income ratio calculated")
            
            # 4. Government expenditure as % of GDP
            if 'Local_Government_Budget_Expenditure' in df.columns and 'GDP_100M_Yuan' in df.columns:
                df['Government_Expenditure_GDP_Ratio'] = (df['Local_Government_Budget_Expenditure'] / 
                                                        df['GDP_100M_Yuan']) * 100
                print("✓ Government expenditure to GDP ratio calculated")
            
            # 5. Consumption rate (as % of GDP)
            if 'Resident_Consumption_100M_Yuan' in df.columns and 'GDP_100M_Yuan' in df.columns:
                df['Consumption_GDP_Ratio'] = (df['Resident_Consumption_100M_Yuan'] / 
                                              df['GDP_100M_Yuan']) * 100
                print("✓ Consumption to GDP ratio calculated")
            
            # 6. Per capita GDP in 10,000 Yuan (for consistency)
            if 'GDP_100M_Yuan' in df.columns and 'Total_Population' in df.columns:
                df['GDP_Per_Capita_10K_Yuan'] = (df['GDP_100M_Yuan'] * 10) / df['Total_Population']
                print("✓ GDP per capita (10K Yuan) calculated")
            
            # 7. Economic growth rate (year-over-year GDP growth)
            if 'GDP_Index' in df.columns:
                df['GDP_Growth_Rate'] = df['GDP_Index'] - 100
                print("✓ GDP growth rate calculated")
            
            # 8. Unemployment rate per 10,000 people
            if 'Urban_Registered_Unemployment_10K_People' in df.columns and 'Total_Population' in df.columns:
                df['Unemployment_Rate_Per_10K'] = (df['Urban_Registered_Unemployment_10K_People'] / 
                                                  df['Total_Population']) * 100
                print("✓ Unemployment rate per 10K calculated")
            
            self.merged_data = df
            print(f"Derived variables calculation completed")
            print(f"   Final dataset shape: {df.shape}")
            
            return True
            
        except Exception as e:
            print(f"Error calculating derived variables: {e}")
            return False
    
    def save_processed_data(self, output_file='merged_socioeconomic_data.csv'):
        """
        Save the processed and merged socioeconomic data.
        
        Args:
            output_file (str): Output filename
            
        Returns:
            bool: Success status
        """
        print(f"\n=== Step 4: Saving Processed Data ===")
        
        if self.merged_data is None:
            print("No processed data to save!")
            return False
        
        try:
            # Create output directory
            os.makedirs('output', exist_ok=True)
            output_path = f'output/{output_file}'
            
            # Save the data
            self.merged_data.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            print(f"Data saved to: {output_path}")
            
            # Generate summary statistics
            df = self.merged_data
            print(f"\nDataset Summary:")
            print(f"   Shape: {df.shape}")
            print(f"   Years: {df['Year'].min()} - {df['Year'].max()}")
            print(f"   Provinces: {df['Province'].nunique()}")
            print(f"   Variables: {len(df.columns)}")
            
            # Show variable categories
            print(f"\nVariable Categories:")
            economic_vars = [col for col in df.columns if any(term in col for term in ['GDP', 'Income', 'Value_Added'])]
            population_vars = [col for col in df.columns if 'Population' in col or 'Ratio' in col]
            consumption_vars = [col for col in df.columns if 'Consumption' in col]
            government_vars = [col for col in df.columns if 'Government' in col or 'Budget' in col]
            
            print(f"   Economic indicators: {len(economic_vars)}")
            print(f"   Population indicators: {len(population_vars)}")
            print(f"   Consumption indicators: {len(consumption_vars)}")
            print(f"   Government indicators: {len(government_vars)}")
            
            # Show data completeness
            print(f"\nData Completeness:")
            completeness = (df.notna().sum() / len(df)) * 100
            high_quality_vars = completeness[completeness >= 90].index
            print(f"   Variables with >90% completeness: {len(high_quality_vars)}")
            
            # Show top provinces by coverage
            province_coverage = df.groupby('Province').size().sort_values(ascending=False)
            print(f"\nTop 5 Provinces by Data Coverage:")
            for province, count in province_coverage.head().items():
                print(f"   {province}: {count} year-observations")
            
            return True
            
        except Exception as e:
            print(f"Error saving data: {e}")
            return False


def main():
    """Main execution function for Phase 3: Socioeconomic Data Processing."""
    print("Phase 3: Socioeconomic Data Processing Pipeline")
    print("=" * 60)
    
    processor = SocioeconomicProcessor()
    
    # Step 1: Load and process all files
    if not processor.load_and_process_all_files():
        print("Failed at Step 1: Load and process files")
        return False
    
    # Step 2: Merge all datasets
    if not processor.merge_all_datasets():
        print("Failed at Step 2: Merge datasets")
        return False
    
    # Step 3: Calculate derived variables
    if not processor.calculate_derived_variables():
        print("Failed at Step 3: Calculate derived variables")
        return False
    
    # Step 4: Save processed data
    if not processor.save_processed_data():
        print("Failed at Step 4: Save processed data")
        return False
    
    print("\nPhase 3: Socioeconomic Data Processing completed successfully!")
    print(f"   Output: output/merged_socioeconomic_data.csv")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
