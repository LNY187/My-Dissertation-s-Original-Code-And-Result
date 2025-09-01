#!/usr/bin/env python3
"""
Phase 2: Crime Data Processing and Classification Pipeline
=========================================================

This script consolidates all crime data processing and classification operations:
1. Extract unique criminal charges from judgment texts
2. Classify crime types using pattern recognition
3. Aggregate crime data by province, year, and crime category
"""

import pandas as pd
import re
import os
import sys
from collections import defaultdict

class CrimeDataProcessor:
    """Main class for crime data processing and classification."""
    
    def __init__(self):
        self.unique_charges = set()
        self.crime_categories = {}
        self.processed_data = None
        
        # Define comprehensive crime category keywords
        self.violent_crime_keywords = [
            "故意伤害", "故意杀人", "伤害致死", "伤害致残", "伤害致人重伤", "伤害致人死亡", "恶意伤害", "打人致伤", "聚众斗殴致伤",
            "抢劫", "抢劫致人伤亡", "携带凶器抢劫", "抢劫致死", "抢夺", "抢夺枪支、弹药", "抢劫银行", "抢劫杀人", "暴力抢劫", "抢劫强奸",
            "强奸", "强奸未遂", "群体性侵", "轮奸", "强奸致死", "猥亵", "猥亵儿童", "猥亵妇女", "强制猥亵", "性侵害", "性暴力", "性骚扰",
            "家庭暴力", "虐待", "虐待被监管人", "虐待儿童", "虐待老人", "暴力取证", "暴力催收", "酷刑", "非法拘禁",
            "绑架", "绑架杀人", "拐卖儿童", "拐骗", "控制人身自由", "限制人身自由", "劫持人质", "软禁",
            "聚众斗殴", "寻衅滋事", "斗殴致伤", "斗殴致死", "团伙斗殴", "社会恶势力暴力行为",
            "妨害公务", "暴力袭警", "拒不配合执法", "聚众冲击国家机关", "聚众扰乱社会秩序", "暴力阻碍执行公务", "暴力抗法", "群体械斗", "暴力拆迁",
            "暴力威胁", "暴力胁迫", "暴力讨债", "暴力执行合同", "恐吓", "恐吓取证", "威胁他人安全", "威胁致人精神伤害", "恐怖活动",
            "投毒", "放火", "爆炸", "爆炸杀人", "投放危险物质", "故意毁坏财物致伤", "故意破坏公共设施致人伤亡",
            "黑社会性质组织", "组织、领导、参加黑社会", "恶势力犯罪集团", "暴力催债团伙", "暴力护伞", "有组织的暴力犯罪"
        ]
        
        self.property_crime_keywords = [
            "盗窃", "入室盗窃", "扒窃", "盗抢", "夜间盗窃", "团伙盗窃", "累犯盗窃", "盗掘古墓葬", "盗伐林木",
            "抢夺", "抢夺财物",
            "侵占", "职务侵占", "挪用资金", "挪用公款", "非法占有", "非法侵占",
            "敲诈勒索", "威胁获取财物", "恐吓勒索", "网络敲诈", "校园敲诈",
            "故意毁坏财物", "损坏公私财产", "财产损毁", "财产破坏行为", "拆毁他人财产",
            "掩饰、隐瞒犯罪所得", "掩饰赃物", "销赃", "收赃", "窝藏赃物", "隐匿赃款", "掩饰违法所得", "掩饰犯罪所得收益",
            "非法侵入住宅", "非法侵入公司财产", "非法占有财物", "非法侵占土地财产", "非法控制财产", "非法查封、扣押、冻结财物",
            "财产损害赔偿", "返还原物", "不当得利", "恢复原状",
            "诈骗兼盗窃", "诈骗兼抢夺", "非法处置担保财产", "假冒注册商标", "假币犯罪", "伪造货币", "使用假币", "盗用他人财产", 
            "盗掘化石、珍贵文物", "非法采矿、盗采资源", "刑事", "刑事判决书"
        ]
        
        self.financial_crime_keywords = [
            "诈骗", "合同诈骗", "信用卡诈骗", "贷款诈骗", "网络诈骗", "电信诈骗", "保险诈骗", "票据诈骗", "金融诈骗", "集资诈骗",
            "贿赂", "行贿", "受贿", "单位行贿", "利用影响力受贿", "对非国家工作人员行贿", "对有影响力的人行贿", "国家工作人员受贿", 
            "国家工作人员滥用职权", "玩忽职守",
            "伪造", "变造", "买卖证件", "虚假材料", "假发票", "假章", "假公文", "假合同", "假票据", "假身份证件", "虚开增值税专用发票", "虚假宣传",
            "金融", "证券", "操纵市场", "虚假披露", "非法集资", "非法吸收公众存款", "非法发行股票", "非法经营证券", "金融借款", "洗钱", "利率欺诈",
            "借款", "民间借贷", "融资租赁", "企业借贷", "企业承包", "合伙", "出资纠纷", "公司印章", "伪造公司资料", "公司治理", "公司决议",
            "保险", "意外险", "保单作假", "虚假赔付", "代理投保",
            "发票", "增值税发票", "财务造假", "会计造假", "税务欺诈", "虚开发票", "非法出票", "公司账目问题",
            "隐匿资产", "财产转移", "非法持有财物",
            "非法经营", "无证经营", "非法买卖外汇", "非法传销", "非法融资", "传销", "超范围经营", "擅自开设账户"
        ]
    
    def extract_unique_charges(self, input_file, output_file='unique_criminal_charges_final.txt'):
        """
        Extract unique criminal charges from the crime dataset.
        
        Args:
            input_file (str): Path to the main crime dataset
            output_file (str): Path to output unique charges file
        """
        print("=== Step 1: Extracting Unique Criminal Charges ===")
        print(f"Processing {input_file} to find unique charges...")
        
        unique_charges = set()
        
        try:
            chunk_iter = pd.read_csv(input_file, chunksize=10000, low_memory=False, on_bad_lines='skip')
            
            for i, chunk in enumerate(chunk_iter):
                print(f"Processing chunk {i+1}")
                
                for index, row in chunk.iterrows():
                    case_type = row.get('case_type')
                    if pd.isna(case_type):
                        continue
                    
                    charge_to_add = None
                    
                    # Process generic "刑事" cases by extracting from judgment text
                    if "刑事" in str(case_type):
                        judgment = row.get('judgment')
                        if pd.notna(judgment):
                            # Extract using pattern: 犯[crime type]罪
                            match = re.search(r'犯([\u4e00-\u9fa5].*?罪)', str(judgment))
                            if match:
                                charge = match.group(1)
                                # Clean and validate
                                charge = re.sub(r'[\s,\uff0c\u3002].*$', '', charge).strip()
                                if charge.endswith('罪') and len(charge) > 1:
                                    charge_to_add = charge
                    else:
                        # Use existing case_type if it's already specific
                        charge = str(case_type).strip()
                        if charge.endswith('罪') and len(charge) > 1:
                            charge_to_add = charge
                    
                    if charge_to_add:
                        # Final validation - exclude invalid entries
                        if not re.search(r'[、X\d]', charge_to_add):
                            unique_charges.add(charge_to_add)
            
            self.unique_charges = unique_charges
            
            print(f"Found {len(unique_charges)} unique charges.")
            
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                for charge in sorted(list(unique_charges)):
                    f.write(f"{charge}\n")
            
            print(f"Successfully wrote unique charges to {output_file}")
            return True
            
        except Exception as e:
            print(f"Error extracting charges: {e}")
            return False
    
    def classify_crime_type(self, row):
        """
        Classify a single crime record into major categories.
        
        Args:
            row: DataFrame row containing case_type and judgment fields
            
        Returns:
            str: Crime category classification
        """
        case_type = str(row['case_type'])
        judgment = str(row['judgment'])
        
        # Check for specific crime patterns
        if re.search(r'盗窃', case_type) or re.search(r'盗窃', judgment):
            return 'Theft'
        if re.search(r'诈骗', case_type) or re.search(r'诈骗', judgment):
            return 'Fraud'
        if re.search(r'故意伤害', case_type) or re.search(r'故意伤害', judgment):
            return 'Assault'
        if re.search(r'毒品|贩卖|运输', case_type) or re.search(r'毒品|贩卖|运输', judgment):
            return 'Drug-related'
        if re.search(r'交通肇事', case_type) or re.search(r'交通肇事', judgment):
            return 'Traffic Offense'
        if re.search(r'抢劫', case_type) or re.search(r'抢劫', judgment):
            return 'Robbery'
        if re.search(r'故意杀人', case_type) or re.search(r'故意杀人', judgment):
            return 'Homicide'
        
        # If case_type is already specific (not generic "刑事"), use it
        if case_type and case_type != '刑事':
            return case_type
        
        return 'Other'
    
    def categorize_by_keywords(self, text):
        """
        Categorize crime using keyword matching approach.
        
        Args:
            text (str): Combined case_type and judgment text
            
        Returns:
            str: Crime category (Violent, Property, Financial, Other)
        """
        text = str(text).lower()
        
        # Check violent crime keywords
        for keyword in self.violent_crime_keywords:
            if keyword in text:
                return 'Violent'
        
        # Check property crime keywords  
        for keyword in self.property_crime_keywords:
            if keyword in text:
                return 'Property'
        
        # Check financial crime keywords
        for keyword in self.financial_crime_keywords:
            if keyword in text:
                return 'Financial'
        
        return 'Other'
    
    def process_crime_data_chunk(self, chunk, chunk_number):
        """
        Process a single chunk of crime data.
        
        Args:
            chunk: DataFrame chunk
            chunk_number: Chunk identifier
            
        Returns:
            DataFrame: Processed chunk with classifications
        """
        print(f"--- Processing Chunk {chunk_number} ---")
        print(f"Initial chunk shape: {chunk.shape}")
        
        # Extract year from formatted_datetime
        chunk['year'] = pd.to_datetime(chunk['formatted_datetime'], errors='coerce').dt.year
        
        # For missing years, try to extract from case_number (format: (YYYY)...)
        missing_year_mask = chunk['year'].isna()
        if missing_year_mask.any():
            chunk.loc[missing_year_mask, 'year'] = chunk.loc[missing_year_mask, 'case_number'].str.extract(r'\((\d{4})\)')[0].astype(float)
        
        # Filter for study period (2013-2019)
        chunk = chunk[(chunk['year'] >= 2013) & (chunk['year'] <= 2019)]
        print(f"After temporal filtering (2013-2019): {chunk.shape}")
        
        # Remove records with missing essential information
        chunk = chunk.dropna(subset=['incident_province'])
        print(f"After removing missing provinces: {chunk.shape}")
        
        # Apply crime classifications
        chunk['crime_category'] = chunk.apply(self.classify_crime_type, axis=1)
        
        # Apply keyword-based categorization
        chunk['combined_text'] = chunk['case_type'].astype(str) + ' ' + chunk['judgment'].astype(str)
        chunk['crime_type'] = chunk['combined_text'].apply(self.categorize_by_keywords)
        
        # Clean up
        chunk = chunk.drop('combined_text', axis=1)
        
        print(f"Final chunk shape: {chunk.shape}")
        return chunk
    
    def aggregate_crime_data(self, input_file, output_file='aggregated_crime_data.csv'):
        """
        Aggregate crime data by province, year, and crime type.
        
        Args:
            input_file (str): Path to the main crime dataset
            output_file (str): Path to output aggregated data
        """
        print("\n=== Step 3: Aggregating Crime Data ===")
        print(f"Processing {input_file} for aggregation...")
        
        try:
            # Process data in chunks
            chunk_iter = pd.read_csv(input_file, chunksize=10000, low_memory=False)
            all_chunks = []
            
            for i, chunk in enumerate(chunk_iter):
                processed_chunk = self.process_crime_data_chunk(chunk, i+1)
                if not processed_chunk.empty:
                    all_chunks.append(processed_chunk)
            
            if not all_chunks:
                print("No data to process!")
                return False
            
            # Combine all chunks
            print("\nCombining all processed chunks...")
            combined_df = pd.concat(all_chunks, ignore_index=True)
            print(f"Combined dataset shape: {combined_df.shape}")
            
            # Aggregate by province, year, and crime type
            print("\nAggregating by province, year, and crime type...")
            
            aggregated_data = []
            
            # Get unique combinations
            for province in combined_df['incident_province'].unique():
                for year in range(2013, 2020):
                    province_year_data = combined_df[
                        (combined_df['incident_province'] == province) & 
                        (combined_df['year'] == year)
                    ]
                    
                    if not province_year_data.empty:
                        # Count by crime types
                        violent_count = len(province_year_data[province_year_data['crime_type'] == 'Violent'])
                        property_count = len(province_year_data[province_year_data['crime_type'] == 'Property'])
                        financial_count = len(province_year_data[province_year_data['crime_type'] == 'Financial'])
                        other_count = len(province_year_data[province_year_data['crime_type'] == 'Other'])
                        total_count = len(province_year_data)
                        
                        aggregated_data.append({
                            'Province': province,
                            'Year': year,
                            'Violent_Crime_Count': violent_count,
                            'Property_Crime_Count': property_count,
                            'Financial_Crime_Count': financial_count,
                            'Other_Crime_Count': other_count,
                            'Total_Crime_Count': total_count
                        })
            
            # Create aggregated DataFrame
            aggregated_df = pd.DataFrame(aggregated_data)
            
            if aggregated_df.empty:
                print("No aggregated data generated!")
                return False
            
            print(f"Aggregated dataset shape: {aggregated_df.shape}")
            
            # Save aggregated data
            os.makedirs('output', exist_ok=True)
            output_path = f'output/{output_file}'
            aggregated_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            print(f"Aggregated crime data saved to: {output_path}")
            
            # Generate summary statistics
            print(f"\nSummary Statistics:")
            print(f"  Total province-year observations: {len(aggregated_df)}")
            print(f"  Provinces covered: {aggregated_df['Province'].nunique()}")
            print(f"  Years covered: {sorted(aggregated_df['Year'].unique())}")
            print(f"  Total crimes in dataset: {aggregated_df['Total_Crime_Count'].sum():,}")
            print(f"  Average crimes per province-year: {aggregated_df['Total_Crime_Count'].mean():.1f}")
            
            # Crime type distribution
            print(f"\nCrime Type Distribution:")
            print(f"  Violent crimes: {aggregated_df['Violent_Crime_Count'].sum():,}")
            print(f"  Property crimes: {aggregated_df['Property_Crime_Count'].sum():,}")
            print(f"  Financial crimes: {aggregated_df['Financial_Crime_Count'].sum():,}")
            print(f"  Other crimes: {aggregated_df['Other_Crime_Count'].sum():,}")
            
            self.processed_data = aggregated_df
            return True
            
        except Exception as e:
            print(f"Error aggregating crime data: {e}")
            return False


def main():
    """Main execution function for Phase 2: Crime Data Processing and Classification."""
    print("Phase 2: Crime Data Processing and Classification Pipeline")
    print("=" * 70)
    
    processor = CrimeDataProcessor()
    
    # Input file path
    crime_data_file = 'data/ChinaCrimeDatas.csv'
    
    if not os.path.exists(crime_data_file):
        print(f"Crime data file not found: {crime_data_file}")
        print("Please update the path to your main crime dataset")
        return False
    
    # Step 1: Extract unique charges
    if not processor.extract_unique_charges(crime_data_file):
        print("Failed at Step 1: Extract unique charges")
        return False
    
    # Step 2: Aggregate crime data with classifications
    if not processor.aggregate_crime_data(crime_data_file):
        print("Failed at Step 2: Aggregate crime data")
        return False
    
    print("\nPhase 2: Crime Data Processing and Classification completed successfully!")
    print(f"   Output: output/aggregated_crime_data.csv")
    print(f"   Output: unique_criminal_charges_final.txt")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
