#!/usr/bin/env python3
"""
Phase 1: Geographic Standardization Pipeline
============================================

This script consolidates all geographic data processing operations:
1. Extract unique cities from crime data
2. Create province-city mapping
3. Fill missing province information using dual-column approach
"""

import pandas as pd
import json
import os
import sys

class GeographicProcessor:
    """Main class for geographic data standardization."""
    
    def __init__(self):
        self.unique_cities = []
        self.province_city_mapping = {}
        self.city_to_province = {}
    
    def extract_unique_cities(self, input_file='ChinaCrimeDatas.csv', 
                            output_file='unique_cities.txt'):
        """
        Extract unique cities from the city column of CSV file.
        
        Args:
            input_file (str): Path to input CSV file
            output_file (str): Path to output text file
        """
        print("=== Step 1: Extracting Unique Cities ===")
        
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: {input_file} not found!")
            return False
        
        try:
            # Read the CSV file
            print(f"Reading {input_file}...")
            df = pd.read_csv(input_file)
            
            # Check if 'city' column exists
            if 'city' not in df.columns:
                print("Error: 'city' column not found in the CSV file!")
                print("Available columns:", df.columns.tolist())
                return False
            
            # Extract unique cities (remove NaN values and empty strings)
            unique_cities = df['city'].dropna().astype(str).unique()
            
            # Remove any empty strings
            unique_cities = [city for city in unique_cities if city.strip() != '']
            
            # Sort the cities alphabetically
            self.unique_cities = sorted(unique_cities)
            
            # Write to text file
            print(f"Writing {len(self.unique_cities)} unique cities to {output_file}...")
            with open(output_file, 'w', encoding='utf-8') as f:
                for city in self.unique_cities:
                    f.write(city + '\n')
            
            print(f"Successfully extracted {len(self.unique_cities)} unique cities!")
            print(f"Results saved to: {output_file}")
            return True
            
        except Exception as e:
            print(f"Error extracting unique cities: {e}")
            return False
    
    def create_province_city_mapping(self, unique_cities_file='unique_cities.txt',
                                   output_file='filtered_province_city_mapping.json'):
        """
        Create province-city mapping based on Chinese administrative divisions.
        
        Args:
            unique_cities_file (str): Path to unique cities text file
            output_file (str): Path to output JSON mapping file
        """
        print("\n=== Step 2: Creating Province-City Mapping ===")
        
        # Check if input file exists
        if not os.path.exists(unique_cities_file):
            print(f"Error: {unique_cities_file} not found!")
            return False
        
        try:
            # Read unique cities
            print(f"Reading unique cities from {unique_cities_file}...")
            with open(unique_cities_file, 'r', encoding='utf-8') as f:
                unique_cities = [line.strip() for line in f if line.strip()]
            
            print(f"Found {len(unique_cities)} unique cities")
            
            # Create comprehensive city to province mapping
            city_to_province_map = {
                # 河南省 (Henan Province)
                "郑州": "河南省", "洛阳": "河南省", "开封": "河南省", "安阳": "河南省",
                "新乡": "河南省", "濮阳": "河南省", "许昌": "河南省", "漯河": "河南省",
                "三门峡": "河南省", "南阳": "河南省", "商丘": "河南省",
                
                # 辽宁省 (Liaoning Province)
                "沈阳": "辽宁省", "大连": "辽宁省", "鞍山": "辽宁省", "抚顺": "辽宁省",
                "本溪": "辽宁省", "丹东": "辽宁省", "锦州": "辽宁省", "营口": "辽宁省",
                "阜新": "辽宁省", "辽阳": "辽宁省", "盘锦": "辽宁省",
                
                # 山东省 (Shandong Province)
                "济南": "山东省", "青岛": "山东省", "淄博": "山东省", "枣庄": "山东省",
                "东营": "山东省", "烟台": "山东省", "潍坊": "山东省",
                
                # 江苏省 (Jiangsu Province)
                "南京": "江苏省", "无锡": "江苏省", "徐州": "江苏省", "常州": "江苏省",
                "苏州": "江苏省", "南通": "江苏省", "连云港": "江苏省", "淮安": "江苏省",
                "盐城": "江苏省", "扬州": "江苏省", "镇江": "江苏省", "泰州": "江苏省",
                "宿迁": "江苏省",
                
                # 湖北省 (Hubei Province)
                "武汉": "湖北省", "黄石": "湖北省", "十堰": "湖北省", "宜昌": "湖北省",
                "襄阳": "湖北省", "鄂州": "湖北省", "荆门": "湖北省", "孝感": "湖北省",
                "荆州": "湖北省", "黄冈": "湖北省", "咸宁": "湖北省",
                
                # 四川省 (Sichuan Province)
                "成都": "四川省", "自贡": "四川省", "攀枝花": "四川省", "泸州": "四川省",
                "德阳": "四川省", "绵阳": "四川省", "广元": "四川省", "遂宁": "四川省",
                "内江": "四川省", "乐山": "四川省", "南充": "四川省", "眉山": "四川省",
                "宜宾": "四川省", "广安": "四川省", "达州": "四川省", "雅安": "四川省",
                "巴中": "四川省", "资阳": "四川省",
                
                # 广东省 (Guangdong Province)
                "广州": "广东省", "韶关": "广东省", "深圳": "广东省", "珠海": "广东省",
                "汕头": "广东省", "佛山": "广东省", "江门": "广东省", "湛江": "广东省",
                "茂名": "广东省", "肇庆": "广东省", "惠州": "广东省", "梅州": "广东省",
                "汕尾": "广东省", "河源": "广东省", "阳江": "广东省", "清远": "广东省",
                "东莞": "广东省", "中山": "广东省", "潮州": "广东省", "揭阳": "广东省",
                "云浮": "广东省",
                
                # 浙江省 (Zhejiang Province)
                "杭州": "浙江省", "宁波": "浙江省", "温州": "浙江省", "嘉兴": "浙江省",
                "湖州": "浙江省", "绍兴": "浙江省", "金华": "浙江省", "衢州": "浙江省",
                "舟山": "浙江省", "台州": "浙江省", "丽水": "浙江省",
                
                # Additional provinces and cities
                "福州": "福建省", "厦门": "福建省", "莆田": "福建省", "三明": "福建省",
                "泉州": "福建省", "漳州": "福建省", "南平": "福建省", "龙岩": "福建省",
                "宁德": "福建省",
                
                "石家庄": "河北省", "唐山": "河北省", "秦皇岛": "河北省", "邯郸": "河北省",
                "邢台": "河北省", "保定": "河北省", "张家口": "河北省", "承德": "河北省",
                "沧州": "河北省", "廊坊": "河北省", "衡水": "河北省",
                
                "太原": "山西省", "大同": "山西省", "阳泉": "山西省", "长治": "山西省",
                "晋城": "山西省", "朔州": "山西省", "晋中": "山西省", "运城": "山西省",
                "忻州": "山西省", "临汾": "山西省", "吕梁": "山西省",
                
                "呼和浩特": "内蒙古自治区", "包头": "内蒙古自治区", "乌海": "内蒙古自治区",
                "赤峰": "内蒙古自治区", "通辽": "内蒙古自治区", "鄂尔多斯": "内蒙古自治区",
                "呼伦贝尔": "内蒙古自治区", "巴彦淖尔": "内蒙古自治区", "乌兰察布": "内蒙古自治区",
                
                "长春": "吉林省", "吉林": "吉林省", "四平": "吉林省", "辽源": "吉林省",
                "通化": "吉林省", "白山": "吉林省", "松原": "吉林省", "白城": "吉林省",
                
                "哈尔滨": "黑龙江省", "齐齐哈尔": "黑龙江省", "鸡西": "黑龙江省",
                "鹤岗": "黑龙江省", "双鸭山": "黑龙江省", "大庆": "黑龙江省",
                "伊春": "黑龙江省", "佳木斯": "黑龙江省", "七台河": "黑龙江省",
                "牡丹江": "黑龙江省", "黑河": "黑龙江省", "绥化": "黑龙江省",
                
                # Special Administrative Regions and Municipalities
                "北京": "北京市", "天津": "天津市", "上海": "上海市", "重庆": "重庆市",
                "香港": "香港特别行政区", "澳门": "澳门特别行政区"
            }
            
            # Filter mapping to only include cities present in unique_cities
            filtered_mapping = {}
            cities_found = set()
            
            for city in unique_cities:
                for map_city, province in city_to_province_map.items():
                    if map_city in city:
                        if province not in filtered_mapping:
                            filtered_mapping[province] = []
                        if city not in filtered_mapping[province]:
                            filtered_mapping[province].append(city)
                        cities_found.add(city)
                        break
            
            # Report cities not found in mapping
            cities_not_found = set(unique_cities) - cities_found
            if cities_not_found:
                print(f"Warning: Cities not found in mapping: {list(cities_not_found)}")
            
            # Save the filtered mapping
            self.province_city_mapping = filtered_mapping
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_mapping, f, ensure_ascii=False, indent=2)
            
            # Create reverse mapping for efficient lookup
            self.city_to_province = {city: province for province, cities in filtered_mapping.items() for city in cities}
            
            print(f"Successfully created mapping for {len(cities_found)} cities across {len(filtered_mapping)} provinces")
            print(f"Mapping saved to: {output_file}")
            
            # Display summary
            print("\nProvince-City Mapping Summary:")
            for province, cities in sorted(filtered_mapping.items()):
                print(f"  {province}: {len(cities)} cities")
            
            return True
            
        except Exception as e:
            print(f"Error creating province-city mapping: {e}")
            return False
    
    def fill_provinces_dual_column(self, data_file, mapping_file='filtered_province_city_mapping.json'):
        """
        Fill missing provinces using dual-column approach (incident_city then city).
        
        Args:
            data_file (str): Path to the main crime dataset
            mapping_file (str): Path to the province-city mapping JSON file
        """
        print("\n=== Step 3: Filling Missing Provinces (Dual-Column Approach) ===")
        
        try:
            # Load the dataset
            print(f"Loading dataset: {data_file}")
            try:
                df = pd.read_csv(data_file, low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(data_file, encoding='gb18030', low_memory=False)
            
            print(f"Dataset shape: {df.shape}")
            
            # Load the province-city mapping
            with open(mapping_file, 'r', encoding='utf-8') as f:
                province_to_city = json.load(f)
            
            # Create reverse mapping
            city_to_province = {city: province for province, cities in province_to_city.items() for city in cities}
            
            initial_missing_count = df['incident_province'].isna().sum()
            print(f"Initial missing provinces: {initial_missing_count}")
            
            if initial_missing_count == 0:
                print("No missing provinces to fill.")
                return True
            
            # Function to get province from city name
            def get_province(city_val):
                if pd.isna(city_val):
                    return None
                # Check for substring matches
                for city, province in city_to_province.items():
                    if city in str(city_val):
                        return province
                return None
            
            # Stage 1: Try incident_city column
            print("\nStage 1: Processing incident_city column...")
            missing_mask = df['incident_province'].isna()
            
            if 'incident_city' in df.columns:
                df.loc[missing_mask, 'incident_province'] = df.loc[missing_mask, 'incident_city'].apply(get_province)
                stage1_filled = initial_missing_count - df['incident_province'].isna().sum()
                print(f"Stage 1 results: {stage1_filled} provinces filled using incident_city column")
            else:
                print("incident_city column not found, skipping Stage 1")
                stage1_filled = 0
            
            # Stage 2: Try city column for remaining missing
            print("\nStage 2: Processing city column...")
            missing_mask = df['incident_province'].isna()
            remaining_missing = missing_mask.sum()
            
            if 'city' in df.columns and remaining_missing > 0:
                df.loc[missing_mask, 'incident_province'] = df.loc[missing_mask, 'city'].apply(get_province)
                stage2_filled = remaining_missing - df['incident_province'].isna().sum()
                print(f"Stage 2 results: {stage2_filled} provinces filled using city column")
            else:
                if 'city' not in df.columns:
                    print("city column not found, skipping Stage 2")
                else:
                    print("No remaining missing provinces for Stage 2")
                stage2_filled = 0
            
            # Final reporting
            final_missing_count = df['incident_province'].isna().sum()
            total_filled = initial_missing_count - final_missing_count
            success_rate = (total_filled / initial_missing_count) * 100 if initial_missing_count > 0 else 100
            
            print(f"\nFinal Results:")
            print(f"  Initial missing: {initial_missing_count}")
            print(f"  Stage 1 filled (incident_city): {stage1_filled}")
            print(f"  Stage 2 filled (city): {stage2_filled}")
            print(f"  Total filled: {total_filled}")
            print(f"  Remaining missing: {final_missing_count}")
            print(f"  Success rate: {success_rate:.1f}%")
            
            # Save the updated dataset
            backup_file = data_file.replace('.csv', '_backup.csv')
            print(f"\nCreating backup: {backup_file}")
            df.to_csv(backup_file, index=False, encoding='utf-8-sig')
            
            print(f"Saving updated dataset: {data_file}")
            df.to_csv(data_file, index=False, encoding='utf-8-sig')
            
            print(f"Successfully updated '{data_file}'.")
            print(f"Filled {total_filled} missing province values.")
            
            if final_missing_count > 0:
                print(f"Warning: There are still {final_missing_count} records with no province information.")
            
            return True
            
        except Exception as e:
            print(f"Error filling provinces: {e}")
            return False


def main():
    """Main execution function for Phase 1: Geographic Standardization."""
    print("Phase 1: Geographic Standardization Pipeline")
    print("=" * 60)
    
    processor = GeographicProcessor()
    
    # Step 1: Extract unique cities
    if not processor.extract_unique_cities('data/remaining_missing_provinces.csv'):
        print("Failed at Step 1: Extract unique cities")
        return False
    
    # Step 2: Create province-city mapping
    if not processor.create_province_city_mapping():
        print("Failed at Step 2: Create province-city mapping")
        return False
    
    # Step 3: Fill missing provinces
    crime_data_file = 'data/ChinaCrimeDatas.csv'
    if os.path.exists(crime_data_file):
        if not processor.fill_provinces_dual_column(crime_data_file):
            print("Failed at Step 3: Fill missing provinces")
            return False
    else:
        print(f"Warning: Crime data file not found: {crime_data_file}")
        print("Please update the path to your main crime dataset")
    
    print("\nPhase 1: Geographic Standardization completed successfully!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
