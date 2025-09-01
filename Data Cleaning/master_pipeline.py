#!/usr/bin/env python3
"""
Master Pipeline Controller for China Crime Data Processing
"""

import os
import sys
from datetime import datetime

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all phase modules
try:
    from phase1_geographic_standardization import main as phase1_main
    from phase2_crime_processing import main as phase2_main
    from phase3_socioeconomic_processing import main as phase3_main
    from phase4_final_dataset_creation import main as phase4_main
except ImportError as e:
    print(f"Error importing phase modules: {e}")
    sys.exit(1)

def run_complete_pipeline():
    """
    Run the complete data processing pipeline from Phase 1 to Phase 4.
    """
    start_time = datetime.now()
    
    print("=" * 80)
    print("CHINA CRIME DATA PROCESSING PIPELINE")
    print("=" * 80)
    print(f"Pipeline started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Phase 1: Geographic Data Standardization
        print("PHASE 1: GEOGRAPHIC DATA STANDARDIZATION")
        print("-" * 50)
        success_phase1 = phase1_main()
        if not success_phase1:
            print("Phase 1 failed. Stopping pipeline.")
            return False
        print("Phase 1 completed successfully!")
        print()
        
        # Phase 2: Crime Data Processing
        print("PHASE 2: CRIME DATA PROCESSING")
        print("-" * 50)
        success_phase2 = phase2_main()
        if not success_phase2:
            print("Phase 2 failed. Stopping pipeline.")
            return False
        print("Phase 2 completed successfully!")
        print()
        
        # Phase 3: Socioeconomic Data Processing
        print("PHASE 3: SOCIOECONOMIC DATA PROCESSING")
        print("-" * 50)
        success_phase3 = phase3_main()
        if not success_phase3:
            print("Phase 3 failed. Stopping pipeline.")
            return False
        print("Phase 3 completed successfully!")
        print()
        
        # Phase 4: Final Dataset Creation
        print("PHASE 4: FINAL DATASET CREATION")
        print("-" * 50)
        success_phase4 = phase4_main()
        if not success_phase4:
            print("Phase 4 failed. Stopping pipeline.")
            return False
        print("Phase 4 completed successfully!")
        print()
        
        end_time = datetime.now()
        total_time = end_time - start_time
        
        print("=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Total processing time: {total_time}")
        print(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("Output files generated:")
        print("  - output/unique_cities.csv")
        print("  - output/incident_provinces.csv")
        print("  - output/aggregated_crime_data.csv")
        print("  - output/merged_socioeconomic_data.csv")
        print("  - output/final_panel_dataset.csv")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        return False

def run_individual_phase(phase_number):
    """
    Run an individual phase of the pipeline.
    
    Args:
        phase_number (int): Phase number to run (1-4)
    """
    phase_functions = {
        1: ("Geographic Data Standardization", phase1_main),
        2: ("Crime Data Processing", phase2_main),
        3: ("Socioeconomic Data Processing", phase3_main),
        4: ("Final Dataset Creation", phase4_main)
    }
    
    if phase_number not in phase_functions:
        print(f"Invalid phase number: {phase_number}. Must be 1-4.")
        return False
    
    phase_name, phase_function = phase_functions[phase_number]
    
    print(f"PHASE {phase_number}: {phase_name.upper()}")
    print("-" * 50)
    
    start_time = datetime.now()
    success = phase_function()
    end_time = datetime.now()
    
    if success:
        print(f"Phase {phase_number} completed successfully in {end_time - start_time}")
    else:
        print(f"Phase {phase_number} failed")
    
    return success

def main():
    """
    Main function to handle command line arguments and run the pipeline.
    """
    if len(sys.argv) > 1:
        try:
            phase_number = int(sys.argv[1])
            return run_individual_phase(phase_number)
        except ValueError:
            print("Invalid argument. Use a number 1-4 to run individual phases.")
            print("Usage:")
            print("  python master_pipeline.py        # Run complete pipeline")
            print("  python master_pipeline.py 1      # Run Phase 1 only")
            print("  python master_pipeline.py 2      # Run Phase 2 only")
            print("  python master_pipeline.py 3      # Run Phase 3 only")
            print("  python master_pipeline.py 4      # Run Phase 4 only")
            return False
    else:
        return run_complete_pipeline()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
