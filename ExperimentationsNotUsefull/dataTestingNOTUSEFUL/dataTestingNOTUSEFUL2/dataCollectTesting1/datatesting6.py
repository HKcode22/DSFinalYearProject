import os
import pandas as pd
import re
from collections import defaultdict

# Define input and output directories
input_dir = "AKnownData"
output_dir = "Deduplicated_Data"
os.makedirs(output_dir, exist_ok=True)

# Function to extract base filename without duplicative patterns
def get_base_filename(filename):
    # Strip file extension first
    base = os.path.splitext(filename)[0]
    
    # Remove copy patterns (e.g., "copy", "copy 2")
    base = re.sub(r'[ _-]copy[ _-]?\d*', '', base)
    
    # Handle master copy patterns
    base = re.sub(r'master[ _-]copy[ _-]?\d*', 'master', base)
    
    # Remove dashes/underscores in similar file patterns
    base = base.replace('-', '_')
    
    return base.lower()  # Normalize to lowercase

# Scan the input directory for CSV files
csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]
print(f"Found {len(csv_files)} CSV files in {input_dir}")

# Group files by base name
file_groups = defaultdict(list)
for filename in csv_files:
    base_name = get_base_filename(filename)
    file_groups[base_name].append(filename)

# Filter to find duplicate groups (more than one file per base name)
duplicate_groups = {base: files for base, files in file_groups.items() if len(files) > 1}
single_files = {base: files[0] for base, files in file_groups.items() if len(files) == 1}

print(f"Found {len(duplicate_groups)} groups of duplicate files")
for base, files in duplicate_groups.items():
    print(f"  {base}: {len(files)} files")

# Process each group of duplicate files
for base_name, file_list in duplicate_groups.items():
    print(f"\nProcessing duplicate group: {base_name}")
    
    all_dfs = []
    for filename in file_list:
        file_path = os.path.join(input_dir, filename)
        try:
            # Read the CSV file
            df = pd.read_csv(file_path, low_memory=False)
            
            # Add source column to track origin
            df['source_file'] = filename
            
            # Print info about the file
            print(f"  Read {filename}: {len(df)} rows, {len(df.columns)} columns")
            
            all_dfs.append(df)
        except Exception as e:
            print(f"  Error reading {filename}: {str(e)}")
    
    if all_dfs:
        try:
            # Merge all DataFrames
            merged_df = pd.concat(all_dfs, ignore_index=True)
            
            # Remove duplicate rows
            before_count = len(merged_df)
            merged_df = merged_df.drop_duplicates()
            after_count = len(merged_df)
            
            # Create output filename
            output_filename = f"{base_name}_merged.csv"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save merged file
            merged_df.to_csv(output_path, index=False)
            print(f"  Merged file saved to {output_filename}")
            print(f"  Removed {before_count - after_count} duplicate rows")
        except Exception as e:
            print(f"  Error merging files: {str(e)}")

# Copy non-duplicate files to output directory
print("\nCopying unique files to output directory:")
for base_name, filename in single_files.items():
    try:
        # Read and write to standardize format
        df = pd.read_csv(os.path.join(input_dir, filename), low_memory=False)
        output_path = os.path.join(output_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"  Copied {filename}")
    except Exception as e:
        print(f"  Error copying {filename}: {str(e)}")

print(f"\nAll processing complete. Merged files saved to {output_dir}")
