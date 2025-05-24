import os
import re

# Define the directory path
directory = 'data/huali/train8'

# Ensure the directory exists
if not os.path.exists(directory):
    print(f"Directory {directory} does not exist.")
    exit(1)

# Counter for renamed files
renamed_count = 0

# Regular expression to match files that don't start with underscore
pattern = re.compile(r'^[^_]')

# Walk through the directory and its subdirectories
for root, dirs, files in os.walk(directory):
    for filename in files:
        # Check if the filename doesn't start with underscore
        if pattern.match(filename):
            # Create the new filename with underscore
            new_filename = '_' + filename
            
            # Get the full paths
            old_path = os.path.join(root, filename)
            new_path = os.path.join(root, new_filename)
            
            # Rename the file
            os.rename(old_path, new_path)
            
            # Print the rename operation
            print(f"Renamed: {old_path} -> {new_path}")
            renamed_count += 1

print(f"Renamed {renamed_count} files.")
