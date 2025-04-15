#!/bin/bash

# Check if the input file is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_file.ltr>"
    exit 1
fi

# Input file
input_file="$1"

# Output file
output_file="${input_file%.ltr}.txt"

# Process the file: convert to lowercase and replace | with ▁
cat "$input_file" | tr '[:upper:]' '[:lower:]' > "$output_file"
#sed 's/|/▁/g' "$input_file" | tr '[:upper:]' '[:lower:]' > "$output_file"

echo "Processed file saved as $output_file"
