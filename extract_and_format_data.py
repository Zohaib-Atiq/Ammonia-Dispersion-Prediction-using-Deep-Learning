import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# Read the source data
source_file = "dataSet_preprocessed.xlsx"
df = pd.read_excel(source_file)

# Create a new DataFrame with the required column structure
output_data = pd.DataFrame()

# Map columns from source to required format
# Note: Adjust these mappings based on actual columns in dataSet_preprocessed.xlsx
# This is a template - modify column names to match your source file

column_mapping = {
    # Map source columns to required columns
    # Format: 'source_column_name': 'target_column_name'
    'Air Temperature (°C)': 'Air Temperature (°C)',
    'Wind Speed (m/s)': 'Wind Speed (m/s)',
    'Hole Diameter (mm)': 'Hole Diameter (mm)',
    'Indoor Concentration (ppm)': 'Indoor Concentration (ppm)',
    'Outdoor Concentraton (ppm)': 'Outdoor Concentration (ppm)',
    'Red Zone': 'Red Zone (miles)',
    'Orange Zone': 'Orange Zone (miles)',
    'Yellow Zone': 'Yellow Zone (miles)'
}

# Extract and rename columns
for source_col, target_col in column_mapping.items():
    if source_col in df.columns:
        output_data[target_col] = df[source_col]

# Remove any duplicate rows
output_data = output_data.drop_duplicates()

# Remove rows with missing values
output_data = output_data.dropna()

# Save to new Excel file with exact format
output_file = "unique_points1.xlsx"
output_data.to_excel(output_file, sheet_name="Sheet1", index=False)

print(f"✅ Data extracted and saved to '{output_file}'")
print(f"📊 Total rows: {len(output_data)}")
print(f"📋 Columns: {list(output_data.columns)}")
print("\nFirst 3 rows:")
print(output_data.head(3))