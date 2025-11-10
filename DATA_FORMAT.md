# 📊 Data Format Guide

This document explains the expected data format for the ammonia dispersion prediction model.

## Required Data File

**Filename**: `unique_points.xlsx`  
**Format**: Excel (.xlsx)  
**Sheet Name**: `Sheet1`

## Column Names (Exact Match Required)

The Excel file must contain the following columns with **exact** names (including spaces):

### Input Features (3 columns)

| Column Name | Description | Unit | Type | Example |
|------------|-------------|------|------|---------|
| `Air Temperature (°C)` | Ambient air temperature | Celsius | Float | 25.5 |
| `Wind Speed (m/s)` | Wind velocity | meters per second | Float | 5.2 |
| `Hole Diameter (mm)` | Leak hole diameter | millimeters | Float | 10.0 |

### Output Variables (5 columns)

| Column Name | Description | Unit | Type | Example |
|------------|-------------|------|------|---------|
| `Indoor Concentration (ppm)` | Ammonia concentration indoors | parts per million | Float | 150.5 |
| `Outdoor Concentraton (ppm)` | Ammonia concentration outdoors | parts per million | Float | 75.2 |
| `Red Zone (miles) ` | Distance to red threat zone | miles | Float | 2.5 |
| `Orange Zone (miles) ` | Distance to orange threat zone | miles | Float | 1.8 |
| `Yellow Zone (miles)` | Distance to yellow threat zone | miles | Float | 1.2 |

**⚠️ Important Notes:**
- Note the space after `(miles) ` in "Red Zone (miles) " and "Orange Zone (miles) "
- "Outdoor Concentraton" has a typo (missing 'i' in Concentration) - keep it as is for consistency
- All columns must be present

## Sample Data Structure

### Excel Preview:

| Air Temperature (°C) | Wind Speed (m/s) | Hole Diameter (mm) | Indoor Concentration (ppm) | Outdoor Concentraton (ppm) | Red Zone (miles)  | Orange Zone (miles)  | Yellow Zone (miles) |
|---------------------|------------------|-------------------|---------------------------|---------------------------|-------------------|---------------------|-------------------|
| 25.0 | 5.0 | 10.0 | 150.5 | 75.2 | 2.5 | 1.8 | 1.2 |
| 30.0 | 3.5 | 15.0 | 220.8 | 110.4 | 3.2 | 2.4 | 1.8 |
| 20.0 | 7.0 | 8.0 | 100.3 | 50.1 | 1.8 | 1.3 | 0.9 |

### Minimum Requirements:

- **Minimum rows**: 20 (for meaningful training)
- **Recommended rows**: 100+ for better model performance
- **No missing values**: All cells must have numeric values
- **No text/categorical data**: All columns must be numeric

## Data Ranges (Typical Values)

### Input Features:

```
Air Temperature: 
  - Min: -20°C
  - Max: 50°C
  - Typical: 10-35°C

Wind Speed:
  - Min: 0.1 m/s
  - Max: 20 m/s
  - Typical: 2-10 m/s

Hole Diameter:
  - Min: 1 mm
  - Max: 100 mm
  - Typical: 5-50 mm
```

### Output Variables:

```
Indoor Concentration:
  - Typically: 10-1000+ ppm
  - Varies based on leak size and ventilation

Outdoor Concentration:
  - Typically: 5-500 ppm
  - Usually lower than indoor

Threat Zones:
  - Red Zone: Highest risk area
  - Orange Zone: Medium risk area
  - Yellow Zone: Lower risk area
  - All measured in miles from source
```

## Creating Your Own Dataset

### Option 1: Using Simulation Tools (ALOHA, PHAST)

If you're generating data from simulation software:

1. **Export results** from ALOHA/PHAST to CSV or Excel
2. **Rename columns** to match the required format exactly
3. **Verify units** match the specifications
4. **Check for missing values**
5. **Ensure numeric format** (no text, no special characters)

### Option 2: Experimental Data

If using experimental measurements:

1. **Collect data** with appropriate sensors
2. **Record** all input conditions (temp, wind, hole size)
3. **Measure** concentrations and threat zones
4. **Format** in Excel according to specifications
5. **Quality check** for outliers and errors

### Option 3: Combined Dataset

You can combine data from multiple sources:

1. Ensure **consistent units**
2. **Normalize** measurement methods
3. **Document** data sources
4. **Remove** duplicates
5. **Validate** data quality

## Data Quality Checklist

Before using your data, verify:

- [ ] All column names match exactly (copy-paste from this guide)
- [ ] No missing values (NaN, blank cells)
- [ ] All values are numeric (no text)
- [ ] Reasonable value ranges (no negative distances, etc.)
- [ ] At least 20 data points
- [ ] File saved as .xlsx format
- [ ] Sheet name is "Sheet1"
- [ ] No extra columns or rows before data

## Preprocessing (Automatic in Code)

The model automatically handles:

- ✅ **Scaling**: MinMax scaling to [0, 1]
- ✅ **Train-test split**: 80-20 split
- ✅ **Validation split**: 20% of training data

You do NOT need to:
- ❌ Scale/normalize data manually
- ❌ Split data manually
- ❌ Encode variables (all numeric)

## Example Python Code to Check Your Data

```python
import pandas as pd

# Load your data
data = pd.read_excel("unique_points.xlsx", sheet_name="Sheet1")

# Check structure
print("Shape:", data.shape)
print("\nColumns:")
print(data.columns.tolist())

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())

# Check data types
print("\nData types:")
print(data.dtypes)

# View first few rows
print("\nFirst 5 rows:")
print(data.head())

# Basic statistics
print("\nStatistics:")
print(data.describe())

# Check required columns
required_cols = [
    "Air Temperature (°C)",
    "Wind Speed (m/s)",
    "Hole Diameter (mm)",
    "Indoor Concentration (ppm)",
    "Outdoor Concentraton (ppm)",
    "Red Zone (miles) ",
    "Orange Zone (miles) ",
    "Yellow Zone (miles)"
]

missing_cols = [col for col in required_cols if col not in data.columns]
if missing_cols:
    print("\n⚠️ Missing columns:")
    print(missing_cols)
else:
    print("\n✅ All required columns present!")
```

## Common Issues & Solutions

### Issue 1: "KeyError: 'Column Name'"

**Problem**: Column name doesn't match exactly  
**Solution**: Copy column names from this guide exactly (including spaces)

### Issue 2: "ValueError: could not convert string to float"

**Problem**: Non-numeric data in columns  
**Solution**: Remove text, replace with numeric values

### Issue 3: Model performs poorly

**Problem**: Insufficient or poor quality data  
**Solution**: 
- Collect more data points (100+)
- Check for outliers
- Verify measurement accuracy

### Issue 4: "FileNotFoundError"

**Problem**: Data file not in correct location  
**Solution**: 
- Place `unique_points.xlsx` in same folder as notebook
- Or update file path in code

## Converting Your Data

### From CSV:

```python
import pandas as pd

# Read CSV
data = pd.read_csv("your_data.csv")

# Rename columns to match required format
data.columns = [
    "Air Temperature (°C)",
    "Wind Speed (m/s)",
    "Hole Diameter (mm)",
    "Indoor Concentration (ppm)",
    "Outdoor Concentraton (ppm)",
    "Red Zone (miles) ",
    "Orange Zone (miles) ",
    "Yellow Zone (miles)"
]

# Save as Excel
data.to_excel("unique_points.xlsx", sheet_name="Sheet1", index=False)
```

### From Multiple Files:

```python
import pandas as pd

# Read multiple files
data1 = pd.read_excel("file1.xlsx")
data2 = pd.read_excel("file2.xlsx")

# Combine
combined = pd.concat([data1, data2], ignore_index=True)

# Remove duplicates
combined = combined.drop_duplicates()

# Save
combined.to_excel("unique_points.xlsx", sheet_name="Sheet1", index=False)
```

## Template File

Create a template with proper column names:

```python
import pandas as pd

# Create empty template
template = pd.DataFrame(columns=[
    "Air Temperature (°C)",
    "Wind Speed (m/s)",
    "Hole Diameter (mm)",
    "Indoor Concentration (ppm)",
    "Outdoor Concentraton (ppm)",
    "Red Zone (miles) ",
    "Orange Zone (miles) ",
    "Yellow Zone (miles)"
])

# Save template
template.to_excel("data_template.xlsx", sheet_name="Sheet1", index=False)
print("Template created: data_template.xlsx")
```

## Privacy & Sharing

If your data is sensitive:

1. **Anonymize**: Remove identifying information
2. **Aggregate**: Use average values
3. **Synthetic**: Generate similar synthetic data
4. **Sample**: Share a small representative sample
5. **Document**: Provide data collection methodology instead

## Need Help?

- Check that your Excel file matches the format exactly
- Use the validation code above
- Open an issue on GitHub if you encounter problems
- Contact: [your email]

---

**Ready to prepare your data? Use the template code above to get started!** 📊
