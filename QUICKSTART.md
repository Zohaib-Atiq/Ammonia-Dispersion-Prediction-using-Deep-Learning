# 🚀 Quick Start Guide

Get up and running with the Ammonia Dispersion Prediction model in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Jupyter Notebook (optional, for notebook interface)

## Installation Steps

### 1. Clone or Download the Repository

**Option A: Using Git**
```bash
git clone https://github.com/YOUR_USERNAME/ammonia-dispersion-prediction.git
cd ammonia-dispersion-prediction
```

**Option B: Download ZIP**
- Download the repository as ZIP from GitHub
- Extract to your desired location
- Open terminal/command prompt in that folder

### 2. Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- numpy
- pandas
- matplotlib
- scikit-learn
- tensorflow
- keras-tuner
- tabulate
- openpyxl
- jupyter

### 4. Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

## Running the Project

### Method 1: Jupyter Notebook (Recommended for Beginners)

1. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Open the notebook:**
   - Navigate to `ammonia_dispersion_ml_model.ipynb`
   - Click to open

3. **Run the notebook:**
   - Click "Cell" → "Run All"
   - Or run cells one by one with Shift+Enter

### Method 2: Python Script

```bash
python DrImran_paper4_machinelearning_v2.py
```

### Method 3: Google Colab (No Installation Required!)

1. Go to [Google Colab](https://colab.research.google.com)
2. File → Open Notebook → GitHub tab
3. Enter repository URL
4. Select `ammonia_dispersion_ml_model.ipynb`
5. Run cells!

## 📁 Required Data

Make sure you have `unique_points.xlsx` in the same directory as the notebook. This file should contain:

**Columns:**
- Air Temperature (°C)
- Wind Speed (m/s)
- Hole Diameter (mm)
- Indoor Concentration (ppm)
- Outdoor Concentraton (ppm)
- Red Zone (miles)
- Orange Zone (miles)
- Yellow Zone (miles)

## 🎯 What You'll Get

After running the notebook, you'll have:

1. **Trained Model**: Saved as `ammonia_dispersion_model.keras`
2. **Scalers**: `scaler_X.pkl` and `scaler_y.pkl`
3. **Visualizations**:
   - Training loss curves
   - Predicted vs Actual plots
   - Threat zone comparisons
   - Residual plots
4. **Performance Metrics**: RMSE, R², MAE, MAPE for each output
5. **Summary Table**: `combined_input_output_summary.xlsx`

## 🔧 Common Issues & Solutions

### Issue 1: TensorFlow Installation Failed

**Solution:**
```bash
pip install --upgrade pip
pip install tensorflow --no-cache-dir
```

For Mac M1/M2:
```bash
pip install tensorflow-macos
pip install tensorflow-metal
```

### Issue 2: "No module named 'keras_tuner'"

**Solution:**
```bash
pip install keras-tuner
```

### Issue 3: "File not found: unique_points.xlsx"

**Solution:**
- Make sure the Excel file is in the same directory as the notebook
- Or update the file path in the notebook

### Issue 4: Jupyter Notebook Not Opening

**Solution:**
```bash
pip install --upgrade jupyter
jupyter notebook --generate-config
```

### Issue 5: GPU Not Detected (Optional)

Check GPU availability:
```python
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

For NVIDIA GPUs, install CUDA and cuDNN.

## 📊 Quick Test

To quickly test if everything works:

```python
import numpy as np
import pandas as pd
import tensorflow as tf

print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
```

## 🎓 Next Steps

1. **Explore the Notebook**: Read through the markdown cells
2. **Modify Parameters**: Try different hyperparameters
3. **Use Your Data**: Replace `unique_points.xlsx` with your dataset
4. **Make Predictions**: Use the trained model for new predictions
5. **Contribute**: Found a bug or improvement? Open an issue or PR!

## 📖 Additional Resources

- **Full Documentation**: See [README.md](README.md)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)
- **Open Source Guide**: See [OPEN_SOURCE_GUIDE.md](OPEN_SOURCE_GUIDE.md)
- **Issues**: Report bugs or request features on GitHub Issues

## 💡 Tips

1. **Start Simple**: Run the notebook with default settings first
2. **Save Your Work**: Save the notebook frequently (Ctrl+S)
3. **Experiment**: Try changing hyperparameters in the tuning section
4. **Visualize**: The plots help understand model performance
5. **Document**: Add your own markdown cells to document experiments

## 🆘 Need Help?

- **GitHub Issues**: Open an issue for bugs or questions
- **Email**: contact@yourdomain.com
- **Documentation**: Check the full README.md

## ⚡ Speed Tips

1. **Skip Hyperparameter Tuning**: Set `run_hyperparameter_tuning = False` (default)
2. **Reduce Epochs**: Change `epochs=200` to `epochs=50` for faster training
3. **Use CPU**: Sufficient for this dataset size
4. **Batch Processing**: Increase `batch_size=32` to `batch_size=64`

## ✅ Verification Checklist

Before reporting issues, verify:
- [ ] Python 3.8+ is installed
- [ ] All dependencies are installed (`pip list`)
- [ ] Data file exists in correct location
- [ ] Virtual environment is activated
- [ ] Latest code from repository

---

**Ready to predict ammonia dispersion? Let's go! 🎉**

For detailed documentation, see [README.md](README.md)
