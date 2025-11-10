# Ammonia Dispersion Prediction using Deep Learning

A machine learning project for predicting ammonia concentration and threat zones based on environmental conditions using deep neural networks.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Overview

This project uses deep learning to predict ammonia dispersion patterns and threat zones for safety planning and risk assessment. The model predicts:
- Indoor and outdoor ammonia concentrations (ppm)
- Threat zone distances (Red, Orange, and Yellow zones in miles)

## 🎯 Features

- **Multi-output Prediction**: Simultaneously predicts 5 different outputs
- **Hyperparameter Optimization**: Uses Keras Tuner for automatic hyperparameter search
- **Comprehensive Visualization**: Includes loss curves, prediction plots, and residual analysis
- **Performance Metrics**: Calculates RMSE, R², MAE, and MAPE for each output
- **Easy to Use**: Jupyter notebook with clear documentation

## 📊 Input Features

- Air Temperature (°C)
- Wind Speed (m/s)
- Hole Diameter (mm)

## 🎯 Output Variables

1. Indoor Concentration (ppm)
2. Outdoor Concentration (ppm)
3. Red Zone Distance (miles)
4. Orange Zone Distance (miles)
5. Yellow Zone Distance (miles)

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone this repository:
```bash
git clone https://github.com/Zohaib-Atiq/ammonia-dispersion-prediction.git
cd ammonia-dispersion-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Required Packages

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
keras-tuner>=1.1.0
tabulate>=0.8.9
openpyxl>=3.0.9
```

## 📖 Usage

### Running the Jupyter Notebook

1. Ensure your data file `unique_points.xlsx` is in the same directory
2. Launch Jupyter Notebook:
```bash
jupyter notebook ammonia_dispersion_ml_model.ipynb
```
3. Run cells sequentially to:
   - Load and preprocess data
   - Train the model
   - Generate predictions
   - Visualize results

### Using the Python Script

```bash
python ammonia_dispersion_ml_model_code.py
```

### Making Predictions on New Data

```python
import numpy as np
import pickle
from tensorflow import keras

# Load the trained model
model = keras.models.load_model('ammonia_dispersion_model.keras')

# Load scalers
with open('scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open('scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

# Prepare new data
new_data = np.array([[25.0, 5.0, 10.0]])  # [Air Temp, Wind Speed, Hole Diameter]
new_data_scaled = scaler_X.transform(new_data)

# Predict
predictions_scaled = model.predict(new_data_scaled)
predictions = scaler_y.inverse_transform(predictions_scaled)

print("Predictions:", predictions)
```

## 📁 Project Structure

```
.
├── ammonia_dispersion_ml_model.ipynb  # Main Jupyter notebook
├── ammonia_dispersion_ml_model_code.py  # Python script version
├── unique_points.xlsx                 # Dataset
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore file
└── figures/                           # Generated figures (created during runtime)
```

## 🔬 Methodology

### Model Architecture

The neural network uses:
- 3 hidden layers with ReLU activation
- Dropout layers for regularization
- Adam optimizer with learning rate ~0.00048
- Mean Squared Error (MSE) loss function

### Training Process

1. **Data Preprocessing**: MinMax scaling of features and targets
2. **Train-Test Split**: 80-20 split with random_state=42
3. **Hyperparameter Tuning**: RandomSearch with 10 trials
4. **Early Stopping**: Monitors validation loss with patience=10
5. **Model Evaluation**: Comprehensive metrics on test set

## 📊 Results

The model achieves strong predictive performance across all outputs with high R² scores and low error metrics. Detailed results are shown in the notebook.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Citation

The article is under revision and citation will be available soon:



## 👥 Authors

- **Dr. Zohaib Atiq Khan** - First Author - University of Engineering and Technology (UET Lahore), Pakistan
- **Dr. Imran Rashid** - Corresponding Author - University of Engineering and Technology (UET Lahore), Pakistan

## 📧 Contact

- Email: zohaib_atiq@uet.edu.pk
- LinkedIn: [Your LinkedIn](https://www.linkedin.com/in/dr-zohaib-atiq-khan/)
- Google Scholar: [Your Profile](https://scholar.google.ca/citations?user=NLQyafEAAAAJ&hl=en)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Shamim Irshad High Performance Computing Center
- Department of Chemical, Polymer, and Composite Materials Engineering at the University of Engineering and Technology Lahore, Pakistan
- Contributors and reviewers
- TensorFlow and scikit-learn communities

## 📚 References

Will be provided soon

## 🐛 Known Issues

- Currently optimized for the specific dataset format
- Requires specific column names in input Excel file

## 🔮 Future Enhancements

- [ ] Web application for real-time predictions
- [ ] Support for additional environmental factors
- [ ] Multi-chemical substance prediction
- [ ] Real-time data integration
- [ ] API deployment
- [ ] Docker containerization

## 📈 Version History

- **v1.0.0** (2025-11-10): Initial release
  - Core prediction functionality
  - Jupyter notebook implementation
  - Comprehensive visualization

---

**⭐ If you find this project helpful, please consider giving it a star!**

