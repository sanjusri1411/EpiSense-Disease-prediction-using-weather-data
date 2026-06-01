# 🦟 EpiSense - Disease Predictor

A Flask-based web application for predicting dengue and malaria cases using LightGBM machine learning models.

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## 🚀 Setup Instructions

### Step 1: Install Dependencies

Open your command prompt/terminal and navigate to the project folder:

```bash
cd DiseasePredictor
```

Install all required packages:

```bash
pip install -r requirements.txt
```

**Important:** Make sure to wait for all packages to install completely. This may take 2-3 minutes.

### Step 2: Verify Models Are Present

Check that these files exist in the project folder:
- ✓ `model.pkl` (Dengue model - 163 KB)
- ✓ `malaria_model.pkl` (Malaria model - 3.6 MB)
- ✓ `app.py`
- ✓ `templates/` folder
- ✓ `static/` folder

### Step 3: Run the Application

```bash
python app.py
```

You should see output like:
```
 * Serving Flask app 'app'
 * Running on http://127.0.0.1:5000
```

### Step 4: Access the Web Interface

Open your web browser and go to:
```
http://127.0.0.1:5000
```

## 🔧 Troubleshooting

### Issue 1: "ModuleNotFoundError: No module named 'lightgbm'"

**Solution:** Install missing package:
```bash
pip install lightgbm
```

### Issue 2: "ModuleNotFoundError: No module named 'statsmodels'"

**Solution:** Install missing package:
```bash
pip install statsmodels
```

### Issue 3: "ModuleNotFoundError: No module named 'flask'"

**Solution:** Install Flask and dependencies:
```bash
pip install -r requirements.txt
```

### Issue 4: Port 5000 already in use

**Solution:** Change the port in `app.py`:
```python
if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Change 5001 to any available port
```

### Issue 5: Models not loading

**Solution:** 
1. Verify `model.pkl` and `malaria_model.pkl` are in the correct folder
2. Check file sizes match: 
   - `model.pkl`: ~163 KB
   - `malaria_model.pkl`: ~3.6 MB
3. Run with verbose output to see which model fails

## 📊 Models Used

Both models use **LightGBM** (Light Gradient Boosting Machine):
- **Dengue Model**: Trained on 6 environmental features
- **Malaria Model**: Trained on 14 environmental features

## 🎯 How to Use

1. Select **DENGUE** or **MALARIA** from the home page
2. Enter environmental parameters:
   - **Temperature** (°C): Ambient temperature
   - **Rainfall** (mm): Monthly rainfall
   - **Month** (1-12): Month number
3. Click "RUN PREDICTION"
4. View the forecasted case load and risk status

## 📁 Project Structure

```
DiseasePredictor/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── model.pkl                 # Dengue LightGBM model
├── malaria_model.pkl         # Malaria LightGBM model
├── templates/
│   ├── index.html           # Home page
│   └── dashboard.html       # Prediction interface
└── static/
    └── images/              # Analysis charts
        ├── dengue/
        └── malaria/
```

## 🛠️ System Requirements

- RAM: 2+ GB recommended
- Disk Space: 500 MB
- Internet: Not required after initial setup

## 📝 Notes

- Models are loaded on startup. Check console for loading status.
- Predictions take 1-2 seconds to compute
- Historical data visualizations are included for reference
- All data is processed locally; nothing is sent to external servers

## 🔗 Dependencies

- **Flask**: Web framework
- **joblib**: Model serialization
- **LightGBM**: Machine learning model
- **NumPy/Pandas**: Data processing
- **scikit-learn**: ML utilities
- **statsmodels**: Statistical analysis

---

For issues or questions, ensure all dependencies are installed and try clearing your browser cache.
