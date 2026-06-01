"""
Disease Predictor Flask App — FULLY INTEGRATED
Dengue: uses classifier (risk level) + regressor (case count) from v2 pipeline
Malaria: uses LightGBM regression model
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import warnings
import traceback

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ============================================================================
# LOAD ALL MODELS
# ============================================================================
print("\n" + "="*60)
print("  LOADING MODELS")
print("="*60)

# Dengue (v2 pipeline)
dengue_clf    = None
dengue_reg    = None
dengue_scaler = None

try:
    dengue_clf    = joblib.load("dengue_classifier.pkl")
    dengue_reg    = joblib.load("dengue_regressor.pkl")
    dengue_scaler = joblib.load("dengue_scaler.pkl")
    print("✓ Dengue classifier, regressor, scaler loaded")
except Exception as e:
    print(f"✗ Dengue model error: {e}")

# Malaria
malaria_model    = None
malaria_scaler   = None
malaria_features = None

try:
    malaria_model    = joblib.load("malaria_regressor.pkl")
    malaria_scaler   = joblib.load("malaria_scaler.pkl")
    malaria_features = joblib.load("malaria_feature_cols.pkl")
    print(f"✓ Malaria model loaded ({len(malaria_features)} features)")
except Exception as e:
    print(f"✗ Malaria model error: {e}")

print("="*60 + "\n")


# ============================================================================
# DENGUE FEATURE BUILDER — matches dengue_refined_v2.py exactly
# ============================================================================

DENGUE_FEATURES = [
    'Month_sin', 'Month_cos',
    'Is_Monsoon', 'Is_Peak',
    'Temp', 'Rainfall', 'Humidity',
    'Lag1', 'Lag2', 'Lag3',
    'Roll3_mean', 'Roll3_max',
    'Year_idx'
]

# Historical monthly case averages from your cleaned_dengue_data.csv
MONTHLY_CASE_AVG = {
    1: 21, 2: 21, 3: 16, 4: 12, 5: 18,
    6: 53, 7: 121, 8: 175, 9: 168,
    10: 142, 11: 155, 12: 95
}

def build_dengue_features(temp, rainfall, month, year=2025):
    month = int(month)
    year  = int(year)

    month_sin  = np.sin(2 * np.pi * month / 12)
    month_cos  = np.cos(2 * np.pi * month / 12)
    is_monsoon = int(month in [6, 7, 8, 9, 10])
    is_peak    = int(month in [7, 8, 9])
    humidity   = 65 + 20 * np.sin((month - 5) * np.pi / 6)

    prev1 = (month - 2) % 12 + 1
    prev2 = (month - 3) % 12 + 1
    prev3 = (month - 4) % 12 + 1

    lag1 = MONTHLY_CASE_AVG.get(prev1, 50)
    lag2 = MONTHLY_CASE_AVG.get(prev2, 50)
    lag3 = MONTHLY_CASE_AVG.get(prev3, 50)

    roll3_mean = np.mean([lag1, lag2, lag3])
    roll3_max  = max(lag1, lag2, lag3)
    year_idx   = year - 2021

    return np.array([[
        month_sin, month_cos,
        is_monsoon, is_peak,
        float(temp), float(rainfall), humidity,
        lag1, lag2, lag3,
        roll3_mean, roll3_max,
        year_idx
    ]])


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict/rainy')
def predict_rainy():
    return render_template('dashboard.html')


@app.route('/season/summer')
def season_summer():
    return render_template('summer.html')


@app.route('/season/winter')
def season_winter():
    return render_template('winter.html')

@app.route('/predict/<disease>', methods=['GET', 'POST'])
def predict(disease):
    return render_template('dashboard.html', disease=disease)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data     = request.json
        disease  = data.get('disease', '').lower()
        temp     = float(data.get('temp', 28))
        rainfall = float(data.get('rainfall', 150))
        month    = int(data.get('month', 6))
        year     = int(data.get('year', 2025))

        print(f"\n📊 {disease.upper()} | Temp:{temp} Rainfall:{rainfall} Month:{month}")

        # Validation
        if disease not in ['dengue', 'malaria']:
            return jsonify({'success': False, 'error': 'Invalid disease'}), 400
        if not (0 <= temp <= 50):
            return jsonify({'success': False, 'error': 'Temperature must be 0-50°C'}), 400
        if not (0 <= rainfall <= 1000):
            return jsonify({'success': False, 'error': 'Rainfall must be 0-1000mm'}), 400
        if not (1 <= month <= 12):
            return jsonify({'success': False, 'error': 'Month must be 1-12'}), 400

        # ── Dengue ──────────────────────────────────────────
        if disease == 'dengue':
            if None in [dengue_clf, dengue_reg, dengue_scaler]:
                return jsonify({
                    'success': False,
                    'error': 'Dengue models not found. Place dengue_classifier.pkl, '
                             'dengue_regressor.pkl, dengue_scaler.pkl in project folder.'
                }), 500

            feat    = build_dengue_features(temp, rainfall, month, year)
            feat_sc = dengue_scaler.transform(feat)

            risk_idx   = int(dengue_clf.predict(feat_sc)[0])
            risk_level = {0: 'LOW', 1: 'MODERATE', 2: 'HIGH'}.get(risk_idx, 'LOW')
            case_pred  = max(0.0, float(dengue_reg.predict(feat_sc)[0]))

        # ── Malaria ─────────────────────────────────────────
        else:
            if malaria_model is None:
                return jsonify({'success': False, 'error': 'Malaria model not loaded'}), 500

            case_pred  = _predict_malaria(temp, rainfall, month)
            risk_level = _cases_to_risk(case_pred)

        # For malaria: risk already set by _cases_to_risk (100/250 thresholds)
        # For dengue: apply small override for very extreme values
        if disease == 'dengue':
            if case_pred < 20:
                risk_level = 'LOW'

        risk_color = {'LOW': '#39ff14', 'MODERATE': '#ffeb3b', 'HIGH': '#ff3131'}.get(risk_level, '#ffeb3b')
        gauge_pct  = {'LOW': 12, 'MODERATE': 50, 'HIGH': 88}.get(risk_level, 50)

        print(f"   ✓ Risk: {risk_level} | Cases: {case_pred:.1f} | Gauge: {gauge_pct}%")

        return jsonify({
            'success'   : True,
            'prediction': round(case_pred, 1),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'gauge_pct' : gauge_pct,
            'message'   : f'Forecasted case load: {round(case_pred)} cases'
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# HELPERS
# ============================================================================

def _cases_to_risk(cases):
    # Malaria thresholds: LOW<100, MODERATE 100-250, HIGH>250
    if cases < 100:   return 'LOW'
    elif cases < 250: return 'MODERATE'
    else:             return 'HIGH'


def _predict_malaria(temp, rainfall, month):
    try:
        fd = {}
        # CRITICAL: Training data uses mm/day (0.2–12 range), NOT mm/month
        # User enters mm/month (0-1000), so divide by 30 to convert
        rainfall_day = np.clip(rainfall / 30, 0, 15)
        fd['Rainfall']    = rainfall_day
        fd['Temperature'] = temp
        fd['Month']       = month
        fd['Monsoon']     = 1 if month in [6,7,8,9] else 0

        if month in [1,2,12]:   fd['Season'] = 0
        elif month in [3,4,5]:  fd['Season'] = 1
        elif month in [6,7,8,9]:fd['Season'] = 2
        else:                   fd['Season'] = 3

        fd['Rainfall_Lag1']    = rainfall_day * 0.95
        fd['Rainfall_Lag2']    = rainfall_day * 0.90
        fd['Temperature_Lag1'] = temp * 0.98
        fd['Pf']               = rainfall_day * 1.5
        fd['TPR']              = 0.5 * (rainfall_day / 8)
        fd['Pf_Lag1']          = fd['Pf'] * 0.95
        fd['Pf_Ratio']         = fd['Pf'] / max(1, fd['Pf'] + 10)

        # Base cases: proportional to temp & rainfall, no hardcoded offset
        temp_factor  = max(0.0, (temp - 10) / 20.0)   # 0.0 at <=10C, 1.0 at 30C
        rain_factor  = min(rainfall_day / 5.0, 1.0)   # 0.0 at 0mm, 1.0 at 5+mm/day
        season_boost = 1.5 if fd['Monsoon'] == 1 else 0.8
        base = 600 * temp_factor * rain_factor * season_boost
        fd['Cases_Lag1']         = base * 0.95
        fd['Cases_Lag2']         = base * 0.90
        fd['Cases_Lag3']         = base * 0.85
        fd['Cases_RollMean3']    = base * 0.98
        fd['Cases_RollMean6']    = base * 0.95
        fd['Rainfall_RollMean3'] = rainfall_day * 0.97
        fd['Temp_RollMean3']     = temp * 0.99
        fd['Rainfall_Change']    = rainfall_day * 0.05
        fd['Cases_Change']       = base * 0.05
        fd['Temp_Change']        = temp * 0.02
        fd['Rainfall_Season_Interaction'] = rainfall_day * (fd['Season'] + 1)
        fd['Temp_Season_Interaction']     = temp * (fd['Season'] + 1)

        features = np.array([[fd.get(col, 0) for col in malaria_features]])
        if malaria_scaler:
            features = malaria_scaler.transform(features)

        return float(max(0, min(500, malaria_model.predict(features)[0])))

    except Exception as e:
        print(f"Malaria error: {e}")
        return _fallback_malaria(temp, rainfall, month)


def _fallback_malaria(temp, rainfall, month):
    base     = 150 if 25 <= temp < 30 else 120 if 30 <= temp < 35 else 80 if 20 <= temp < 25 else 40
    rain_f   = 1.5 if 200 <= rainfall < 300 else 1.2 if 100 <= rainfall < 200 else 0.8 if 50 <= rainfall < 100 else 0.5
    season_f = 1.8 if month in [6,7,8,9] else 1.3 if month in [5,10] else 0.6
    return float(max(0, min(500, base * rain_f * season_f)))


# ============================================================================
# RUN
# ============================================================================

if __name__ == '__main__':
    print("Server: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader = False)