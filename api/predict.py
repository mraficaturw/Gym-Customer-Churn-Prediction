from flask import Flask, request, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__, 
           template_folder=os.path.join(os.path.dirname(__file__), '../templates'),
           static_folder=os.path.join(os.path.dirname(__file__), '../static'))

# Load model
model_path = os.path.join(os.path.dirname(__file__), '../model.joblib')
pipeline = joblib.load(model_path)

# Feature configuration
ORIGINAL_FEATURES = [
    'gender',
    'Near_Location',
    'Partner',
    'Promo_friends',
    'Phone',
    'Contract_period',
    'Group_visits',
    'Age',
    'Avg_additional_charges_total',
    'Month_to_end_contract',
    'Lifetime',
    'Avg_class_frequency_total',
    'Avg_class_frequency_current_month'
]

REQUIRED_NUMERIC_FEATURES = [
    'Contract_period',
    'Age',
    'Avg_additional_charges_total',
    'Month_to_end_contract',
    'Lifetime',
    'Avg_class_frequency_total',
    'Avg_class_frequency_current_month'
]

BINARY_FEATURES = [
    'Near_Location',
    'Partner',
    'Promo_friends',
    'Phone',
    'Group_visits'
]

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate required numeric fields
        missing = [f for f in REQUIRED_NUMERIC_FEATURES if f not in request.form]
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")

        input_data = {}
        
        # Process all features
        for feature in ORIGINAL_FEATURES:
            if feature == 'gender':
                gender = request.form.get('gender', 'Male')
                input_data[feature] = [1 if gender == 'Male' else 0]
            
            elif feature in BINARY_FEATURES:
                # Default to 0 if not submitted
                input_data[feature] = [1 if request.form.get(feature) == 'on' else 0]
            
            else:
                # Get value with default 0 for non-required features
                value = request.form.get(feature, '0')
                if not value.replace('.', '', 1).isdigit():
                    raise ValueError(f"Invalid value for {feature}")
                input_data[feature] = [float(value)]

        # Create DataFrame with original feature order
        input_df = pd.DataFrame(input_data, columns=ORIGINAL_FEATURES)
        
        # Make prediction
        prob = pipeline.predict_proba(input_df)[0][1]
        
        # Determine risk level
        if prob > 0.7:
            risk = "High"
        elif prob > 0.5:
            risk = "Medium"
        else:
            risk = "Low"

        return render_template(
            'result.html',
            prediction=round(prob * 100, 2),
            risk_level=risk
        )
    
    except Exception as e:
        return render_template('error.html', error=str(e))

def vercel_handler(request):
    with app.app_context():
        return app.full_dispatch_request(request)

if __name__ == '__main__':
    app.run(debug=True)
