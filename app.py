from flask import Flask, render_template, request
import joblib
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# --- Load the trained model and other artifacts ---
model = joblib.load('launch_price_model.joblib')
encoder = joblib.load('one_hot_encoder.joblib')
features = joblib.load('feature_columns.joblib')
unique_values = joblib.load('unique_values.joblib')

# --- Define Routes ---

# Main route to display the HTML form
@app.route('/')
def home():
    return render_template('index.html', unique_values=unique_values)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        input_data = {
            'Organisation': request.form['organisation'],
            'Location': request.form['location'],
            'Rocket_Status': request.form['rocket_status'],
            'Mission_Status': request.form['mission_status']
        }

        # Convert input to a DataFrame
        input_df = pd.DataFrame([input_data])

        # Apply the same One-Hot Encoding
        input_encoded = encoder.transform(input_df[features])

        # Make a prediction
        prediction = model.predict(input_encoded)

        # Format the prediction result
        predicted_price = f"${prediction[0]:,.2f} Million"

        # Render the page again with the prediction result
        return render_template('index.html',
                               unique_values=unique_values,
                               prediction_text=predicted_price,
                               form_data=input_data) # To keep selected values
    return render_template('index.html', unique_values=unique_values)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)