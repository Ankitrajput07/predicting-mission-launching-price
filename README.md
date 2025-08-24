🚀 Rocket Launch Price Predictor
A web application that predicts the cost of a space mission using a machine learning model. This project demonstrates a full-stack approach, from data cleaning and model training with Scikit-learn to deploying the model with a Flask backend and a simple HTML/CSS frontend.

✨ Features
Interactive Web Interface: A clean and user-friendly UI to select launch parameters.

Dynamic Dropdowns: All selection options are dynamically loaded from the dataset.

Real-time Prediction: Get an estimated launch price instantly based on your inputs.

End-to-End ML Pipeline: Includes scripts for data cleaning, feature engineering, and model training.

🛠️ Tech Stack
Backend: Flask

Machine Learning: Scikit-learn, Pandas, NumPy

Frontend: HTML, CSS

Model: Linear Regression

Deployment: Local Flask Server

⚙️ How It Works
The project is divided into two main components:

Model Training (model_training.py):

The mission_launches.csv dataset is loaded and cleaned. This involves handling missing values, removing unnecessary columns, and converting the Price column to a numerical format.

Categorical features like Organisation, Location, Rocket_Status, and Mission_Status are converted into a numerical format using One-Hot Encoding.

A Linear Regression model is trained on the processed data.

The trained model, the encoder, and a list of unique values for the dropdowns are saved as .joblib files.

Web Application (app.py):

The Flask server loads the pre-trained model and encoder files.

It serves the index.html page, populating the dropdown menus with the unique values saved during training.

When a user submits the form, the server takes the inputs, uses the saved encoder to transform them into the correct numerical format, and feeds them to the model.

The model returns a price prediction, which is formatted and displayed back to the user on the webpage.

🚀 Installation & Usage
To run this project locally, follow these steps:

Clone the repository:

Bash

git clone https://github.com/Ankitrajput07/prediction-mission-launching.git
cd your-repo-name
Create a virtual environment (recommended):

Bash

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required libraries:

Bash

pip install -r requirements.txt
(Note: You will need to create a requirements.txt file by running pip freeze > requirements.txt after installing Flask, Scikit-learn, Pandas, and Joblib).

Train the model:
Run the training script to generate the necessary .joblib files.

Bash

python model_training.py
Run the Flask application:

Bash

python app.py
Open the application:
Navigate to http://127.0.0.1:5000 in your web browser.

⚠️ Model & Limitations
This project currently uses a Linear Regression model as a baseline.

Known Issue: The model can sometimes predict negative prices. This is a common limitation of Linear Regression, as it fits a straight line to the data and does not understand real-world constraints (i.e., that price cannot be negative).

Future Improvements: To improve accuracy and resolve the negative prediction issue, the model could be upgraded to a more robust algorithm like a Random Forest Regressor or Gradient Boosting.
