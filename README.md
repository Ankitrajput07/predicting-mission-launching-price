# 🚀 Rocket Launch Price Predictor

A web application that predicts the cost of a space mission using a machine learning model. This project demonstrates a full-stack approach, from data cleaning and model training with Scikit-learn to deploying the model with a Flask backend and a simple HTML/CSS frontend.

![Image of the rocket launch price predictor UI]

---

## ✨ Features

-   **Interactive Web Interface**: A clean and user-friendly UI to select launch parameters.
-   **Dynamic Dropdowns**: All selection options are dynamically loaded from the dataset.
-   **Real-time Prediction**: Get an estimated launch price instantly based on your inputs.
-   **End-to-End ML Pipeline**: Includes scripts for data cleaning, feature engineering, and model training.

---

## 🛠️ Tech Stack

-   **Backend**: Flask
-   **Machine Learning**: Scikit-learn, Pandas, NumPy
-   **Frontend**: HTML, CSS
-   **Model**: Linear Regression

---

## ⚙️ How It Works

The project is divided into two main components:

### 1. Model Training (`model_training.py`)

-   The `mission_launches.csv` dataset is loaded and cleaned. This involves handling missing values, removing unnecessary columns, and converting the `Price` column to a numerical format.
-   Categorical features like `Organisation`, `Location`, `Rocket_Status`, and `Mission_Status` are converted into a numerical format using One-Hot Encoding.
-   A **Linear Regression** model is trained on the processed data.
-   The trained model, the encoder, and a list of unique values for the dropdowns are saved as `.joblib` files.

### 2. Web Application (`app.py`)

-   The Flask server loads the pre-trained model and encoder files.
-   It serves the `index.html` page, populating the dropdown menus with the unique values saved during training.
-   When a user submits the form, the server takes the inputs, uses the saved encoder to transform them into the correct numerical format, and feeds them to the model.
-   The model returns a price prediction, which is formatted and displayed back to the user on the webpage.

---

## 🚀 Installation & Usage

To run this project locally, follow these steps:

**1. Clone the repository:**
```bash
git clone [https://github.com/Ankitrajput07/prediction-mission-launching.git](https://github.com/Ankitrajput07/prediction-mission-launching.git)
cd prediction-mission-launching
```

**2. Create a virtual environment (recommended):**
```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
