# House Price Prediction Dashboard 🏠

## Project Overview
This project is an interactive web-based application designed to predict residential house prices using Machine Learning. It provides a user-friendly interface built with **Streamlit**, allowing users to input property details and receive instant market value estimations based on pre-trained regression models.

## Core Features
- **Real-time Prediction:** Get house price estimates instantly.
- **Multiple Model Comparison:** The app supports switching between different trained algorithms (Linear, Gradient Boosting, and XGBoost).
- **Interactive UI:** A clean and simple dashboard for data entry and result visualization.

## Project Structure (Files)
To run this project, ensure all the following files are located in the same directory:
- `app.py`: The main Streamlit application code.
- `ML PRO.py`: Source script for data analysis and model training.
- `house_data.csv`: The primary dataset used for this project.
- `house_price_model_Linear.pkl`: Pre-trained Linear Regression model.
- `house_price_model_Gradient.pkl`: Pre-trained Gradient Boosting model.
- `house_price_model_xg_r.pkl`: Pre-trained XGBoost Regressor model.

## Methodology
1. **Data Engineering:** Cleaned the raw housing data and handled missing values.
2. **Analysis:** Conducted Exploratory Data Analysis (EDA) to find correlations between features (like area and location) and price.
3. **Model Development:** Trained several regression models to minimize the Mean Absolute Error (MAE).
4. **Serialization:** Saved the best-performing models as `.pkl` files using the `joblib` / `pickle` library for deployment.

## Installation & Usage
1. **Prepare Environment:**
   Install the required Python libraries:
   ```bash
   pip install pandas scikit-learn xgboost streamlit joblib
   Run the Application:
Open your terminal/command prompt, navigate to the project folder, and execute:
streamlit run app.py

   
