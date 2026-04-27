import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import datetime

st.set_page_config(
    page_title="House Price Prediction 2025",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    div.stButton > button:first-child { width: 100%; }
    .main { text-align: left; }
    </style>
    """, unsafe_allow_html=True)

# 2. Model Loading
@st.cache_resource
def load_models(file_name):
    try:
        with open(file_name, 'rb') as file:
            models = pickle.load(file)
        return models
    except Exception as e:
        st.sidebar.error(f"❌ Error loading models: {e}")
        st.stop()

MODEL_FILE = 'ml.pkl'
loaded_models = load_models(MODEL_FILE)

# 3. Main Interface
st.title("🏡 House Price Prediction - Model Comparison")
st.info("Input property features in the sidebar to get price predictions for 2025.")

# --- Sidebar Inputs ---
st.sidebar.header("📊 Property Features (18 Features)")

with st.sidebar:
    current_year = 2025
    
    st.markdown("#### 🔹 Basic Features")
    bedrooms = st.number_input("1. Bedrooms", min_value=1, max_value=15, value=3)
    bathrooms = st.number_input("2. Bathrooms", min_value=1.0, max_value=10.0, value=2.5, step=0.5)
    sqft_living = st.number_input("3. Living Area (sqft)", min_value=500, max_value=20000, value=2000)
    sqft_lot = st.number_input("4. Lot Area (sqft)", min_value=500, max_value=100000, value=7500)
    floors = st.number_input("5. Floors", min_value=1.0, max_value=3.5, value=2.0, step=0.5)

    st.markdown("#### 🔹 Quality & Rating")
    waterfront = st.selectbox("6. Waterfront View", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    view = st.slider("7. View Quality", 0, 4, 2)
    condition = st.slider("8. Condition", 1, 5, 3)
    grade = st.slider("9. Construction Grade", 1, 13, 7)
    sqft_above = st.number_input("10. Sqft Above Ground", 500, 15000, 1500)
    sqft_basement = st.number_input("11. Sqft Basement", 0, 5000, 500)
    
    st.markdown("#### 🔹 History & Location")
    yr_built = st.number_input("12. Year Built", 1900, current_year, 1990)
    yr_renovated = st.number_input("13. Year Renovated (0 if none)", 0, current_year, 0)
    lat = st.number_input("14. Latitude", 47.0, 48.0, 47.5, format="%.4f")
    long = st.number_input("15. Longitude", -123.0, -121.0, -122.2, format="%.4f")
    zipcode = st.number_input("16. Zipcode", 98000, 99999, 98074)

    st.markdown("#### 🔹 Neighborhood Data")
    sqft_living15 = st.number_input("17. Neighbor Living Area", 500, 10000, 1800)
    sqft_lot15 = st.number_input("18. Neighbor Lot Area", 500, 100000, 7000)

    predict_button = st.button("Predict Price Now 🚀", type="primary")

if predict_button:
    feature_names = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 
                     'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 
                     'lat', 'long', 'zipcode', 'sqft_living15', 'sqft_lot15']
    
    input_values = [bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, 
                    condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated, 
                    lat, long, zipcode, sqft_living15, sqft_lot15]
    
    input_data = pd.DataFrame([input_values], columns=feature_names)
    predictions = {}
    
    try:
        for model_name, model in loaded_models.items():
            price = model.predict(input_data)[0]
            predictions[model_name] = max(0, float(price))
        
        # Display Prediction Cards
        st.subheader("💰 Predicted Prices for 2025")
        cols = st.columns(len(predictions))
        colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']
        max_price = max(predictions.values())

        for i, (model_name, price) in enumerate(predictions.items()):
            with cols[i]:
                st.markdown(f"""
                <div style="background-color: {colors[i % len(colors)]}; padding: 20px; border-radius: 10px; color: white; text-align: center; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
                    <small style="color: white;">{model_name}</small>
                    <h2 style="color: white; margin:0;">${price:,.0f}</h2>
                    {f'<b style="color: #FFEB3B;">★ Highest</b>' if price == max_price else ''}
                </div>
                """, unsafe_allow_html=True)

        # --- 5. Visualizations ---
        st.divider()
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("📊 Price Comparison")
            df_pred = pd.DataFrame(list(predictions.items()), columns=['Model', 'Price'])
            fig_bar = px.bar(df_pred, x='Model', y='Price', color='Model', text_auto=',.0f', color_discrete_sequence=colors)
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        with chart_col2:
            st.subheader("🏠 Area Distribution")
            fig_pie = px.pie(
                values=[sqft_above, sqft_basement], 
                names=['Above Ground', 'Basement'],
                hole=0.4,
                color_discrete_sequence=['#2196F3', '#FF9800']
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Map Visualization
        st.subheader("📍 Property Location")
        st.map(pd.DataFrame({'lat': [lat], 'lon': [long]}))

        st.divider()
        st.subheader("📋 Input Summary")
        st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error occurred: {e}")