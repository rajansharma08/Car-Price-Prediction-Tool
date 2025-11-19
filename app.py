import pandas as pd 
import numpy as np 
import pickle as pk 
import streamlit as st
import base64

# Load model
model = pk.load(open('Model\model.pkl', 'rb'))

# Function to Encode Image to Base64
def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Set Background and Improve Text/Input Contrast
def set_background(image_file):
    base64_str = get_base64(image_file)
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{base64_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    /* Apply 18px font size to all text elements */
   
    
    /* Make Labels & Input Fields Black */
    label, .st-bq, .st-eb, .st-dt, .st-cs, .stSelectbox label, .stSlider label, .stTextInput label {{
        color: white !important;
        font-weight: bold;
    }}

    /* Input Field Styling */
    .stTextInput>div>div>input, 
    .stSelectbox>div>div>select, 
    .stSlider>div>div>div>div {{
        background-color: black !important;
        color: white !important;
        border-radius: 10px;
        padding: 5px;
    }}

    /* Button Styling */
    .stButton>button {{
        background-color: #ff5722 !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 8px;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Apply Background Image
set_background("background.jpg")  # Change filename if needed

# App Header (Larger Size)
st.markdown("<h1 style='text-align: center; font-size: 36px; font-weight: bold;'>🚗 Car Price Prediction ML Model</h1>", unsafe_allow_html=True)

# Load Car Data
cars_data = pd.read_csv('Cardetails.csv')

# Function to extract brand name
def get_brand_name(car_name):
    return car_name.split(' ')[0].strip()
cars_data['name'] = cars_data['name'].apply(get_brand_name)

# UI Elements
name = st.selectbox('Select Car Brand', cars_data['name'].unique())
year = st.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.slider('No of kms Driven', 11, 200000)
fuel = st.selectbox('Fuel type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission type', cars_data['transmission'].unique())
owner = st.selectbox('Owner type', cars_data['owner'].unique())
mileage = st.slider('Car Mileage (km/l)', 10, 40)
engine = st.slider('Engine Capacity (CC)', 700, 5000)
max_power = st.slider('Max Power (BHP)', 0, 200)
seats = st.slider('No of Seats', 2, 10)

# Prediction Logic
if st.button("Predict 🚀"):
    input_data = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats'])

    # Encode categorical variables
    mappings = {
        'owner': {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 5},
        'fuel': {'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4},
        'seller_type': {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3},
        'transmission': {'Manual': 1, 'Automatic': 2},
        'name': {brand: i+1 for i, brand in enumerate(cars_data['name'].unique())}
    }

    for col, mapping in mappings.items():
        input_data[col] = input_data[col].map(mapping)

    # Predict Price
    car_price = model.predict(input_data)

    # Display Result
    st.success(f"🚘 Predicted Car Price: **₹{car_price[0]:,.2f}**")
