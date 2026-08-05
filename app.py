import streamlit as st
import pandas as pd
import joblib
import datetime

# ---------- Load saved model, scalers, and encoders ----------
model = joblib.load('car_price_model.pkl')
scaler_engine = joblib.load('scaler_engine.pkl')
scaler_km = joblib.load('scaler_km.pkl')
encoder_owner = joblib.load('encoder_owner.pkl')
encoder_fuel = joblib.load('encoder_fuel.pkl')
encoder_seller_type = joblib.load('encoder_seller_type.pkl')
encoder_transmission = joblib.load('encoder_transmission.pkl')
encoder_brand = joblib.load('encoder_brand.pkl')

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="centered")

st.title("🚗 Car Price Prediction")
st.write("Enter the car details below to estimate its selling price.")

# ---------- User inputs ----------
brand = st.selectbox("Brand", sorted(encoder_brand.classes_))
year = st.number_input("Manufacturing Year", min_value=1990, max_value=datetime.datetime.now().year, value=2015, step=1)
km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000, step=1000)
engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=6.0, value=1.2, step=0.1)
fuel = st.selectbox("Fuel Type", sorted(encoder_fuel.classes_))
seller_type = st.selectbox("Seller Type", sorted(encoder_seller_type.classes_))
transmission = st.selectbox("Transmission", sorted(encoder_transmission.classes_))
owner = st.selectbox("Owner", sorted(encoder_owner.classes_))

if st.button("Predict Price"):
    # ---------- Feature engineering (must match training pipeline) ----------
    car_age = datetime.datetime.now().year - year

    km_driven_scaled = scaler_km.transform([[km_driven]])[0][0]
    engine_size_scaled = scaler_engine.transform([[engine_size]])[0][0]

    owner_num = encoder_owner.transform([owner])[0]
    fuel_num = encoder_fuel.transform([fuel])[0]
    seller_type_num = encoder_seller_type.transform([seller_type])[0]
    transmission_num = encoder_transmission.transform([transmission])[0]
    brand_num = encoder_brand.transform([brand])[0]

    # ---------- Column order MUST match X_train from the notebook ----------
    # km_driven, engine_size, owner_num, fuel_num, seller_type_num, transmission_num, brand_num, car_age
    features = pd.DataFrame([[
        km_driven_scaled,
        engine_size_scaled,
        owner_num,
        fuel_num,
        seller_type_num,
        transmission_num,
        brand_num,
        car_age
    ]], columns=[
        'km_driven', 'engine_size', 'owner_num', 'fuel_num',
        'seller_type_num', 'transmission_num', 'brand_num', 'car_age'
    ])

    prediction = model.predict(features)[0]

    st.success(f"Estimated Selling Price: {prediction:,.0f}")
