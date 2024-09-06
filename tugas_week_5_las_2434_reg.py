import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title("Prediksi Usia Fossil")

def user_input_features():
    st.sidebar.header("Input Features")
    ranges = {
        'uranium_lead_ratio': (-1.305179, 2.308448),
        'carbon_14_ratio': (-1.206812, 1.702826),
        'radioactive_decay_series': (-1.258500, 2.028883),
        'stratigraphic_layer_depth': (-1.156913, 2.046896),
        'geological_period': (0.0, 10.0),
        'isotopic_composition': (-1.213111, 2.026097),
        'stratigraphic_position': (0.0, 2.0),
        'fossil_size': (-1.260029, 2.007663),
        'fossil_weight': (-1.124372, 2.033519)
    }
    
    features = {}
    for feature, (min_val, max_val) in ranges.items():
        help_text = f"Range: {min_val:.6f} to {max_val:.6f}"
        value = st.sidebar.slider(
            label=f"{feature.replace('_', ' ').title()}",
            min_value=float(min_val),
            max_value=float(max_val),
            value=(min_val + max_val)/2,  
            step=0.0001,  
            help=help_text
        )
        features[feature] = value
    return pd.DataFrame(features, index=[0])

input_df = user_input_features()

data = {
    'uranium_lead_ratio': np.random.uniform(-1.305179, 2.308448, 100),
    'carbon_14_ratio': np.random.uniform(-1.206812, 1.702826, 100),
    'radioactive_decay_series': np.random.uniform(-1.258500, 2.028883, 100),
    'stratigraphic_layer_depth': np.random.uniform(-1.156913, 2.046896, 100),
    'geological_period': np.random.randint(0, 10, 100),
    'isotopic_composition': np.random.uniform(-1.213111, 2.026097, 100),
    'stratigraphic_position': np.random.uniform(0, 2, 100),
    'fossil_size': np.random.uniform(-1.260029, 2.007663, 100),
    'fossil_weight': np.random.uniform(-1.124372, 2.033519, 100),
    'age': np.random.uniform(-1.714439, 2.018088, 100)
}

df = pd.DataFrame(data)

X = df[['uranium_lead_ratio', 'carbon_14_ratio', 'radioactive_decay_series', 'stratigraphic_layer_depth',
        'geological_period', 'isotopic_composition', 'stratigraphic_position', 'fossil_size', 'fossil_weight']]
y = df['age']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

if st.button("Predict Age"):
    prediction = model.predict(input_df)
    st.write(f"Hasil Prediksi Usia: {prediction[0]:.4f}")