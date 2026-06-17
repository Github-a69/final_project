

import pandas as pd
import streamlit as st
import joblib
import category_encoders 
from sklearn.preprocessing import RobustScaler
from category_encoders import BinaryEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

df=pd.read_csv('clean_electric_vehicle.csv')
st.dataframe(df)

st.set_page_config(layout='wide',page_title='electric cars')



# Filters

brand =st.sidebar.selectbox('Brand',df['make'].unique())

car_model =st.sidebar.selectbox('Model',df['model'].unique())


location =st.sidebar.selectbox('Location',df['location'].unique())


district =st.sidebar.selectbox('District No.',df['legislative district'].unique())


cafv =st.sidebar.radio('clean alternative',df['clean alternative fuel vehicle (cafv) eligibility'].unique())

utilities =st.sidebar.radio('number of utilities',df['number of utilities'].unique())

# User Input 

year = st.number_input('Year',min_value=int(df['model year'].min()),
                                max_value= int(df['model year'].max()))

e_range = st.slider('electric range',min_value=float(df['electric range'].min()),
                                            max_value= float(df['electric range'].max()))


# importing random forest pipeline
model = joblib.load('best_model.pkl')




new_df = pd.DataFrame(data=
    [[ year,brand, car_model,cafv, e_range, district, utilities,location ]],
    columns=df.drop('electric vehicle type',axis=1).columns)

# Prediction
if st.button("Predict Vehicle Type"):

    result = model.predict(new_df)[0]

    if result == 0:
        st.write('Battery Electric Vehicle (BEV)')
    else:
        st.write('Plug-in Hybrid Electric Vehicle (PHEV)')



st.image('https://images.unsplash.com/photo-1593941707874-ef25b8b4a92b?q=80&w=1172&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D')


