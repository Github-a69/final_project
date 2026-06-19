

import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
import category_encoders 
from sklearn.preprocessing import RobustScaler
from category_encoders import BinaryEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


st.set_page_config(layout='wide',page_title='electric cars')

st.image('https://images.unsplash.com/photo-1593941707874-ef25b8b4a92b?q=80&w=1172&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D')

df=pd.read_csv('clean_electric_vehicle.csv')
st.dataframe(df)


# importing random forest pipeline
model = joblib.load('best_model.pkl')

# Dividing the webage into multiple sections

page=st.sidebar.radio('choose an option',['Overview','Analysis','prediction'])


if page == 'Overview':


    # explaining features meaning
    cols = {'model year':'The manufacturing/model year of the vehicle',
            'make':'The vehicle manufacturer / brand',
            'model':'The model name of the vehicle' , 
            'electric vehicle type':' BEV (Battery Electric Vehicle) — runs purely on electric battery no combustion engine.\n  PHEV (Plug-in Hybrid Electric Vehicle) — has both an electric motor and a gasoline engine',
            'clean alternative fuel vehicle (cafv) eligibility':'Whether the vehicle qualifies  CAFV program (tax/incentive benefits).\n Three possible values:Clean Alternative Fuel Vehicle Eligible — meets the requirements. Not eligible due to low battery range — battery range is too short to qualify. Eligibility unknown as battery range has not been researched — insufficient data to determine',
            'electric range':"The vehicle's all-electric driving range in miles",
            'legislative distric':'legislative district number where the vehicle is registered — used for policy/representation purposes',
            'number of utilities':'The electric utility company/companies that serve the area where the vehicle is registered',
            'location':'city and county in which the vehicle is registered' }


    for col, meaning in cols.items():
        with st.sidebar.expander(col):
            st.write(meaning)




# visualization and analysis page

elif page == 'Analysis':


    tab_1,tab_2,tab_3,tab_4 = st.tabs(['electric range over time','vehicle type vs. electric range',
                          'electric range  vs. cafv eligibility','TOP 5 Brands'])
    with tab_1:
        mean = df.groupby('model year')['electric range'].mean().reset_index()
        line = px.line(mean,y='electric range',x='model year')
        st.plotly_chart(line,use_container_width=False)

    with tab_2:

        group = df.groupby('electric vehicle type')['electric range'].mean().reset_index()
        bar_1 = px.bar(group,y='electric vehicle type',x='electric range',barmode='group')
        st.plotly_chart(bar_1,use_container_width=False)


    with tab_3:
        group_2 = df.groupby('clean alternative fuel vehicle (cafv) eligibility')['electric range'].mean().reset_index()
        bar_2 = px.bar(group_2,y='electric range',x='clean alternative fuel vehicle (cafv) eligibility',barmode='group')
        st.plotly_chart(bar_2,use_container_width=False)



    with tab_4:
        group_3 = df.groupby('make')['electric range'].mean().reset_index().sort_values(by='electric range',ascending=False)
        pie=px.pie(group_3.head(5),values='electric range',names='make',
                   color_discrete_sequence=px.colors.sequential.Blues_r)
        st.plotly_chart(pie,use_container_width=False)



# prediction page

else:


    # filters and inputs

    year = st.selectbox('Year',df['model year'].unique())


    brand =st.sidebar.selectbox('Brand',df['make'].unique())

    # filter models based on selected brand
    filtered_models = df[df['make'] == brand]['model'].unique()
    car_model = st.sidebar.selectbox('Model', filtered_models)



    cafv =st.sidebar.radio('clean alternative',df['clean alternative fuel vehicle (cafv) eligibility'].unique())

    e_range = st.slider('electric range',min_value=float(df['electric range'].min()),
                                                max_value= float(df['electric range'].max()))

    district =st.sidebar.selectbox('District No.',df['legislative district'].unique())

    utilities =st.sidebar.radio('number of utilities',df['number of utilities'].unique())

    # filtered location based on selected brand and model
    filtered_locations = df[(df['make'] == brand) & (df['model'] == car_model)]['location'].unique()
    location = st.sidebar.selectbox('Location', filtered_locations)

    new_df = pd.DataFrame(data= [[year,brand, car_model,cafv, e_range, district, utilities,location ]],
        columns=df.drop('electric vehicle type',axis=1).columns)

    st.dataframe(new_df)


    # Prediction
    if st.button("Predict Vehicle Type"):



        result = model.predict(new_df)[0]

        if result == 0:
            st.write('Battery Electric Vehicle (BEV)')
        elif result == 1:
            st.write('Plug-in Hybrid Electric Vehicle (PHEV)')






