from sklearn.linear_model import Lasso
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew
#from sklearn.preprocessing import L
import numpy as np
def load_data():
    df1=pd.read_csv("2000Feb2012.csv")
    df2=pd.read_csv("19901999.csv")
    df3=pd.read_csv("Jan2015toDec2016.csv")
    df4=pd.read_csv("Jan2017onwards.csv")
    df5=pd.read_csv("Mar2012toDec2014.csv")
    df=pd.concat([df1,df2,df3,df4,df5])
    return df

def preprocess(df):
    le=LabelEncoder()
    df["town"]=le.fit_transform(df["town"])
    df["flat_type"]=le.fit_transform(df["flat_type"])
    df["block"]=le.fit_transform(df["block"])
    df["street_name"]=le.fit_transform(df["street_name"])
    df["flat_model"]=le.fit_transform(df["flat_model"])
    df["lease_commence_date"]=le.fit_transform(df["lease_commence_date"])
     
    df['month']=le.fit_transform(df['month'])
    df['storey_range']=le.fit_transform(df['storey_range'])
    # df['month']=df['month'].astype('category')
    # df['town']=df['town'].astype('category')
    # df['flat_type']=df['flat_type'].astype('category')
    # df['block']=df['block'].astype('category')
    # df['street_name']=df['street_name'].astype('category')
    # df['storey_range']=df['storey_range'].astype('category')
    # df['flat_model']=df['flat_model'].astype('category')
    # df['lease_commence_date']=df['lease_commence_date'].astype('category')



    df["storey_range"]=df["storey_range"].astype('category')
    c=pd.Categorical(df["storey_range"])
    df["storey_range"]=c.codes+1
    df['storey_range']=np.log(df['storey_range'])
    #print("storey_log",skew(storey_log))
    # df["month"]=df["month"].astype('category')
    # c=pd.Categorical(df['month'])
    # df['month']=c.codes+1
    # df['month']=np.log(df['month'])

    #df['month']=np.sqrt(df['month'])
    #df['storey_range']=np.sqrt(df['storey_range'])

    df=df[df['storey_range']<10]
    df=df[df['lease_commence_date']<50]
    return df

def build_model(df):
    x=df.iloc[:,[0,1,2,3,4,5,6,7,8]]
    y=df.iloc[:,[9]]


    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


    regressor=RandomForestRegressor()
    regressor.fit(x_train,y_train)
    y_pred=regressor.predict(x_test)
    lr=r2_score(y_pred,y_test)
    me=mean_squared_error(y_pred,y_test)
    print(lr)
    print(me)
    return regressor,lr

st.title("Singapore House Price Prediction")
col1,col2,col3=st.columns(3)
month1=col1.text_input("month",353)
town1=col2.text_input("town",14)
flat_type1=col3.text_input("flat_type",3)
col4,col5,col6=st.columns(3)
block1=col4.text_input("block",1)
street_name1=col5.text_input("street_name",441)
storey_range1=col6.text_input("storey_range",1)
col7,col8,col9=st.columns(3)
flat_model1=col7.text_input("flat_model",16)
lease_commence_date1=col8.text_input("lease_commence_date",19)
floor_area_sqm1=col9.text_input("floor_area_sqm",104)
#st.button("Predict")
if st.button("Predict"):
    df=load_data()
    df=preprocess(df)
    regressor,lr=build_model(df)
    y_pred=[[int(month1),int(town1),int(flat_type1),int(block1),int(street_name1),int(storey_range1),int(flat_model1),int(lease_commence_date1),int(floor_area_sqm1)]] #[month1,town1,flat_type1,block1,street_name1,storey_range1,floor_area_sqm1,flat_model1,lease_commence_date1]
    y_pred1=regressor.predict(y_pred)
    st.write(y_pred1)
    st.write("r2_score",lr)