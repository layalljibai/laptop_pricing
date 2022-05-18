#-----------------------------------------------Importing needed python modules-----------------------------------------------------------------------------------
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import streamlit as st
from datetime import datetime, timedelta
import plotly as py
import plotly.graph_objs as go
import plotly.offline as pyoff
from plotly.subplots import make_subplots
from pathlib import Path

#-----------------------------------------------Defining needed functions-----------------------------------------------------------------------------------
#Extracting X-axis and Y-axis screen resolution dimensions by splitting the string with the cross sign
def findXresolution(s):
  return s.split()[-1].split("x")[0]
def findYresolution(s):
  return s.split()[-1].split("x")[1]
  
#Extracting Name of CPU which is first 3 words from Cpu column and then we will check which processor it is
def fetch_processor(x):
  cpu_name = " ".join(x.split()[0:3])
  if cpu_name == 'Intel Core i7' or cpu_name == 'Intel Core i5' or cpu_name == 'Intel Core i3':
    return cpu_name
  elif cpu_name.split()[0] == 'Intel':
    return 'Other Intel Processor'
  else:
    return 'AMD Processor'

#Get which OP sys
def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'
        
def read_html_file(markdown_file):
    return Path(markdown_file).read_text()
    
@st.cache(show_spinner=False)
def loadAndPreprocessData():
  laptopdata = pd.read_csv("laptop_prices.csv")
  #checking if there is null values 
  laptopdata.isnull().sum()
  #removing the id column from the dataset
  laptopdata.drop(columns=['Unnamed: 0'],inplace=True)
  ## removing gb and kg from Ram and weight and convert the columns to numeric
  laptopdata['Ram'] = laptopdata['Ram'].str.replace("GB", "")
  laptopdata['Weight'] = laptopdata['Weight'].str.replace("kg", "")
  laptopdata['Ram'] = laptopdata['Ram'].astype('int32')
  laptopdata['Weight'] = laptopdata['Weight'].astype('float32')
  #extracting the field Touchscreen from ScreenResolution
  laptopdata['Touchscreen'] = laptopdata['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)
  #extract IPS column from ScreenResolution
  laptopdata['Ips'] = laptopdata['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)
  #finding the x_res and y_res from screen resolution
  laptopdata['X_res'] = laptopdata['ScreenResolution'].apply(lambda x: findXresolution(x))
  laptopdata['Y_res'] = laptopdata['ScreenResolution'].apply(lambda y: findYresolution(y))
  #convert x_res and y_res to numeric
  laptopdata['X_res'] = laptopdata['X_res'].astype('int')
  laptopdata['Y_res'] = laptopdata['Y_res'].astype('int')

  #Calculating the pixel per inch field and dropping unnecessary columns after performing the calculation
  laptopdata['ppi'] = (((laptopdata['X_res']**2) + (laptopdata['Y_res']**2))**0.5/laptopdata['Inches']).astype('float')
  laptopdata.drop(columns = ['ScreenResolution', 'Inches','X_res','Y_res'], inplace=True)


  laptopdata['Cpu_brand'] = laptopdata['Cpu'].apply(lambda x: fetch_processor(x))


  #preprocessing
  laptopdata['Memory'] = laptopdata['Memory'].astype(str).replace('.0', '', regex=True)

  laptopdata["Memory"] = laptopdata["Memory"].str.replace('GB', '')
  laptopdata["Memory"] = laptopdata["Memory"].str.replace('TB', '000')
  new = laptopdata["Memory"].str.split("+", n = 1, expand = True)
  laptopdata["first"]= new[0]
  laptopdata["first"]=laptopdata["first"].str.strip()
  laptopdata["second"]= new[1]
  laptopdata["Layer1HDD"] = laptopdata["first"].apply(lambda x: 1 if "HDD" in x else 0)
  laptopdata["Layer1SSD"] = laptopdata["first"].apply(lambda x: 1 if "SSD" in x else 0)
  laptopdata["Layer1Hybrid"] = laptopdata["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
  laptopdata["Layer1Flash_Storage"] = laptopdata["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)
  laptopdata['first'] = laptopdata['first'].str.replace('Flash Storage', '')
  laptopdata['first'] = laptopdata['first'].str.replace('SSD', '')
  laptopdata['first'] = laptopdata['first'].str.replace('HDD', '')
  laptopdata['first'] = laptopdata['first'].str.replace('Hybrid', '')
  laptopdata["second"].fillna("0", inplace = True)
  laptopdata['second'] = laptopdata['second'].str.replace('Hybrid', '')
  laptopdata['second'] = laptopdata['second'].str.replace('H', '')
  laptopdata['second'] = laptopdata['second'].str.replace('SS', '')

  #binary encoding
  laptopdata["Layer2HDD"] = laptopdata["second"].apply(lambda x: 1 if "HDD" in x else 0)
  laptopdata["Layer2SSD"] = laptopdata["second"].apply(lambda x: 1 if "SSD" in x else 0)
  laptopdata["Layer2Hybrid"] = laptopdata["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
  laptopdata["Layer2Flash_Storage"] = laptopdata["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

  #only keep integer(digits)
  laptopdata['second'] = laptopdata['second'].str.replace(r'D', '')

  #convert to numeric
  laptopdata["first"] = laptopdata["first"].astype(int)
  laptopdata["second"] = laptopdata["second"].astype(int)

  
  laptopdata["HDD"]=(laptopdata["first"]*laptopdata["Layer1HDD"]+laptopdata["second"]*laptopdata["Layer2HDD"])
  laptopdata["SSD"]=(laptopdata["first"]*laptopdata["Layer1SSD"]+laptopdata["second"]*laptopdata["Layer2SSD"])
  laptopdata["Hybrid"]=(laptopdata["first"]*laptopdata["Layer1Hybrid"]+laptopdata["second"]*laptopdata["Layer2Hybrid"])
  laptopdata["Flash_Storage"]=(laptopdata["first"]*laptopdata["Layer1Flash_Storage"]+laptopdata["second"]*laptopdata["Layer2Flash_Storage"])
  #Drop the un needed columns
  laptopdata.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
        'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
        'Layer2Flash_Storage'],inplace=True)

  laptopdata.drop(columns=['Hybrid','Flash_Storage','Memory','Cpu'],inplace=True)
        
  # Extracting the brand of the laptop
  laptopdata['Gpu_brand'] = laptopdata['Gpu'].apply(lambda x:x.split()[0])
  #there is only 1 row of ARM GPU so lets remove it
  laptopdata = laptopdata[laptopdata['Gpu_brand'] != 'ARM']
  laptopdata.drop(columns=['Gpu'],inplace=True)

  #Extracting the operating system
  laptopdata['os'] = laptopdata['OpSys'].apply(cat_os)
  laptopdata.drop(columns=['OpSys'],inplace=True)

  return laptopdata


#-----------------------------------------------Main program-----------------------------------------------------------------------------------
#Adjusting the display
st.set_page_config(layout="wide")
header_markdown = read_html_file("header.md")
st.markdown(header_markdown, unsafe_allow_html=True)
#Adding the left menu
st.sidebar.title("Analysis Type")
config_option = st.sidebar.radio("", ['Descriptive Analysis', 'Predict laptop price'], 0)
st.sidebar.markdown('---')

data = loadAndPreprocessData()
if config_option == 'Descriptive Analysis':
  company_price= data.groupby(['Company'])['Price'].mean().reset_index()
  fig = px.bar(company_price, x= "Company" , y="Price")
  fig.update_layout(xaxis_type='category')
  checkbox1 = st.checkbox('Price by manufacturer')
  if checkbox1:
    st.plotly_chart(fig)
    st.write("This bar chart shows that the most expensive laptops are those manufactured by Razer, LG, MSI and Apple")
    st.markdown('---')
  
  touchscreen_price= data.groupby(['Touchscreen'])['Price'].mean().reset_index()
  fig = px.bar(touchscreen_price, x= "Touchscreen" , y="Price")
  fig.update_layout(xaxis_type='category')
  checkbox2 = st.checkbox('Touch screen effect on the price')
  if checkbox2:
    st.plotly_chart(fig)
    st.write("Laptops having Touch Screen feature are more expensive")
    st.markdown('---')

  os_price= data.groupby(['os'])['Price'].mean().reset_index()
  fig = px.pie(os_price, values='Price', names='os')
  checkbox3 = st.checkbox('Price by operating system')
  if checkbox3:
    st.plotly_chart(fig)
    st.write("Obviously Mac is the most expensive operating system")
    st.markdown('---')
  
  
  cpu_price= data.groupby(['Cpu_brand'])['Price'].mean().reset_index()
  fig = px.bar(cpu_price, x= "Cpu_brand" , y="Price")
  fig.update_layout(xaxis_type='category')
  checkbox4 = st.checkbox('Price by CPU')
  if checkbox4:
    st.plotly_chart(fig)
    st.write("This graph shows that the processor type affects significantly the laptop price")
    st.markdown('---')

  cat_price= data.groupby(['TypeName'])['Price'].mean().reset_index()
  fig = px.bar(cat_price, x= "TypeName" , y="Price")
  fig.update_layout(xaxis_type='category')
  checkbox5 = st.checkbox('Price by laptop type')
  if checkbox5:
    st.plotly_chart(fig)
    st.write("Workstation and Gaming laptops have the highest prices")
    st.markdown('---')

#-----------------------------------------------Building the machine learning prediction model-----------------------------------------------------------------------------------
#Importing needed librairies only when the 'Predict laptop price' option is chosen 
if config_option == 'Predict laptop price':
  from sklearn.model_selection import train_test_split
  from sklearn.compose import ColumnTransformer
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import OneHotEncoder
  from sklearn.metrics import r2_score,mean_absolute_error
  from sklearn.linear_model import LinearRegression,Ridge,Lasso
  from sklearn.neighbors import KNeighborsRegressor
  from sklearn.tree import DecisionTreeRegressor
  from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
  from sklearn.svm import SVR
  from xgboost import XGBRegressor
  import pickle

  X = data.drop(columns=['Price'])

  y = np.log(data['Price'])

  X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=2)

  step1 = ColumnTransformer(transformers=[

  ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])

  ],remainder='passthrough')

  step2 = RandomForestRegressor(n_estimators=100,

  random_state=3,

  max_samples=0.5,

  max_features=0.75,

  max_depth=15)

  pipe = Pipeline([

  ('step1',step1),

  ('step2',step2)

  ])

  pipe.fit(X_train,y_train)

  y_pred = pipe.predict(X_test)

  print('R2 score',r2_score(y_test,y_pred))

  print('MAE',mean_absolute_error(y_test,y_pred))
  
  data.to_csv("df.csv", index=False)
  pickle.dump(pipe,open('pipe.pkl','wb'))

  #load the model and dataframe
  df = pd.read_csv("df.csv")
  pipe = pickle.load(open("pipe.pkl", "rb"))
  #building the user interface
  st.title("Choose the desired laptop specifications")
  #Now we will take user input one by one as per our dataframe
  #Brand
  company = st.selectbox('Brand', df['Company'].unique())
  #Type of laptop
  lap_type = st.selectbox("Type", df['TypeName'].unique())
  #Ram
  ram = st.selectbox("Ram(in GB)", [2,4,6,8,12,16,24,32,64])
  #weight
  weight = st.number_input("Weight of the Laptop")
  #Touch screen
  touchscreen = st.selectbox("TouchScreen", ['No', 'Yes'])
  #IPS
  ips = st.selectbox("IPS", ['No', 'Yes'])
  #screen size
  screen_size = st.number_input('Screen Size')
  # resolution
  resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
  #cpu
  cpu = st.selectbox('CPU',df['Cpu_brand'].unique())
  hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])
  ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])
  gpu = st.selectbox('GPU',df['Gpu_brand'].unique())
  os = st.selectbox('OS',df['os'].unique())
 
  if st.button('Predict Price'):
      ppi = None
      if touchscreen == "Yes":
          touchscreen = 1
      else:
          touchscreen = 0
      if ips == "Yes":
          ips = 1
      else:
          ips = 0
      X_res = int(resolution.split('x')[0])
      Y_res = int(resolution.split('x')[1])
      #validating the value of the screen size field
      if(screen_size <=0.0):
        st.subheader("The screen size field should be greater than 0")
      else:
        ppi = ((X_res ** 2) + (Y_res**2)) ** 0.5 / screen_size
        query = np.array([company,lap_type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
        query = query.reshape(1, 12)
        prediction = str(int(np.exp(pipe.predict(query)[0]))/100)
        
        st.subheader("The predicted price of this laptop is " + prediction + " USD")
