import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from boruta import BorutaPy
import io

def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))
    
st.set_page_config(layout='wide')
c1, padding, c2 = st.columns((10,2,10))

c1.title('Classification')
c1.markdown('Classification Problem: What is the basket size brought by the customer based on the customers feature?')
c1.markdown('#')
c1.markdown('**Exploratory Data Analysis**')
c1.markdown('Below is the stacked barplot of body size against basket size value of big or small.')

df = pd.read_csv("df_final.csv", index_col=0)

fig1,ax1 = plt.subplots(1,1,figsize=(16,20))

sns.histplot(data=df, x="Body_Size",hue="Basket_Size", ax=ax1)
c1.pyplot(fig1)

# Create an in-memory buffer
fn = 'stackedplot_class.pdf'
img = io.BytesIO()
 
plt.savefig(fn, format = 'pdf')
with open(fn, "rb") as img:
    btn = c1.download_button(
        label="Download PDF",
        data=img,
        file_name='figure_class.pdf',
        mime="application/pdf"
    )

c2.markdown('#')
c2.markdown('#')
c2.markdown('#')
c2.markdown('#')
c2.markdown('#')

c2.markdown('**Classification Model Prediction**')

c2.write('Choose the preferred input to predict the basket size value.')

age_val = c2.selectbox("Age: ", sorted(df['Age'].unique()))
spec_val = c2.selectbox("Spectacle: ", sorted(df['Spectacles'].unique()))
attire_val = c2.selectbox("Attire: ", sorted(df['Attire'].unique()))

kidscat_val = c2.selectbox("Kids Category: ", sorted(df['Kids_Category'].unique()))
lat_val = c2.selectbox("Latitude: ", sorted(df['latitude'].unique()))
lon_val = c2.selectbox("Longitude: ", sorted(df['longitude'].unique()))

pantscol_val = c2.selectbox("Pants Colour: ", sorted(df['Pants_Colour'].unique()))
basketcol_val = c2.selectbox("Basket Colour: ", sorted(df['Basket_colour'].unique()))
shirtcol_val = c2.selectbox("Shirt Colour: ", sorted(df['Shirt_Colour'].unique()))

timespent_val = c2.selectbox("Time Spent: ", sorted(df['TimeSpent_minutes'].unique()))
time_val = c2.selectbox("Time (Hour): ", sorted(df['Time1'].unique()))

if c2.button('Predict'):
    new_row = {'Age':age_val, 'Spectacles':spec_val, 'Attire':attire_val, 'Kids_Category':kidscat_val, 'latitude':lat_val, 'longitude':lon_val, 'Pants_Colour':pantscol_val, 'Basket_colour':basketcol_val, 'Shirt_Colour':shirtcol_val, 'Time1':time_val, 'TimeSpent_minutes':timespent_val}
    #append row to the dataframe
    df = df.append(new_row, ignore_index=True)

    le = preprocessing.LabelEncoder()
    df_object = df.select_dtypes(include=['object'])
    df_object = df_object.apply(le.fit_transform)
    df_int = df.select_dtypes(exclude=["object"])

    df_encoded= df_int.join(df_object)

    # Removing column
    list_drop = ['Time']
    df_encoded.drop(list_drop, axis=1, inplace=True)

    # For random forest, class weight should be "balanced" and maximum of tree depth is 5.
    # set random_state = 1
    # set n_jobs=-1
    # n_estimators="auto"

    df_encoded,last_row=df_encoded.drop(df_encoded.tail(1).index),df_encoded.tail(1)
    # pred_row = last_row.dropna(axis=1)

    X = df_encoded.drop(["Basket_Size","Date"],axis=1)
    y = df_encoded["Basket_Size"]
    colnames = X.columns
    # using the BorutaPy function
    # your codes here...
    rf = RandomForestClassifier(n_jobs=-1,class_weight="balanced_subsample",max_depth=5)
    feat_selector = BorutaPy(rf,n_estimators="auto",random_state=1)

    feat_selector.fit(X.values,y.values.ravel())
    boruta_score = ranking(list(map(float, feat_selector.ranking_)), colnames, order=-1)
    boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features', 'Score'])
    boruta_score = boruta_score.sort_values("Score", ascending = False)

    X = df_encoded.drop(["Basket_Size","Date"],axis=1)
    y = df_encoded["Basket_Size"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=27)

    X_train_basket = X_train[boruta_score["Features"][0:10]]
    X_test_basket = X_test[boruta_score["Features"][0:10]]

    pred_row = last_row[boruta_score["Features"][0:10]]
    c2.write(pred_row)

    svm = SVC(C=10, gamma=0.001,probability=True)
    svm.fit(X_train_basket,y_train)
    y_pred = svm.predict(pred_row)
    
    if y_pred[0] == 0:
        basket_sz = 'Big'
    else: 
        basket_sz = 'Small'

    with c2:
        st.write("Predicted value: ", y_pred[0], " - ", basket_sz)
