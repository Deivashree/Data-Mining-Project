import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

import io

st.set_page_config(layout='wide')
c1, padding, c2 = st.columns((10,2,10))

c1.title('Regression')
c1.markdown('Regression Problem: What is the total amount spent by customers depending on the weather condition along with customer race, gender, kids and attire?')
c1.markdown('#')
c1.markdown('**Exploratory Data Analysis**')
c1.markdown('Below is the boxplot of weather condition, customer race, customer gender, customer attire and customer number of kids that is plotted against total amount of spent by customers.')

df = pd.read_csv("df_final.csv")

fig1,ax1 = plt.subplots(3,2,figsize=(16,20))

#Plot boxplot for the each variable with total amount spent 
sns.boxplot(x = 'Weather_Condition',
            y = 'TotalSpent_RM',
            data = df,ax=ax1[0][0])
sns.boxplot(x = 'Race',
            y = 'TotalSpent_RM',
            data = df,ax=ax1[0][1])
sns.boxplot(x = 'Gender',
            y = 'TotalSpent_RM',
            data = df,ax=ax1[1][0])
sns.boxplot(x = 'Attire',
            y = 'TotalSpent_RM',
            data = df,ax=ax1[1][1])
sns.boxplot(x = 'Kids_Category',
            y = 'TotalSpent_RM',
            data = df,ax=ax1[2][0])

c1.pyplot(fig1)

fn = 'boxplot_reg.pdf'
img = io.BytesIO()
 
plt.savefig(fn, format = 'pdf')
with open(fn, "rb") as img:
    btn = c1.download_button(
        label="Download PDF",
        data=img,
        file_name='figure_reg.pdf',
        mime="application/pdf"
    )

c2.markdown('#')
c2.markdown('#')
c2.markdown('#')
c2.markdown('#')
c2.markdown('#')
c2.markdown('#')

c2.markdown('**Regression Model Prediction**')

c2.write('Choose the preferred input below to predict the total amount spent by the customer.')

weathercond_val = c2.selectbox("Weather Condition: ", sorted(df['Weather_Condition'].unique()))
attire_val = c2.selectbox("Attire: ", sorted(df['Attire'].unique()))
kidscat_val = c2.selectbox("Kids Category: ", sorted(df['Kids_Category'].unique()))
race_val = c2.selectbox("Race: ", sorted(df['Race'].unique()))
gender_val = c2.selectbox("Gender: ", sorted(df['Gender'].unique()))

if c2.button('Predict'):
    new_row = {'Weather Condition: ':weathercond_val, 'Attire':attire_val,'Kids_Category':kidscat_val, 'Race':race_val, 'Gender':gender_val}
    #append row to the dataframe
    df = df.append(new_row, ignore_index=True)

    le = preprocessing.LabelEncoder()
    df_object = df.select_dtypes(include=['object'])
    df_object = df_object.apply(le.fit_transform)
    df_int = df.select_dtypes(exclude=["object"])

    df_encoded= df_int.join(df_object)

    df_encoded,last_row=df_encoded.drop(df_encoded.tail(1).index),df_encoded.tail(1)

    X = df_encoded[['Race','Gender', 'Basket_Size', 'With_Kids',
        'Kids_Category', 'Attire',
            'Weather_Condition']]

    Y = df_encoded['TotalSpent_RM']

    #setting 5 selection of features
    num_of_features = 5
    # configure to select all features
    fs = SelectKBest(score_func=f_regression, k=num_of_features)
    # learn relationship from training data
    fit = fs.fit(X, Y)

    #placing feature name and feature score into dataframe
    features_score = pd.DataFrame(fit.scores_)
    features = pd.DataFrame(X.columns)
    feature_score = pd.concat([features,features_score],axis=1)
    # Assigning column names
    feature_score.columns = ["Input_Features","F_Score"]

    best_feature_reg = feature_score.nlargest(num_of_features,columns="F_Score")['Input_Features'].tolist()
    X = X[best_feature_reg]

    skf = StratifiedKFold(n_splits=3,shuffle=True, random_state=42)
    train_index, test_index = next(skf.split(X, Y))  #obtaining indexes of 1/3 of original dataset rows##

    X_fold = X.iloc[test_index]
    y_fold = Y.iloc[test_index]

    X_train_fold, X_test_fold, y_train_fold, y_test_fold = train_test_split(X_fold, y_fold, test_size=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state=42)


    params = { 
            'n_estimators': [1000],
            'max_depth': [4],                  #'max_depth': [4,5,6,7,8,9]
            'max_features': ['auto', 'sqrt'],  #'max_features': ['auto', 'sqrt']
            'min_samples_leaf': [4],           #'min_samples_leaf': [2,3,4,5,6,7]
            'min_samples_split' : [0.01],      #'min_samples_split' : [0.01]
            }

    rf = RandomForestRegressor(random_state=42)
    rs = GridSearchCV(estimator=rf, param_grid=params, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    rs.fit(X_train_fold, y_train_fold)

    rf = rs.best_estimator_
    rf.fit(X_train, y_train)

    pred_row = last_row[best_feature_reg]
    pred = rf.predict(pred_row)

    with c2:
        st.write("Predicted value: ", pred[0])
    



