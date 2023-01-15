import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from kmodes.kmodes import KModes
import io

st.set_page_config(layout='wide')
c1, padding, c2 = st.columns((10,2,10))

c1.title('Clustering')
c1.markdown('Clustering Problem: Can we identify groups based on their race, gender and attire?')
c1.markdown('#')
c1.markdown('**Exploratory Data Analysis**')
c1.markdown('Below is the stacked barplot of customer race and gender against customer attire.')

df = pd.read_csv("df_final.csv")

fig, axes = plt.subplots(figsize=(8,8))
df.groupby(['Race','Gender', 'Attire']).size().unstack().plot(kind='bar',stacked=True, ax = axes)
c1.pyplot(fig)

fn = 'barplot_clust.pdf'
img = io.BytesIO()
 
plt.savefig(fn, format = 'pdf')
with open(fn, "rb") as img:
    btn = c1.download_button(
        label="Download PDF",
        data=img,
        file_name='figure_clust.pdf',
        mime="application/pdf"
    )

c2.markdown('#')
c2.markdown('#')
c2.markdown('#')
c2.markdown('#')
c2.markdown('#')


c2.markdown('**Clustering Model Prediction**')
c2.write('Choose the preferred input to predict the cluster group.')

race_val = c2.selectbox("Race: ", sorted(df['Race'].unique()))
gender_val = c2.selectbox("Gender: ", sorted(df['Gender'].unique()))
attire_val = c2.selectbox("Attire: ", sorted(df['Attire'].unique()))

if c2.button('Submit'):
    new_row = {'Race':race_val, 'Gender':gender_val, 'Attire':attire_val}
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

    dck = df_encoded[['Race', 'Gender', 'Attire']]

    km_cao = KModes(n_clusters=3, init = "Cao", n_init = 1, verbose=1)
    fitClusters_cao = km_cao.fit_predict(dck)

    dck1 = df[['Race', 'Gender', 'Attire']]
    dck1 = dck1.reset_index()

    clustersDf = pd.DataFrame(fitClusters_cao)
    clustersDf.columns = ['cluster_predicted']
    combinedDf = pd.concat([dck1, clustersDf], axis = 1).reset_index()
    combinedDf = combinedDf.drop(['index', 'level_0'], axis = 1)

    combinedDf['cluster_predicted'] = combinedDf['cluster_predicted'].map({0:'First', 1:'Second', 2:'Third'})

    with c2:
        st.write("Predicted value: ", combinedDf['cluster_predicted'].iloc[-1])