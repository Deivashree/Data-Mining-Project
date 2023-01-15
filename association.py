import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from apyori import apriori
import io
from boruta import BorutaPy

st.set_page_config(layout='wide')
c1, padding, c2 = st.columns((10,2,10))

c1.title('Association Rule Mining')
c1.markdown('Association Rule Mining Problem: Is there any correlation between customer’s age, customer with spectacles, gender of the customer and customer’s attire with the washer number?')
c1.markdown('#')
c1.markdown('**Exploratory Data Analysis**')
c1.markdown('Below is the histogram to identify the relationship between customer age, race, gender, attire, spectacles and the washer number of the laundry store.')

df = pd.read_csv("df_final.csv")

graph = sns.FacetGrid(df, row ='Washer_No', col ='Spectacles')
# map the above form facetgrid with some attributes
graph.map(plt.hist, 'Age', bins = 15, color ='orange')
c1.pyplot(graph)

fn = 'hist_assoc1.pdf'
img = io.BytesIO()
 
plt.savefig(fn, format = 'pdf')
with open(fn, "rb") as img:
    btn = c1.download_button(
        label="Download PDF",
        data=img,
        file_name='figure_assoc1.pdf',
        mime="application/pdf"
    )

graph = sns.FacetGrid(df, row ='Washer_No', col ='Gender')
# map the above form facetgrid with some attributes
graph.map(plt.hist, 'Age', bins = 15, color ='blue')
c1.pyplot(graph)

fn = 'hist_assoc2.pdf'
img = io.BytesIO()
 
plt.savefig(fn, format = 'pdf')
with open(fn, "rb") as img:
    btn = c1.download_button(
        label="Download PDF",
        data=img,
        file_name='figure_assoc2.pdf',
        mime="application/pdf"
    )

graph = sns.FacetGrid(df, row ='Washer_No', col ='Attire')
# map the above form facetgrid with some attributes
graph.map(plt.hist, 'Age', bins = 15, color ='green')
c1.pyplot(graph)

fn = 'hist_assoc3.pdf'
img = io.BytesIO()
 
plt.savefig(fn, format = 'pdf')
with open(fn, "rb") as img:
    btn = c1.download_button(
        label="Download PDF",
        data=img,
        file_name='figure_assoc3.pdf',
        mime="application/pdf"
    )

c2.markdown('#')
c2.markdown('#')
c2.markdown('#')
c2.markdown('#')
c2.markdown('#')
c2.markdown('#')
c2.markdown('**Association Rule Mining Model Prediction**')

df1 = df[['Washer_No','Age','Spectacles','Gender','Attire']]

c2.write('Choose the preferred input to predict the target value.')

washer_val = c2.selectbox("Washer Number: ", sorted(df1['Washer_No'].unique()))
age_val = c2.selectbox("Age: ", sorted(df1['Age'].unique()))
spec_val = c2.selectbox("Spectacle: ", sorted(df1['Spectacles'].unique()))
gender_val = c2.selectbox("Gender: ", sorted(df1['Gender'].unique()))
attire_val = c2.selectbox("Attire: ", sorted(df1['Attire'].unique()))

if c2.button('Submit'):
    new_row = {'Washer_No':washer_val, 'Age':age_val, 'Spectacles':spec_val, 'Gender':gender_val, 'Attire':attire_val}
    #append row to the dataframe
    df1 = df1.append(new_row, ignore_index=True)

    num_rec = len(df1)

    records = []
    # set i in range to num_records so it can display 4000 rows
    for i in range (0, num_rec):
        # set j in range 5 to display 5 columns that we have choose
        records.append([str(df1.values[i,j]) for j in range(0,5)])

    association_rules = apriori(records, min_support = 0.0025, min_lift=2.5, min_length=2)
    # transform it to a list
    rules = list(association_rules)

    for i in rules:

    # first index of the inner list
        pair = i[0] 
        items = [x for x in pair]
        # print rule
        with c2:
            st.write("Rule: " + items[0] + " -> " + items[1])
            #second index of the inner list
            # print support
            st.write("Support: " + str(i[1]))
            #third index of the list located at 0th
            #of the third index of the inner list

            st.write("Confidence: " + str(i[2][0][2]))
            st.write("Lift: " + str(i[2][0][3]))
            st.write("=====================================")
