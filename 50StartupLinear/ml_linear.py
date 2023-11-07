import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error,r2_score



st.title('50startup')
st.write("This app uses 6 inputs to predict the Variety of Iris using "
         "a model built on the Palmer's Iris's dataset. Use the form below"
         " to get started!")

csv_file = st.file_uploader('Upload your own Iris data')
if csv_file is None:
    linear_pickle = pickle.load(open('/mount/src/linearregression/50StartupLinear/model1.pkl', 'rb'))
else:
    df = pd.read_csv(csv_file)
    # iris_df = iris_df.dropna()
    linear_pickle = pickle.load(open('/mount/src/linearregression/50StartupLinear/model1.pkl', 'rb'))
    output = df['Profit']
    features = df[['R&D Spend',
           'Administration',
           'Marketing Spend']]

    # features = pd.get_dummies(features)

    output, unique_penguin_mapping = pd.factorize(output)

    x_train, x_test, y_train, y_test = train_test_split(
        features, output, test_size=0.2)
    models = LinearRegression()
    models.fit(x_train,y_train)

    y_pred = models.predict(x_test)
    # score = round(accuracy_score(y_pred, y_test), 2)
    st.title("Linear Regression ")
    st.markdown('ตัวอย่างการทำนาย ยอดที่ได้รับ ของข้อมูล  **50 Startups\'s **  **ทดสอบ**')

    # choices = ['R&D Spend',
    #        'Administration',
    #        'Marketing Spend']
    #
    # selected_x_var = st.selectbox('เลือก แกน x', (choices))
    # selected_y_var = st.selectbox('เลือก แกน y', (choices))

    st.subheader('ข้อมูลตัวอย่าง')
    st.write(df)

    # st.subheader('แสดงผลข้อมูล')
    # sns.set_style('darkgrid')
    # markers = {"Setosa": "v", "Versicolor": "s", "Virginica": 'o'}
    #
    # fig, ax = plt.subplots()
    # ax = sns.scatterplot(data=iris_df,
    #                      x=selected_x_var, y=selected_y_var,
    #                      hue='Profit', markers=markers, style='Profit')
    # plt.xlabel(selected_x_var)
    # plt.ylabel(selected_y_var)
    # plt.title("Palmer's Penguins Data")
    # st.pyplot(fig)

    # textscore='<p style="font-family:Courier; color:Black; font-size: 16px;">We trained a Random Forest model on these data ,it has a score of {}! Use the inputs below to try out the model.</p>'
    # st.write(textscore.format(score),unsafe_allow_html=True)

with st.form('user_inputs'):
    RD_Spend = st.number_input(
        'R&D Spend', min_value=0.0, max_value=800000.0, value=10.0)
    administration = st.number_input(
         'Administration', min_value=0.0, max_value=800000.0, value=10.0)
    marketing_spend = st.number_input(
        'Marketing Spend', min_value=0.0, max_value=800000.0, value=10.0)
    st.form_submit_button()


model = pickle.load(open('/mount/src/linearregression/50StartupLinear/model1.pkl', 'rb'))
new_prediction =model.predict([[RD_Spend, administration, marketing_spend]])
# prediction_species = unique_penguin_mapping[new_prediction][0]
textpredict = '<p style="font-family:Courier; color:Black; font-size: 20px;">We predict your Profit is of the {} US</p>'
st.markdown(textpredict.format(new_prediction), unsafe_allow_html=True)








