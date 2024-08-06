import streamlit as st
import pickle
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def load_model_and_scaler():
    with open('scale.pkl', 'rb') as file1, open('model.pkl', 'rb') as file2:
        scale = pickle.load(file1)
        model = pickle.load(file2)
    return scale, model

def main():
    st.write("## MediClaim Case Study")
    st.image('12.jpg')

    # Accept data from user
    age = st.number_input('Age', format='%d', step=1)  # Ensure integer input
    gender_list = ['Male', 'Female']
    gender = st.selectbox('Select Gender', gender_list, index=gender_list.index('Male'))
    gender = 1 if gender == 'Male' else 0
    bmi = st.number_input('BMI', format='%f')  # Floating-point number
    children_list = [0, 1, 2, 3, 4, 5,6,7,8,9,10]
    children = st.selectbox('Enter number of children', children_list, index=children_list.index(0))
    smoker_list = ['Smoker', 'Non-Smoker']
    smoker = st.selectbox('Select Smoker/Non_smoker from options', smoker_list, index=smoker_list.index('Smoker'))
    smoker = 1 if smoker == 'Smoker' else 0
    region_list = [0, 1, 2, 3]
    region = st.selectbox('Enter region number', region_list, index=region_list.index(0))
    charges = st.number_input('Charges', format='%f')  # Floating-point number

    if st.button('Predict'):
        try:
            scale, model = load_model_and_scaler()
            X = [age, gender, bmi, children, smoker, region, charges]
            features = np.array([X])
            features = scale.transform(features)
            Y_pred = model.predict(features)[0]

            if Y_pred == 1:
                st.image('ytick.png')
                st.write('Customer will claim for insurance')
            else:
                st.image('xtick.png')
                st.write('Customer will not claim for insurance')

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
