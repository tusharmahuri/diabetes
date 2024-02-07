import streamlit as st
import pickle
import pandas as pd

# Load the saved Logistic Regression model
with open('diabetes_lr_model.pkl', 'rb') as model_file:
    LRModel = pickle.load(model_file)

# Define the prediction function
def predict_diabetes(data):
    prediction = LRModel.predict(data)
    return prediction

# Create the Streamlit app
def main():
    st.title('Diabetes Prediction App')

    # Create columns to align the inputs and the prediction button in the middle
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.header('Input Data')
        # Collect input data from the user
        pregnancies = st.slider('Pregnancies', 0, 17, 3)
        glucose = st.slider('Glucose', 0, 199, 117)
        blood_pressure = st.slider('Blood Pressure', 0, 122, 72)
        skin_thickness = st.slider('Skin Thickness', 0, 99, 23)
        insulin = st.slider('Insulin', 0, 846, 30)
        bmi = st.slider('BMI', 0.0, 67.1, 32.0)
        diabetes_pedigree_function = st.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)
        age = st.slider('Age', 21, 81, 29)

        # Create a DataFrame from the input data
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [diabetes_pedigree_function],
            'Age': [age]
        })

        # Predict diabetes
        if st.button('Predict'):
            prediction = predict_diabetes(input_data)
            if prediction[0] == 0:
                st.success('No diabetes detected.')
            else:
                st.error('Diabetes detected.')

if __name__ == '__main__':
    main()
