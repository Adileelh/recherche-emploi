import streamlit as st
import joblib
import pandas as pd


# Load the pre-trained model
pipe_model = joblib.load('random_forest_regressor')


fn_data = 'data_bdd.csv'

df = pd.read_csv(fn_data)


# Define a function to make predictions with the model
def predict(pipe_model, data):
    df_pred = pd.DataFrame(data, index=[0])
    pred_salaire = pipe_model.predict(df_pred)
    return pred_salaire[0]


# Use Streamlit to build the app
st.title("Salaire Predictor")

# Input data



options = unique_values = df["Intitulé du poste"].unique()
selected_intitule = st.selectbox(
    "Sélectionnez le métier de votre choix", options)

start_index = 9
columns = df.columns[start_index:]
options = columns
selected_skills = st.multiselect("Sélectionnez vos compétences", options)
selected_skills = ",".join(options)


options = unique_values = df["type_contrat"].unique()
selected_contrat = st.selectbox(
    "Sélectionnez un type de contract", options)

options = unique_values = df["lieu"].unique()
selected_lieu = st.selectbox(
    "Sélectionnez une ville", options)

entreprise = ""
show_entreprise = False
if show_entreprise:
    entreprise = st.text_input("entreprise", "")
else:
    st.write("")



# Make the prediction
if st.button("Predict"):
    new_data = {
        'Intitulé du poste': selected_intitule,
        'competences': selected_skills,
        'entreprise': entreprise,
        'type_contrat': selected_contrat,
        'lieu': selected_lieu
    }
    result = predict(pipe_model, new_data)
    st.write("Le salaire prédit est :", result[0], result[1])
