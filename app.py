import pandas as pd
import joblib
import streamlit as st

# Load the encoders and the model
onehot_encoder = joblib.load('onehot_encoder.joblib')
minmax_scaler = joblib.load('minmax_scaler.joblib')
model = joblib.load('best_boosting_model_tuned.joblib')

def preprocess_data(df):
    """
    Preprocesses the input DataFrame by dropping columns, applying one-hot encoding,
    and scaling.

    Args:
        df: Input pandas DataFrame with raw data.

    Returns:
        A preprocessed pandas DataFrame ready for model prediction.
    """
    # Drop specified columns
    df = df.drop(['ID', 'Año - Semestre'], axis=1)

    # Apply one-hot encoding to 'Felder'
    felder_encoded = onehot_encoder.transform(df[['Felder']])
    felder_encoded_df = pd.DataFrame(felder_encoded, columns=onehot_encoder.get_feature_names_out(['Felder']))

    # Apply MinMaxScaler to 'Examen_admisión_Universidad'
    examen_scaled = minmax_scaler.transform(df[['Examen_admisión_Universidad']].values)
    examen_scaled_df = pd.DataFrame(examen_scaled, columns=['Examen_admisión_Universidad_scaled'])

    # Concatenate the processed features
    processed_df = pd.concat([felder_encoded_df, examen_scaled_df], axis=1)

    return processed_df

def make_predictions(processed_df):
    """
    Makes predictions using the loaded model on the preprocessed data.

    Args:
        processed_df: Preprocessed pandas DataFrame.

    Returns:
        An array of predictions.
    """
    predictions = model.predict(processed_df)
    return predictions

st.title('Course Approval Prediction App')

uploaded_file = st.file_uploader("Upload your input Excel file", type=["xlsx"])

if uploaded_file is not None:
    df_input = pd.read_excel(uploaded_file, sheet_name=1)
    st.write("Uploaded Data:")
    st.dataframe(df_input.head())

    if st.button('Predict Course Approval'):
        processed_data = preprocess_data(df_input.copy())
        predictions = make_predictions(processed_data)
        st.write("Predictions:")
        st.write(predictions)

st.markdown("""
This application predicts the likelihood of course approval based on student data.

**How to use:**
1. Upload an Excel file containing student data. The file should have a sheet named "Sheet2" (or the second sheet) with columns including 'ID', 'Año - Semestre', 'Felder', and 'Examen_admisión_Universidad'.
2. Click the 'Predict Course Approval' button to get the predictions.

The model used for prediction is a Gradient Boosting Regressor, trained on historical student data. The prediction indicates whether a student is likely to approve ('si') or not approve ('no') the course.
""")
