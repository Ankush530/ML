# knn_iris_app.py
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np # Will be needed for creating the input array

# --- 1. Load and Prepare Data (and cache it) ---
@st.cache_data # Decorator to cache the data loading and model training
def load_and_train_model():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    target_names = iris.target_names # Get target names for display

    # Split data - using a fixed random_state for consistent demo
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize & Train KNN Model
    # We can let users choose k later, or fix it for simplicity
    knn = KNeighborsClassifier(n_neighbors=3) # Default k=3
    knn.fit(X_train, y_train)

    # Calculate accuracy on the test set (for display)
    y_pred_test = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    
    return knn, X.columns, target_names, accuracy, X # Return X for min/max values

# Load the model and related data
knn_model, feature_names, iris_target_names, model_accuracy, X_data = load_and_train_model()

# --- 2. Streamlit App Interface ---
st.set_page_config(page_title="Iris KNN Classifier", layout="wide")

st.title("🌸 Iris Flower Species Classifier using KNN")
st.write(f"This app uses a K-Nearest Neighbors (KNN) model to predict the species of an Iris flower based on its sepal and petal measurements. The underlying model was trained on the Iris dataset and has an accuracy of **{model_accuracy*100:.2f}%** on its test set.")
st.markdown("---")

# --- 3. User Input for Features ---
st.sidebar.header("Input Flower Measurements (cm):")

# Create input fields for each feature in the sidebar
input_features = {}
for feature in feature_names:
    # Get min and max for sliders from the original dataset X_data
    min_val = float(X_data[feature].min())
    max_val = float(X_data[feature].max())
    default_val = float(X_data[feature].mean()) # Use mean as default

    input_features[feature] = st.sidebar.slider(
        label=feature.replace('_', ' ').capitalize(), # Make labels nice
        min_value=min_val,
        max_value=max_val,
        value=default_val, # Default to the mean value
        step=0.1
    )

# --- 4. Prediction and Display ---
if st.sidebar.button("Predict Species 🔎", type="primary"):
    # Create a NumPy array from the input features in the correct order
    # The model expects a 2D array, so we reshape
    input_data_array = np.array([input_features[fn] for fn in feature_names]).reshape(1, -1)

    # Make prediction
    prediction_index = knn_model.predict(input_data_array)
    predicted_species = iris_target_names[prediction_index[0]]
    prediction_proba = knn_model.predict_proba(input_data_array)

    # Display the prediction
    st.subheader("Prediction Result:")
    if predicted_species == "setosa":
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Kosaciec_syberyjski_Iris_sibirica.jpg/440px-Kosaciec_syberyjski_Iris_sibirica.jpg", caption=f"Predicted: {predicted_species.capitalize()}", width=200)
    elif predicted_species == "versicolor":
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Iris_versicolor_3.jpg/440px-Iris_versicolor_3.jpg", caption=f"Predicted: {predicted_species.capitalize()}", width=200)
    elif predicted_species == "virginica":
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/440px-Iris_virginica.jpg", caption=f"Predicted: {predicted_species.capitalize()}", width=200)
    
    st.success(f"The model predicts the Iris species as: **{predicted_species.capitalize()}**")

    st.write("Prediction Probabilities:")
    # Create a DataFrame for better display of probabilities
    proba_df = pd.DataFrame(prediction_proba, columns=iris_target_names)
    st.dataframe(proba_df.style.format("{:.2%}")) # Format as percentage

    # Display input values for confirmation
    st.write("Input Values Provided:")
    input_df = pd.DataFrame([input_features])
    st.dataframe(input_df.style.format("{:.1f} cm"))

else:
    st.info("Adjust the sliders in the sidebar and click 'Predict Species' to see the classification.")

st.markdown("---")
st.subheader("About the Model & Data")
st.write("""
- **Dataset:** Iris flower dataset (classic dataset in machine learning).
- **Features Used:** Sepal Length, Sepal Width, Petal Length, Petal Width (all in cm).
- **Target Variable:** Iris Species (Setosa, Versicolor, Virginica).
- **Algorithm:** K-Nearest Neighbors (KNN) with k=3.
- **Purpose:** Educational demo for a Machine Learning workshop.
""")

# Optional: Allow user to change k and retrain (more advanced)
# For simplicity in a demo, k is fixed in load_and_train_model()
# If you want to add this:
# k_value = st.sidebar.number_input("Select K for KNN", min_value=1, max_value=15, value=3, step=2)
# if st.sidebar.button("Retrain Model with new K"):
#     # Need to modify load_and_train_model to accept k
#     # And remove @st.cache_data or handle caching appropriately if k changes
#     st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.info("Built by Student for the ML Workshop.")