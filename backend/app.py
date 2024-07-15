from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import pickle

app = Flask(__name__)
CORS(app)

# Load the label encoders for features
with open('label_encoder_X.pkl', 'rb') as f:
    label_encoder_X = pickle.load(f)

# Load the label encoder for the target
with open('label_encoder_Y.pkl', 'rb') as f:
    label_encoder_Y = pickle.load(f)

# Load the saved model
loaded_model = joblib.load('best_xgb_model.pkl')

# Function to preprocess input data for testing
def preprocess_input(input_data, label_encoders_X):
    encoded_input = []
    for col in ['Age', 'Gender', 'Lifestyle', 'Habit 01', 'Habit 02', 'Habit 03', 'Habit 04', 'Habit 05', 'Habit 06']:
        if col in label_encoders_X:
            # Use the corresponding LabelEncoder to transform feature variables
            encoded_value = label_encoders_X[col].transform([input_data[col]])
            encoded_input.append(encoded_value[0])
        else:
            # If the column is not categorical, use the value as is
            encoded_input.append(input_data[col])
    return np.array(encoded_input).reshape(1, -1)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Ensure input data is properly encoded
    preprocessed_input = preprocess_input(data, label_encoder_X).astype(float)  # Convert to float

    predictions = loaded_model.predict(preprocessed_input)  # Make predictions using the loaded model
    
    # Get the predicted class
    predicted_class = label_encoder_Y.inverse_transform(predictions)
    
    return jsonify({'prediction': predicted_class[0]})

if __name__ == '__main__':
    app.run(debug=True)
