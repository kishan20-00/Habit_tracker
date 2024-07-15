import requests

# Define the URL of the prediction endpoint
url = 'http://127.0.0.1:5000/predict'

# Define the input data
input_data = {
    'Age': 25,
    'Gender': 'Male',
    'Lifestyle': 'Freelancer',
    'Habit 01': 'Take a walk outside',
    'Habit 02': 'Yoga',
    'Habit 03': 'Sleep at least 8 hours ',
    'Habit 04': 'Exercise',
    'Habit 05': 'Pray',
    'Habit 06': 'Be Grateful'
}

# Send a POST request to the prediction endpoint
response = requests.post(url, json=input_data)

# Check the status code of the response
if response.status_code == 200:
    # Get the prediction from the response
    prediction = response.json()
    print("Predicted Recommendation:", prediction['Predicted Recommendation'])
else:
    print("Error:", response.status_code)
    print(response.text)
