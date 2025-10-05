from flask import Flask, request, render_template, jsonify
import joblib # Import joblib instead of pickle

# Initialize the Flask application
app = Flask(__name__)

# Load the vectorizer and model using joblib
try:
    vectorizer = joblib.load('vectorizer.pkl')
    model = joblib.load('model.pkl')
except FileNotFoundError:
    print("Error: Model or vectorizer files not found.")
    exit()

# ... (the rest of your app.py code remains exactly the same) ...

# Route for the main page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling the prediction
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['news_text']
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)
    return jsonify({'category': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)