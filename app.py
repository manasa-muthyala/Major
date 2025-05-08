from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize Flask app
app = Flask(__name__)

# Load your pre-trained model (the .h5 file)
model = load_model('lstm_model.h5')
# Load the tokenizer (instead of vectorizer)
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Define max_length for padding
MAX_LENGTH = 100  # Use the same length as used in training

# Home page
@app.route('/')
def home():
    return render_template('home.html')

# Login page with username and password (both "admin")
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin':
            return redirect(url_for('input'))
        else:
            return "Invalid credentials! Please try again."
    return render_template('login.html')

# Input page to accept job description
@app.route('/input', methods=['GET', 'POST'])
def input():
    if request.method == 'POST':
        job_description = request.form['job_description']
        return redirect(url_for('result', description=job_description))
    return render_template('input.html')
# Define max_length for padding, use the same value as in training
MAX_LENGTH = 150  # Update this to match your model's expected input length

# Result page to display prediction
@app.route('/result', methods=["POST", "GET"])
def result():
    if request.method == "POST":
        # Get job description from form data
        job_description = request.form.get('job_description')
        
        if not job_description:
            return "No job description provided!", 400
        
        # Tokenize and pad the input job description
        input_sequence = tokenizer.texts_to_sequences([job_description])
        
        # Ensure padding length matches the model's expected input length
        input_padded = pad_sequences(input_sequence, maxlen=MAX_LENGTH, padding='post')
        
        # Predict the class (0 = Real, 1 = Fraudulent)
        prediction = model.predict(input_padded)
        prediction = np.round(prediction).astype(int)
        
        result = 'Fraudulent Job' if prediction[0][0] == 1 else 'Real Job'
        
        return render_template('result.html', prediction=result)

# Chart page to display any data visualizations
@app.route('/chart')
def chart():
    # You can include code here to generate charts using matplotlib or other libraries
    return render_template('chart.html')

if __name__ == '__main__':
    app.run(debug=True)


   