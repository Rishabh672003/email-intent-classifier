from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('email_intent_model.joblib')

# Intent descriptions from search results :cite[2]:cite[7]
intent_descriptions = {
    'goodbye': "Ending conversation",
    'volume control': "Adjusting audio levels",
    'play games': "Requesting entertainment",
    'covid cases': "Seeking health information",
    'open website': "Navigation request",
    'tell me joke': "Entertainment",
    'play on youtube': "Media consumption",
    'places near me': "Location-based query",
    'greet': "Greeting or check-in",
    'asking time': "Time inquiry",
    'asking date': "Date inquiry",
    'send email': "Communication request"
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    email = data['email']
    prediction = model.predict([email])[0]
    proba = np.max(model.predict_proba([email]))
    
    return jsonify({
        'intent': prediction,
        'description': intent_descriptions.get(prediction, "General inquiry"),
        'confidence': float(proba)
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)
