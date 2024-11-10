from flask import Flask, jsonify, request
from flask_cors import CORS
import hackfin  # This should be your modified Python script that includes get_predictions function

app = Flask(__name__)
CORS(app)  # This is necessary for cross-origin requests from your React app

@app.route('/predict', methods=['GET'])
def predict():
    ticker = request.args.get('ticker', 'NVDA')  # Default ticker
    # Assuming you modify hackfin to include a function that handles fetching and processing data
    predictions, metrics = hackfin.get_predictions(ticker)
    return jsonify({
        'predictions': predictions,
        'metrics': metrics
    })

if __name__ == "__main__":
    app.run(debug=True)
