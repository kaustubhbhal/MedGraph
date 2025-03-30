from flask import Flask, request, jsonify
import torch
import io
import logging
import pickle

app = Flask(__name__)

# Set up logging for better visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example federated model weights
federated_weights = None

# Federate the weights (average them)
def federate_weights(weights_list):
    federated_weights = weights_list[0]
    for i in range(1, len(weights_list)):
        for key in federated_weights.keys():
            federated_weights[key] += weights_list[i][key]
    for key in federated_weights.keys():
        federated_weights[key] /= len(weights_list)
    return federated_weights

# Decrypt the weights (Removed encryption logic)
def load_model_weights(serialized_weights):
    """Deserialize the model weights and return as a dictionary."""
    return pickle.loads(serialized_weights)

@app.route('/receive_weights', methods=['POST'])
def receive_weights():
    logger.info("Received a request to receive model weights.")
    
    # Ensure the weights are in the request data
    if not request.data:
        logger.error("No data received.")
        return jsonify({'error': 'No data received'}), 400

    serialized_weights = request.data
    try:
        model_weights = load_model_weights(serialized_weights)
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        return jsonify({'error': 'Failed to load weights'}), 500

    # Simulate federating weights (average them)
    global federated_weights
    federated_weights = federate_weights([model_weights])  # For now, using just one weight for demo

    logger.info(f"Federated weights: {federated_weights}")

    return jsonify({'message': 'Weights federated successfully'}), 200

@app.route('/get_federated_model', methods=['GET'])
def get_federated_model():
    if federated_weights is None:
        return jsonify({'error': 'Federated model not ready yet'}), 400

    # Serialize federated model weights and send them back
    serialized_weights = pickle.dumps(federated_weights)
    return serialized_weights

if __name__ == '__main__':
    logger.info("Starting Flask server on port 5000...")
    app.run(debug=True, host='0.0.0.0', port=5001)
