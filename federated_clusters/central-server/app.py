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

# Simulate federating the weights (average them)
def federate_weights(weights_list):
    # Assuming weights_list contains PyTorch model states, aggregate them
    federated_weights = weights_list[0]
    for i in range(1, len(weights_list)):
        for key in federated_weights.keys():
            federated_weights[key] += weights_list[i][key]
    for key in federated_weights.keys():
        federated_weights[key] /= len(weights_list)
    return federated_weights

@app.route('/receive_weights', methods=['POST'])
def receive_weights():
    logger.info("Received a request to receive model weights.")
    
    # Ensure the weights are in the request data
    if not request.data:
        logger.error("No data received.")
        return jsonify({'error': 'No data received'}), 400

    try:
        # Deserialize model weights
        received_weights = pickle.loads(request.data)

        # Simulate federating weights (average them)
        global federated_weights
        federated_weights = federate_weights([received_weights])  # For now, using just one weight for demo

        # Generate fake statistics for weight change
        stats = {}
        for key in federated_weights.keys():
            original_weight = torch.randn_like(federated_weights[key])
            new_weight = federated_weights[key]
            weight_change = torch.abs(new_weight - original_weight).sum().item()

            stats[key] = {
                'original_mean': original_weight.mean().item(),
                'new_mean': new_weight.mean().item(),
                'change_sum': weight_change
            }

        logger.info(f"Federated weights: {federated_weights}")
        logger.info(f"Weight change statistics: {stats}")

        return jsonify({'message': 'Weights federated successfully', 'weight_change_stats': stats}), 200
    except Exception as e:
        logger.error(f"Failed to process the weights: {e}")
        return jsonify({'error': 'Processing failed'}), 500

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
