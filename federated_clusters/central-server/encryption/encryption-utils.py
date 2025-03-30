from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import torch
import io

def encrypt_weights(model_weights, key):
    iv = b'0123456789abcdef'  # 16-byte IV for AES
    # Serialize model weights to bytes
    buffer = io.BytesIO()
    torch.save(model_weights, buffer)
    buffer.seek(0)
    model_weights_bytes = buffer.read()
    
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    encryptor = cipher.encryptor()
    encrypted_weights = encryptor.update(model_weights_bytes) + encryptor.finalize()
    return encrypted_weights

def decrypt_weights(encrypted_weights, key):
    iv = b'0123456789abcdef'  # 16-byte IV for AES
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    decrypted_weights = decryptor.update(encrypted_weights) + decryptor.finalize()

    # Load the decrypted bytes into a PyTorch model
    model_weights = torch.load(io.BytesIO(decrypted_weights))
    return model_weights
