from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Paths
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load your trained model
generator = load_model(r"F:\My_Projects\Sih\WebInterface\model\generator.h5")

# Helper functions
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((256, 256))
    image = np.array(image) / 127.5 - 1.0
    return np.expand_dims(image, axis=0)  # Shape (1, 256, 256, 1)

def save_result_image(result_image, result_path):
    result_image = (result_image + 1.0) * 127.5
    result_image = np.clip(result_image, 0, 255).astype(np.uint8)
    result_image = Image.fromarray(result_image[0])
    result_image.save(result_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        # Process image
        input_image = preprocess_image(file_path)
        generated_image = generator.predict(input_image)
        
        # Save result
        result_path = os.path.join(RESULT_FOLDER, 'colorized_image.png')
        save_result_image(generated_image, result_path)
        
        return jsonify({
            'colorized_image': 'results/colorized_image.png'
        })

@app.route('/results/<filename>')
def serve_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
