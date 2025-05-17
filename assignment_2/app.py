from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

#  Flask app Initialization
app = Flask(__name__)

# Loading the saved model
model = load_model('model_vgg.h5')

# image size
sz = 224

# direction mapping
direction_mapping = {
    0: "South",
    1: "North-West",
    2: "North",
    3: "North-East",
    4: "East",
    5: "South-East",
    6: "South-West",
    7: "West"
}

# Prediction function
def predict_car_angle(image_path):
    # Loading and preprocessing the image
    image = load_img(image_path, target_size=(sz, sz))
    image = img_to_array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    # Making prediction
    predictions = model.predict(image)
    
    # Getting the predicted class (car angle) and confidence score
    predicted_class = np.argmax(predictions[0])
    confidence_score = np.max(predictions[0])
    
    # Getting the corresponding direction
    direction = direction_mapping.get(predicted_class, "Unknown direction")
    
    return predicted_class, confidence_score, direction

# Defining a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file:
        # Saving the file to disk
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        # Getting the predicted car angle, confidence, and direction
        predicted_class, confidence_score, direction = predict_car_angle(file_path)
        
        # Deleting the uploaded file after prediction
        os.remove(file_path)
        
        # Returning the prediction as a JSON response
        return jsonify({
            "predicted_class": int(predicted_class),
            "confidence_score": float(confidence_score),
            "direction": direction
        })

# Running the app
if __name__ == '__main__':
    app.run(debug=True)
