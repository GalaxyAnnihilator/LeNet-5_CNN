import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("lenet5_model.h5")

# Class labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def predict_image(img_path):
    """Preprocess and predict image class."""
    img = image.load_img(img_path, target_size=(32, 32))  # Resize to match model input
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    
    return class_names[predicted_class], predictions[0][predicted_class]

if __name__ == "__main__":
    img_path = "test_image.jpg"  # Replace with any image path
    predicted_label, confidence = predict_image(img_path)
    print(f"Predicted: {predicted_label} ({confidence:.2f} confidence)")
