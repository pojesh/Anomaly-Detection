from ultralytics import YOLO 
import tensorflow as tf

# Initialize the YOLO model with your trained weights
model = YOLO('./runs/classify/train25/weights/last.pt')


# Export the YOLO model to a TensorFlow SavedModel
saved_model_path = 'yolo_saved_model'

# Create a TensorFlow model
tf_model = model.model()

# Save the TensorFlow model as a SavedModel
tf.saved_model.save(model, saved_model_path)

# Convert the SavedModel to TensorFlow Lite (TFLite)
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('yolo_model.tflite', 'wb') as f:
    f.write(tflite_model)