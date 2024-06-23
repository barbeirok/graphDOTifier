from io import BytesIO
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# Define and register the custom loss function
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


# Register the custom loss function
tf.keras.utils.get_custom_objects().update({'mse': mse})

# Load the model with the custom loss function
model = tf.keras.models.load_model('D:/Projetos/SistemasSensiveisAoContexto/proj/100_3x3_1c.h5',
                                   custom_objects={'mse': mse})

# Define image size (adjust as needed)
img_size = (240, 240)


# Function to predict on a new image
def predict_image(img):
    img = img / 255.0  # Normalization
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    class_predictions, bbox_predictions = model.predict(img, verbose=0)
    bbox_predictions = bbox_predictions.reshape(-1, 4)  # Adjust the shape of bounding box predictions

    return class_predictions, bbox_predictions


def visualize_predictions(img, class_predictions, bbox_predictions):
    for i in range(len(bbox_predictions)):
        confidence = class_predictions[0][0]  # Only one class, so take the single prediction
        bbox = bbox_predictions[i]

        x_center, y_center, width, height = bbox
        x_center *= img_size[0]
        y_center *= img_size[1]
        width *= img_size[0]
        height *= img_size[1]

        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        color = (255, 0, 0)  # Red
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f'Class: 0, Conf: {confidence:.2f}'  # Adjust for class 0
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Print the bounding box coordinates
        print(f"Bounding Box {i}: x1={x1}, y1={y1}, x2={x2}, y2={y2}, Confidence={confidence:.2f}")

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def resize(graph):
    print("graph - ", graph)
    image_data = graph.read()
    print("image_data - ", image_data)
    image = Image.open(BytesIO(image_data))
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, img_size)
    return img

