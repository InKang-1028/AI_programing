# Classify webcam frames with a trained model. Press q to quit.

from tensorflow.keras.models import load_model  # Load a saved Keras model.
import cv2  # Camera and image operations.
import numpy as np  # Array operations.

# Suppress scientific notation in NumPy output.
np.set_printoptions(suppress=True)

# Load the trained model for inference.
model = load_model("./model_4_c200_m200/keras_model.h5", compile=False)

# Load labels in the same order as the model outputs.
class_names = open("./model_4_c200_m200/labels.txt", "r").readlines()

# Open the default camera.
camera = cv2.VideoCapture(0)

while True:
    # Read a frame; successful capture is assumed.
    ret, image_ = camera.read()

    # Resize to the model input size, keeping the original for display.
    image = cv2.resize(image_, (224, 224), interpolation=cv2.INTER_AREA)

    # Convert to float32 and add a batch dimension: (1, 224, 224, 3).
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Scale pixels to [-1, 1]. The BGR order must match training.
    image = (image / 127.5) - 1

    # Predict class scores and select the highest-scoring class.
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display the class name, removing its single-digit index and whitespace.
    image_    = cv2.putText(image_, class_name[2:].strip(), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    # Display the confidence as a rounded percentage.
    image_    = cv2.putText(image_, str(np.round(confidence_score * 100))[:-2]+"%", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.imshow("Webcam Image", image_)

    # Print the prediction and confidence.
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Update the window and quit when q is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows.
camera.release()
cv2.destroyAllWindows()
