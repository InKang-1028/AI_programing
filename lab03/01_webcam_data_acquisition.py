# Capture and save webcam frames. Press q to quit.

import cv2  # Camera and image operations.
import os  # File and folder operations.

# Open the default camera.
camera = cv2.VideoCapture(0)

# Create the output folder in the working directory.
output_folder = "./captured_images"
os.makedirs(output_folder, exist_ok=True)

# Start numbering at 1; existing files with the same names are overwritten.
count = 1

while True:
    # Read a frame; ret indicates success.
    ret, frame = camera.read()

    if not ret:
        print("cannot read the frame.")
        break

    # Use six-digit filenames, e.g., 000001.jpg.
    filename = f"{count:06d}.jpg"
    save_path = os.path.join(output_folder, filename)

    # Save and display each frame.
    cv2.imwrite(save_path, frame)
    cv2.imshow("Webcam", frame)

    # Update the window and quit when q is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    count += 1

# Release the camera and close all windows.
camera.release()
cv2.destroyAllWindows()
