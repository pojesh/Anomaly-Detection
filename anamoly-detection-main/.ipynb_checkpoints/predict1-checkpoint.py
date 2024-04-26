from ultralytics import YOLO
import numpy as np
import cv2

# Load the YOLO model
model = YOLO("./runs/classify/train43/weights/best.pt")

# Open the webcam
cap = cv2.VideoCapture(1)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Predict on the frame using YOLO
    results = model(frame)

    # Extract prediction information
    names_dict = results[0].names

    probs = results[0].probs.data.tolist()

    print(names_dict)
    print(probs)

    output = names_dict[np.argmax(probs)]

    if output == "no-fire" or output == "no-fight":
        output = "normal"

    # Display the label as text at the bottom center of the frame
    text_size = cv2.getTextSize(output, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = frame.shape[0] - 10

    cv2.putText(
        frame, output, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
    )

    # Display the frame with class labels
    cv2.imshow("YOLO Predictions", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
