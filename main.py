import cv2
from ultralytics import YOLO
import cvzone
import math



# --- 1. CONFIGURATION ---
# Initialize video capture.
# Replace the path with 0 to use your webcam: cv2.VideoCapture(0)
cap=cv2.VideoCapture(0)

# Load your custom-trained YOLOv8 model
# 'ppt-detection.pt' contains the weights learned from construction site dataset
model=YOLO('ppt-detection.pt')

# Move the model to the GPU (CUDA) for faster real-time inference
# This significantly boosts FPS compared to running on CPU
#if you don't have nvidia GPU replace 'cuda' with 'cpu'
#if you have Apple M-series chip replace 'cuda' with 'mps'
model.to('cuda')

# --- 2. INFERENCE LOOP ---
while True:
    # Read the next frame from the video
    success, frame = cap.read()

    # Break the loop if the video ends or cannot be read
    if not success:
        break

    # Run YOLOv8 inference on the current frame
    # stream=True is efficient for processing long videos as it uses a generator
    results=model(frame,stream=True)

    # Loop through the results (usually one result per frame)
    for result in results:

        # Loop through each bounding box detected in the frame
        for box in result.boxes:

            # --- 3. COORDINATES & VISUALIZATION ---
            # Get the coordinates: x1, y1 (top-left), x2, y2 (bottom-right)
            x1,y1,x2,y2=box.xyxy[0]

            # Convert coordinates to integer and calculate width/height for cvzone
            bbox=[int(x1),int(y1),int(x2-x1),int(y2-y1)]



            # Get the Class Name and Confidence Score
            class_index = int(box.cls[0])
            class_name = model.names[class_index]

            if class_name in ['NO-Hardhat','NO-Mask','NO-Safety Vest']:
                color=(0, 0, 255)
            elif class_name in ['Hardhat','Mask','Gloves','Safety Vest']:
                color=(0, 255, 0)
            else:
                color = (255, 0, 255)


            conf_score = math.ceil((box.conf[0] * 100)) / 100  # Round to 2 decimal places

            # Draw a sophisticated corner rectangle around the object
            cvzone.cornerRect(frame, bbox,colorR=color)

            # Display the Class Name and Confidence Score above the box
            cvzone.putTextRect(
                frame,
                f"{class_name} {conf_score}",
                (max(0, int(x1)), int(y1)),
                scale=1,
                thickness=1,
                colorR=color
            )

    # --- 4. DISPLAY output---
    # Show the frame with detections in a window
    cv2.imshow("frame",frame)

    # Wait for 1ms to allow the window to refresh
    # This loop runs as fast as the GPU can process frames
    cv2.waitKey(1)

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()




