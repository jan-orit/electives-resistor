from ultralytics import YOLO
import cv2

# Load the lightweight pre-trained YOLOv8 nano model
# (It will automatically download the 'yolov8n.pt' file the very first time it runs)
model = YOLO('yolov8n.pt')

# Connect to the USB webcam
camera = cv2.VideoCapture(0)

while True:
    # Capture the video frame
    success, frame = camera.read()
    if not success:
        print("Failed to grab frame")
        break
        
    # Run YOLOv8 inference on the current frame
    results = model(frame)
    
    # Extract the frame with the AI bounding boxes drawn on it
    annotated_frame = results[0].plot()
    
    # Display the annotated video feed
    cv2.imshow("Smart Component Bin - YOLOv8 Test", annotated_frame)
    
    # Wait for 1 millisecond; press 'q' to break the loop and close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Safely turn off the camera and close the window
camera.release()
cv2.destroyAllWindows()