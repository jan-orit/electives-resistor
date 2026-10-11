import cv2

# "0" tells the Pi to use the first USB camera it finds
camera = cv2.VideoCapture(0) 

while True:
    # Read a single frame from the camera
    success, frame = camera.read() 
    
    # Show that frame in a window named "Smart Component Bin - Camera Test"
    cv2.imshow("Smart Component Bin - Camera Test", frame) 
    
    # Wait for 1 millisecond; if you press the 'q' key on your keyboard, it breaks the loop
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# Turn off the camera and close the window
camera.release()
cv2.destroyAllWindows()