import cv2
import numpy as np

def nothing(x):
    pass

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("Tuner", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tuner", 400, 300)
cv2.createTrackbar("Threshold", "Tuner", 100, 255, nothing)
cv2.createTrackbar("Invert (0/1)", "Tuner", 1, 1, nothing)
cv2.createTrackbar("Horiz Bridge", "Tuner", 35, 120, nothing)
cv2.createTrackbar("Erode Leads", "Tuner", 12, 40, nothing)

while True:
    success, frame = camera.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    thresh_val = cv2.getTrackbarPos("Threshold", "Tuner")
    invert_val = cv2.getTrackbarPos("Invert (0/1)", "Tuner")
    bridge_val = max(1, cv2.getTrackbarPos("Horiz Bridge", "Tuner"))
    erode_val = max(1, cv2.getTrackbarPos("Erode Leads", "Tuner"))

    thresh_type = cv2.THRESH_BINARY_INV if invert_val == 1 else cv2.THRESH_BINARY
    _, thresh = cv2.threshold(blurred, thresh_val, 255, thresh_type)

    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_val, 5))
    closed_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, bridge_kernel)

    lead_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_val, erode_val))
    body_only_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, lead_kernel)

    isolated_colors = cv2.bitwise_and(frame, frame, mask=body_only_mask)

    contours, _ = cv2.findContours(body_only_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated = frame.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 3000:
            # 1. Get the Rotated Minimum Area Rectangle
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect
            
            # 2. Filter out squares, keeping elongated resistor shapes
            aspect_ratio = max(w, h) / max(min(w, h), 1)
            
            if 1.1 < aspect_ratio < 6.0:
                # 3. Draw the tilted bounding box tracking the actual angle
                box = cv2.boxPoints(rect)
                box = np.int32(box) 
                cv2.drawContours(annotated, [box], 0, (0, 255, 0), 2)
                
                # 4. Force horizontal orientation for the math
                if h > w:
                    w, h = h, w
                    angle += 90
                
                # 5. Rotate the image around the center to straighten the resistor
                M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                rotated = cv2.warpAffine(isolated_colors, M, (frame.shape[1], frame.shape[0]))
                
                # 6. Crop the perfectly flat resistor out of the rotated frame
                y1 = max(0, int(cy - h/2))
                y2 = min(frame.shape[0], int(cy + h/2))
                x1 = max(0, int(cx - w/2))
                x2 = min(frame.shape[1], int(cx + w/2))
                
                resistor_crop = rotated[y1:y2, x1:x2]
                
                if resistor_crop.size > 0:
                    cv2.imshow("Straightened Resistor (Ready for HSV)", resistor_crop)

    cv2.imshow("Original with Detection", annotated)
    cv2.imshow("Tuned Mask", body_only_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()