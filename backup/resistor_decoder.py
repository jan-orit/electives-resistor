import cv2
import numpy as np

# Global variable to store the strip for the mouse click
current_strip = None

def get_hsv_on_click(event, x, y, flags, param):
    global current_strip
    if event == cv2.EVENT_LBUTTONDOWN and current_strip is not None:
        # Scale the X coordinate back down since we doubled the window size for visibility
        orig_x = int(x / 2)
        
        # Grab the BGR color of the clicked pixel and convert it to HSV
        b, g, r = current_strip[0, orig_x]
        hsv_pixel = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        print(f"Clicked Band -> Hue: {hsv_pixel[0]:03d} | Saturation: {hsv_pixel[1]:03d} | Value: {hsv_pixel[2]:03d}")

def nothing(x):
    pass

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("Tuner", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tuner", 400, 250)
cv2.createTrackbar("Threshold", "Tuner", 100, 255, nothing)
cv2.createTrackbar("Invert (0/1)", "Tuner", 1, 1, nothing)
cv2.createTrackbar("Horiz Bridge", "Tuner", 35, 120, nothing)
cv2.createTrackbar("Erode Leads", "Tuner", 12, 40, nothing)

# Attach the mouse click event to the color strip window
cv2.namedWindow("Sampled Color Strip")
cv2.setMouseCallback("Sampled Color Strip", get_hsv_on_click)

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
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect
            aspect_ratio = max(w, h) / max(min(w, h), 1)

            if 1.1 < aspect_ratio < 6.0:
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cv2.drawContours(annotated, [box], 0, (0, 255, 0), 2)

                if h > w:
                    w, h = h, w
                    angle += 90

                M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                rotated = cv2.warpAffine(isolated_colors, M, (frame.shape[1], frame.shape[0]))

                y1 = max(0, int(cy - h/2))
                y2 = min(frame.shape[0], int(cy + h/2))
                x1 = max(0, int(cx - w/2))
                x2 = min(frame.shape[1], int(cx + w/2))

                resistor_crop = rotated[y1:y2, x1:x2]

                if resistor_crop.size > 0 and resistor_crop.shape[0] > 10 and resistor_crop.shape[1] > 10:
                    ch, cw = resistor_crop.shape[:2]
                    mid_y1 = int(ch * 0.3)
                    mid_y2 = int(ch * 0.7)
                    center_slice = resistor_crop[mid_y1:mid_y2, :]

                    crop_display = resistor_crop.copy()
                    cv2.rectangle(crop_display, (0, mid_y1), (cw, mid_y2), (0, 255, 255), 1)

                    # Average in BGR space first to prevent circular color wrapping
                    avg_bgr_per_col = np.mean(center_slice, axis=0).astype(np.uint8)
                    
                    bgr_bar = np.zeros((40, cw, 3), dtype=np.uint8)
                    for col in range(cw):
                        bgr_bar[:, col] = avg_bgr_per_col[col]
                    
                    # Update global variable for the mouse click event
            
                    current_strip = bgr_bar

                    cv2.imshow("Straightened Body", cv2.resize(crop_display, (cw * 2, ch * 2)))
                    cv2.imshow("Sampled Color Strip", cv2.resize(bgr_bar, (cw * 2, 40)))

    cv2.imshow("Original with Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()