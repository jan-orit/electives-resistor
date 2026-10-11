import cv2
import numpy as np

current_strip = None

def get_hsv_on_click(event, x, y, flags, param):
    global current_strip
    if event == cv2.EVENT_LBUTTONDOWN and current_strip is not None:
        orig_x = int(x / 2)
        b, g, r = current_strip[0, orig_x]
        hsv_pixel = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        print(f"Clicked Band -> Hue: {hsv_pixel[0]:03d} | Saturation: {hsv_pixel[1]:03d} | Value: {hsv_pixel[2]:03d}")

def nothing(x):
    pass

def get_color_name(h, s, v):
    # 1. Black Band (Value strictly under 50 based on your data)
    if v < 50: return "Black"
        
    # 2. Beige Body Filter (H:12-20, S:75-105, V:165-205)
    if 12 <= h <= 20 and 75 <= s <= 105 and 165 <= v <= 205: return "Body"
        
    # 3. Gold Tolerance Filter (H:14-20, S:108-130, V:115-160)
    if 14 <= h <= 20 and 108 <= s <= 130 and 115 <= v <= 160: return "Gold"

    # 4. Red Band (H:0-5 or H:170-179, S:160-255, V:95-145)
    if (h <= 5 or h >= 170) and s >= 160 and 95 <= v <= 145: return "Red"
        
    # 5. Standard fallbacks for the remaining colors
    if 10 <= h <= 20 and s > 150 and v < 100: return "Brown"
    if 11 <= h <= 25 and s > 150: return "Orange"
    if 26 <= h <= 35: return "Yellow"
    if 36 <= h <= 80 and v > 50: return "Green"
    if 81 <= h <= 130 and v > 50: return "Blue"
    if 131 <= h <= 165 and v > 50: return "Purple"
    
    return "Unknown"

color_math = {
    'Black': 0, 'Brown': 1, 'Red': 2, 'Orange': 3, 
    'Yellow': 4, 'Green': 5, 'Blue': 6, 'Purple': 7,
    'Gold': -1 # Retained in dictionary in case of future 5-band expansion
}

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("Tuner", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tuner", 400, 250)
cv2.createTrackbar("Threshold", "Tuner", 100, 255, nothing)
cv2.createTrackbar("Invert (0/1)", "Tuner", 1, 1, nothing)
cv2.createTrackbar("Horiz Bridge", "Tuner", 35, 120, nothing)
cv2.createTrackbar("Erode Leads", "Tuner", 12, 40, nothing)

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

                    avg_bgr_per_col = np.mean(center_slice, axis=0).astype(np.uint8)
                    
                    bgr_bar = np.zeros((40, cw, 3), dtype=np.uint8)
                    for col in range(cw):
                        bgr_bar[:, col] = avg_bgr_per_col[col]
                    
                    current_strip = bgr_bar

                    bgr_strip_1d = np.zeros((1, cw, 3), dtype=np.uint8)
                    bgr_strip_1d[0] = avg_bgr_per_col
                    hsv_strip_1d = cv2.cvtColor(bgr_strip_1d, cv2.COLOR_BGR2HSV)
                    avg_hsv_per_col = hsv_strip_1d[0]

                    # --- NOISE FILTERING & DEBOUNCING ---
                    detected_bands = []
                    current_color = None
                    color_streak = 0
                    MIN_THICKNESS = 3 
                    
                    start_col = int(cw * 0.1)
                    end_col = int(cw * 0.9)

                    for col in range(start_col, end_col):
                        h_val, s_val, v_val = avg_hsv_per_col[col]
                        color = get_color_name(h_val, s_val, v_val)

                        if color == current_color:
                            color_streak += 1
                        else:
                            # Filters out Gold so it acts purely as tolerance for 4-band components
                            if current_color not in ["Body", "Background", "Unknown", "Gold", None]:
                                if color_streak >= MIN_THICKNESS:
                                    if not detected_bands or detected_bands[-1] != current_color:
                                        detected_bands.append(current_color)
                            
                            current_color = color
                            color_streak = 1

                    if current_color not in ["Body", "Background", "Unknown", "Gold", None] and color_streak >= MIN_THICKNESS:
                        if not detected_bands or detected_bands[-1] != current_color:
                            detected_bands.append(current_color)
                    # ------------------------------------

                    if len(detected_bands) >= 3:
                        b1, b2, b3 = detected_bands[0], detected_bands[1], detected_bands[2]
                        
                        if b1 in color_math and b2 in color_math and b3 in color_math:
                            val1 = color_math[b1]
                            val2 = color_math[b2]
                            multiplier = 10 ** color_math[b3]
                            
                            ohms = (val1 * 10 + val2) * multiplier
                            
                            display_val = f"{ohms:.1f} Ohms" if ohms < 1000 else f"{ohms} Ohms"
                            if ohms >= 1000000:
                                display_val = f"{ohms/1000000:.1f}M Ohms"
                            elif ohms >= 1000:
                                display_val = f"{ohms/1000:.1f}k Ohms"

                            cv2.putText(annotated, display_val, (x1, max(y1 - 30, 20)), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
                            cv2.putText(annotated, f"[{b1}-{b2}-{b3}]", (x1, max(y1 - 10, 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    cv2.imshow("Straightened Body", cv2.resize(crop_display, (cw * 2, ch * 2)))
                    cv2.imshow("Sampled Color Strip", cv2.resize(bgr_bar, (cw * 2, 40)))

    cv2.imshow("Smart Component Bin", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()