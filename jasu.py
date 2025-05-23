import cv2
import numpy as np

def detect_traffic_lights(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red mask
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)

    # Yellow mask
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # Green mask
    green_lower = np.array([40, 100, 100])
    green_upper = np.array([70, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # Find color with largest area
    if cv2.countNonZero(red_mask) > 500:
        cv2.putText(frame, 'Red Light', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    elif cv2.countNonZero(yellow_mask) > 500:
        cv2.putText(frame, 'Yellow Light', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    elif cv2.countNonZero(green_mask) > 500:
        cv2.putText(frame, 'Green Light', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(frame, 'No Light Detected', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    return frame

if __name__ == "__main__":
    image_path = "traffic.jpeg"
    frame = cv2.imread(image_path)

    if frame is None:
        print("Error: Could not load image")
    else:
        output = detect_traffic_lights(frame)
        cv2.imshow("Traffic Light Detection", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()