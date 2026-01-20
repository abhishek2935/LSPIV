import cv2
import numpy as np

points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select ROI", frame)

# Load first frame
video = cv2.VideoCapture("Videos/good_river.mp4")
ret, frame = video.read()
video.release()

if not ret:
    print("Frame not read")
    exit()

clone = frame.copy()

cv2.imshow("Select ROI", frame)
cv2.setMouseCallback("Select ROI", mouse_callback)

# Wait until 4 points are selected
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 13 and len(points) == 4:  # ENTER key
        break

cv2.destroyAllWindows()

# Perspective transform
pts = np.array(points, dtype="float32")

width = int(max(
    np.linalg.norm(pts[1] - pts[0]),
    np.linalg.norm(pts[2] - pts[3])
))

height = int(max(
    np.linalg.norm(pts[2] - pts[1]),
    np.linalg.norm(pts[3] - pts[0])
))

dst = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
], dtype="float32")

matrix = cv2.getPerspectiveTransform(pts, dst)
roi = cv2.warpPerspective(clone, matrix, (width, height))

cv2.imwrite("roi_corrected.jpg", roi)

cv2.imshow("Cropped ROI", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
