import cv2
import numpy as np
import screeninfo


class ROISelector:
    def __init__(self):
        self.points = []
        self.scale = 1.0
        self.display_frame = None
        self.original_frame = None

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert click back to original image coordinates
            orig_x = int(x / self.scale)
            orig_y = int(y / self.scale)

            self.points.append((orig_x, orig_y))
            print(f"Selected point: ({orig_x}, {orig_y})")

            # Draw on display image
            cv2.circle(self.display_frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select ROI", self.display_frame)

    def _resize_to_screen(self, frame):
        screen = screeninfo.get_monitors()[0]
        screen_w, screen_h = screen.width, screen.height

        h, w = frame.shape[:2]

        scale_w = screen_w / w
        scale_h = screen_h / h
        scale = min(scale_w, scale_h, 1.0)

        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h))
        return resized, scale

    def select(self, frame):
        self.original_frame = frame.copy()
        self.display_frame, self.scale = self._resize_to_screen(frame)

        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        cv2.imshow("Select ROI", self.display_frame)
        cv2.setMouseCallback("Select ROI", self._mouse_callback)

        print("Click 4 points (quadrilateral ROI). Press ENTER to confirm.")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == 13 and len(self.points) == 4:  # ENTER
                break

        cv2.destroyAllWindows()

        return np.array(self.points, dtype=np.float32)
