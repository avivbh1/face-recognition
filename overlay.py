import cv2

GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_face_box(frame, bbox, match, score):
    color = GREEN if match else RED
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{'MATCH' if match else 'NO MATCH'}  {score:.2f}"
    cv2.putText(frame, label, (x1, y1 - 10), FONT, 0.7, color, 2)


def draw_status(frame, text, color=WHITE):
    cv2.putText(frame, text, (10, 30), FONT, 0.8, color, 2)


def draw_enroll_progress(frame, count, total):
    msg = f"Enrolling: {count}/{total} samples — keep still"
    cv2.putText(frame, msg, (10, 30), FONT, 0.8, YELLOW, 2)
