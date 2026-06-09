import argparse
import cv2
import numpy as np
from verifier import FaceVerifier
from reference import save_reference, load_reference
from overlay import draw_face_box, draw_status, draw_enroll_progress

ENROLL_SAMPLES = 5
RECOGNITION_INTERVAL = 5  # frames


def enroll(verifier):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open camera.")
        return

    embeddings = []
    print(f"Collecting {ENROLL_SAMPLES} face samples. Press Q to abort.")

    while len(embeddings) < ENROLL_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            break

        emb, bbox = verifier.get_embedding(frame)
        draw_enroll_progress(frame, len(embeddings), ENROLL_SAMPLES)

        if emb is not None:
            embeddings.append(emb)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cv2.imshow("Enrollment", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) == ENROLL_SAMPLES:
        reference = np.mean(embeddings, axis=0)
        reference /= np.linalg.norm(reference)
        save_reference(reference)
        print("Enrollment complete. Reference saved.")
    else:
        print("Enrollment aborted.")


def run(verifier, threshold):
    reference = load_reference()
    if reference is None:
        print("No reference found. Run with --enroll first.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open camera.")
        return

    frame_count = 0
    last_match = False
    last_score = 0.0
    last_bbox = None

    print("Running verification. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % RECOGNITION_INTERVAL == 0:
            emb, bbox = verifier.get_embedding(frame)
            last_match, last_score = verifier.verify(emb, reference, threshold)
            last_bbox = bbox

        if last_bbox is not None:
            draw_face_box(frame, last_bbox, last_match, last_score)
        else:
            draw_status(frame, "No face detected")

        frame_count += 1
        cv2.imshow("Face Verification", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Real-time face verification")
    parser.add_argument("--enroll", action="store_true", help="Enroll a new reference face")
    parser.add_argument("--threshold", type=float, default=0.4, help="Similarity threshold (default 0.4)")
    args = parser.parse_args()

    verifier = FaceVerifier()

    if args.enroll:
        enroll(verifier)
    else:
        run(verifier, args.threshold)


if __name__ == "__main__":
    main()
