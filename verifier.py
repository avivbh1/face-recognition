import numpy as np
from insightface.app import FaceAnalysis


class FaceVerifier:
    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def get_embedding(self, frame):
        """Return (embedding, bbox) for the largest detected face, or (None, None)."""
        faces = self.app.get(frame)
        if not faces:
            return None, None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return face.normed_embedding, face.bbox.astype(int)

    def similarity(self, emb1, emb2):
        return float(np.dot(emb1, emb2))

    def verify_against_all(self, embedding, faces, threshold=0.4):
        """Return (name, score) for the best match above threshold, else (None, best_score)."""
        if embedding is None or not faces:
            return None, 0.0
        best_name, best_score = None, 0.0
        for face in faces:
            score = self.similarity(embedding, face["embedding"])
            if score > best_score:
                best_score = score
                best_name = face["name"] if score >= threshold else None
        return best_name, round(best_score, 3)
