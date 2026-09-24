import os

ARTIFACT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.joblib")
SEGMENTER_PATH = os.path.join(ARTIFACT_DIR, "segmenter.joblib")
SEGMENT_MAP_PATH = os.path.join(ARTIFACT_DIR, "segment_map.joblib")
THRESHOLDS_PATH = os.path.join(ARTIFACT_DIR, "thresholds.joblib")
