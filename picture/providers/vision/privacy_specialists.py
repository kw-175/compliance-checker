from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from picture.domain.enums import FindingType
from picture.domain.models import BBox, PictureFinding, Polygon, RegionMask

logger = logging.getLogger(__name__)

DEFAULT_VISUAL_PRIVACY_MODEL_DIR = Path("/data/kw/compliance-checker/models/visual_privacy")
QR_CODE_MODEL_DIR = DEFAULT_VISUAL_PRIVACY_MODEL_DIR / "wechat_qrcode"
SCRFD_MODEL_PATH = DEFAULT_VISUAL_PRIVACY_MODEL_DIR / "scrfd" / "det_10g.onnx"


class OpenCVQRCodeDetector:
    """OpenCV WeChatQRCode wrapper for deterministic QR localization."""

    def __init__(self, model_dir: str | Path = QR_CODE_MODEL_DIR) -> None:
        self._model_dir = Path(model_dir)
        self._detector: Any | None = None

    @property
    def name(self) -> str:
        return "OpenCVWeChatQRCode"

    def detect(self, image_path: str) -> list[PictureFinding]:
        cv2 = _import_cv2()
        image = cv2.imread(image_path)
        if image is None:
            return []
        detector = self._get_detector()
        try:
            decoded, points = detector.detectAndDecode(image)
        except Exception:
            logger.exception("OpenCV WeChatQRCode failed: image_path=%s", image_path)
            return []
        texts = list(decoded or [])
        point_sets = _normalize_point_sets(points)
        findings: list[PictureFinding] = []
        for index, polygon_points in enumerate(point_sets):
            decoded_text = str(texts[index] if index < len(texts) else "").strip()
            if not decoded_text:
                continue
            region = _region_from_polygon(polygon_points, confidence=0.99)
            if region is None:
                continue
            findings.append(
                PictureFinding(
                    finding_type=FindingType.VISION_OBJECT,
                    category="qr_code",
                    label="二维码",
                    score=0.99,
                    region=region,
                    reason_code="VISION_QR_CODE",
                    provider=self.name,
                    threshold_used=0.0,
                    explanation="检测到二维码，可能包含跳转链接或身份编码，需要脱敏。",
                    metadata={
                        "operator_id": "VPI_006",
                        "source_detectors": ["opencv_wechat_qrcode"],
                        "decoded_text_present": True,
                        "review_required": False,
                        "localization_status": "localized_by_specialist",
                        "redaction_hint": {"prefer_polygon": True, "padding_ratio": 0.05},
                    },
                )
            )
        return _dedupe_by_iou(findings)

    def _get_detector(self) -> Any:
        if self._detector is not None:
            return self._detector
        cv2 = _import_cv2()
        paths = {
            "det_prototxt": self._model_dir / "detect.prototxt",
            "det_model": self._model_dir / "detect.caffemodel",
            "sr_prototxt": self._model_dir / "sr.prototxt",
            "sr_model": self._model_dir / "sr.caffemodel",
        }
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise RuntimeError(f"WeChatQRCode model files missing: {missing}")
        self._detector = cv2.wechat_qrcode_WeChatQRCode(
            str(paths["det_prototxt"]),
            str(paths["det_model"]),
            str(paths["sr_prototxt"]),
            str(paths["sr_model"]),
        )
        return self._detector


class OpenCVBarcodeDetector:
    """OpenCV BarcodeDetector wrapper for deterministic barcode localization."""

    def __init__(self) -> None:
        self._detector: Any | None = None

    @property
    def name(self) -> str:
        return "OpenCVBarcodeDetector"

    def detect(self, image_path: str) -> list[PictureFinding]:
        cv2 = _import_cv2()
        image = cv2.imread(image_path)
        if image is None:
            return []
        detector = self._get_detector()
        try:
            result = detector.detectAndDecodeWithType(image)
        except AttributeError:
            result = detector.detectAndDecode(image)
        except Exception:
            logger.exception("OpenCV BarcodeDetector failed: image_path=%s", image_path)
            return []
        decoded_texts = _barcode_decoded_texts(result)
        point_sets = _barcode_point_sets(result)
        findings: list[PictureFinding] = []
        for index, polygon_points in enumerate(point_sets):
            decoded_text = decoded_texts[index] if index < len(decoded_texts) else ""
            if not decoded_text:
                continue
            region = _region_from_polygon(polygon_points, confidence=0.98)
            if region is None:
                continue
            findings.append(
                PictureFinding(
                    finding_type=FindingType.VISION_OBJECT,
                    category="barcode",
                    label="条形码",
                    score=0.98,
                    region=region,
                    reason_code="VISION_BARCODE",
                    provider=self.name,
                    threshold_used=0.0,
                    explanation="检测到条形码，可能包含编码信息，需要脱敏。",
                    metadata={
                        "operator_id": "VPI_007",
                        "source_detectors": ["opencv_barcode"],
                        "decoded_text_present": True,
                        "review_required": False,
                        "localization_status": "localized_by_specialist",
                        "redaction_hint": {
                            "prefer_polygon": True,
                            "padding_long_side_ratio": 0.08,
                            "padding_short_side_ratio": 0.04,
                        },
                    },
                )
            )
        return _dedupe_by_iou(findings)

    def _get_detector(self) -> Any:
        if self._detector is not None:
            return self._detector
        cv2 = _import_cv2()
        self._detector = cv2.barcode_BarcodeDetector()
        return self._detector


class SCRFDFaceDetector:
    """Lightweight SCRFD ONNX face detector for high-confidence face privacy regions."""

    def __init__(
        self,
        model_path: str | Path = SCRFD_MODEL_PATH,
        input_size: int = 640,
        confidence_threshold: float = 0.40,
        nms_threshold: float = 0.40,
        max_faces: int = 256,
    ) -> None:
        self._model_path = Path(model_path)
        self._input_size = int(input_size)
        self._confidence_threshold = float(confidence_threshold)
        self._nms_threshold = float(nms_threshold)
        self._max_faces = int(max_faces)
        self._session: Any | None = None

    @property
    def name(self) -> str:
        return "SCRFDFaceDetector"

    def detect(self, image_path: str) -> list[PictureFinding]:
        cv2 = _import_cv2()
        image = cv2.imread(image_path)
        if image is None:
            return []
        height, width = image.shape[:2]
        blob, scale, pad_left, pad_top = _scrfd_blob(image, self._input_size)
        session = self._get_session()
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: blob})
        boxes, scores, keypoints = _decode_scrfd_outputs(
            outputs,
            score_threshold=self._confidence_threshold,
            input_size=self._input_size,
        )
        keep = _nms(boxes, scores, self._nms_threshold)[: self._max_faces]
        findings: list[PictureFinding] = []
        for index in keep:
            box = boxes[index]
            mapped = _map_letterbox_box(box, scale, pad_left, pad_top, width, height)
            if mapped is None:
                continue
            kp = _map_letterbox_keypoints(keypoints[index], scale, pad_left, pad_top, width, height)
            score = float(scores[index])
            identifiability = _face_identifiability(mapped, kp, score, width, height)
            if not identifiability["is_identifiable_face"]:
                logger.info("Drop non-identifiable SCRFD face candidate: %s", identifiability)
                continue
            findings.append(
                PictureFinding(
                    finding_type=FindingType.VISION_OBJECT,
                    category="face",
                    label="人脸",
                    score=score,
                    region=RegionMask(bbox=mapped, confidence=score),
                    reason_code="VISION_FACE",
                    provider=self.name,
                    threshold_used=self._confidence_threshold,
                    explanation="检测到可识别人物面部，需要脱敏。",
                    metadata={
                        "operator_id": "VPI_001",
                        "source_detectors": ["scrfd"],
                        "face_keypoints": kp,
                        **identifiability,
                        "review_required": False,
                        "localization_status": "localized_by_specialist",
                    },
                )
            )
        return _dedupe_by_iou(findings)

    def _get_session(self) -> Any:
        if self._session is not None:
            return self._session
        if not self._model_path.exists():
            raise RuntimeError(f"SCRFD model file missing: {self._model_path}")
        import onnxruntime as ort

        self._session = ort.InferenceSession(str(self._model_path), providers=["CPUExecutionProvider"])
        return self._session


class MediaPipeFullRangeFaceDetector:
    """MediaPipe full-range face detector for regular and distant front-facing faces."""

    def __init__(
        self,
        min_detection_confidence: float = 0.35,
    ) -> None:
        self._min_detection_confidence = float(min_detection_confidence)
        self._detector: Any | None = None

    @property
    def name(self) -> str:
        return "MediaPipeFullRangeFaceDetector"

    def detect(self, image_path: str) -> list[PictureFinding]:
        cv2 = _import_cv2()
        image = cv2.imread(image_path)
        if image is None:
            return []
        height, width = image.shape[:2]
        detector = self._get_detector()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)
        detections = list(results.detections or [])
        findings: list[PictureFinding] = []
        for detection in detections:
            score = _mediapipe_score(detection)
            bbox = _mediapipe_bbox(detection, width, height)
            if bbox is None:
                continue
            keypoints = _mediapipe_keypoints(detection, width, height)
            identifiability = _face_identifiability(bbox, keypoints, score, width, height)
            if not identifiability["is_identifiable_face"]:
                logger.info("Drop non-identifiable MediaPipe face candidate: %s", identifiability)
                continue
            findings.append(
                PictureFinding(
                    finding_type=FindingType.VISION_OBJECT,
                    category="face",
                    label="人脸",
                    score=score,
                    region=RegionMask(bbox=bbox, confidence=score),
                    reason_code="VISION_FACE",
                    provider=self.name,
                    threshold_used=self._min_detection_confidence,
                    explanation="检测到可识别人物面部，需要脱敏。",
                    metadata={
                        "operator_id": "VPI_001",
                        "source_detectors": ["mediapipe_full_range"],
                        "face_keypoints": keypoints,
                        **identifiability,
                        "review_required": False,
                        "localization_status": "localized_by_specialist",
                    },
                )
            )
        return _dedupe_by_iou(findings)

    def _get_detector(self) -> Any:
        if self._detector is not None:
            return self._detector
        import mediapipe as mp

        if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "face_detection"):
            raise RuntimeError(
                "MediaPipe Full-Range local FaceDetection API is unavailable. "
                "Use mediapipe==0.10.14 for the local model_selection=1 runtime."
            )
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=self._min_detection_confidence,
        )
        return self._detector


class VisualPrivacySpecialistDetectors:
    """Runs deterministic specialist detectors only for requested privacy categories."""

    def __init__(
        self,
        qr_detector: OpenCVQRCodeDetector | None = None,
        barcode_detector: OpenCVBarcodeDetector | None = None,
        face_detector: SCRFDFaceDetector | None = None,
        mediapipe_face_detector: MediaPipeFullRangeFaceDetector | None = None,
    ) -> None:
        self._qr = qr_detector or OpenCVQRCodeDetector()
        self._barcode = barcode_detector or OpenCVBarcodeDetector()
        self._face = face_detector or SCRFDFaceDetector()
        self._mediapipe_face = mediapipe_face_detector or MediaPipeFullRangeFaceDetector()

    @property
    def name(self) -> str:
        return "VisualPrivacySpecialists"

    def detect(self, image_path: str, target_types: list[str]) -> list[PictureFinding]:
        targets = {str(item).strip().lower().replace("-", "_").replace(".", "_") for item in target_types}
        findings: list[PictureFinding] = []
        if "qr_code" in targets:
            findings.extend(_safe_detect(self._qr, image_path))
        if "barcode" in targets:
            findings.extend(_safe_detect(self._barcode, image_path))
        if "face" in targets:
            findings.extend(_safe_detect(self._mediapipe_face, image_path))
            findings.extend(_safe_detect(self._face, image_path))
        return _dedupe_by_iou(findings)


def _safe_detect(detector: Any, image_path: str) -> list[PictureFinding]:
    try:
        return detector.detect(image_path)
    except Exception:
        logger.exception("Visual privacy specialist failed: detector=%s image_path=%s", getattr(detector, "name", detector), image_path)
        return []


def _import_cv2() -> Any:
    import cv2

    return cv2


def _normalize_point_sets(points: Any) -> list[list[tuple[float, float]]]:
    if points is None:
        return []
    try:
        array = np.asarray(points, dtype=np.float32)
    except (TypeError, ValueError):
        return []
    if array.size == 0:
        return []
    if array.ndim == 2 and array.shape[0] >= 4 and array.shape[1] >= 2:
        array = array[None, :, :]
    if array.ndim == 3:
        return [
            [(float(point[0]), float(point[1])) for point in poly[:4]]
            for poly in array
            if len(poly) >= 4
        ]
    return []


def _barcode_point_sets(result: Any) -> list[list[tuple[float, float]]]:
    if not isinstance(result, tuple):
        return []
    for item in reversed(result):
        point_sets = _normalize_point_sets(item)
        if point_sets:
            return point_sets
    return []


def _barcode_decoded_texts(result: Any) -> list[str]:
    if not isinstance(result, tuple):
        return []
    for item in result:
        texts = _normalize_decoded_texts(item)
        if texts:
            return texts
    return []


def _normalize_decoded_texts(value: Any) -> list[str]:
    if value is None or isinstance(value, (bool, bytes)):
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        texts: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                texts.append(item.strip())
        return texts
    return []


def _region_from_polygon(points: list[tuple[float, float]], confidence: float) -> RegionMask | None:
    if len(points) < 4:
        return None
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return RegionMask(
        bbox=BBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1),
        polygon=Polygon(points=points),
        confidence=confidence,
    )


def _mediapipe_score(detection: Any) -> float:
    values = list(getattr(detection, "score", []) or [])
    if not values:
        return 0.0
    try:
        return float(values[0])
    except (TypeError, ValueError):
        return 0.0


def _mediapipe_bbox(detection: Any, image_width: int, image_height: int) -> BBox | None:
    location = getattr(detection, "location_data", None)
    relative = getattr(location, "relative_bounding_box", None)
    if relative is None:
        return None
    x = float(relative.xmin) * image_width
    y = float(relative.ymin) * image_height
    w = float(relative.width) * image_width
    h = float(relative.height) * image_height
    x1 = max(0.0, min(x, float(image_width)))
    y1 = max(0.0, min(y, float(image_height)))
    x2 = max(0.0, min(x + w, float(image_width)))
    y2 = max(0.0, min(y + h, float(image_height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return BBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)


def _mediapipe_keypoints(detection: Any, image_width: int, image_height: int) -> list[list[float]]:
    location = getattr(detection, "location_data", None)
    raw_points = list(getattr(location, "relative_keypoints", []) or [])
    points: list[list[float]] = []
    for point in raw_points:
        points.append(
            [
                max(0.0, min(float(point.x) * image_width, float(image_width))),
                max(0.0, min(float(point.y) * image_height, float(image_height))),
            ]
        )
    return points


def _face_identifiability(
    bbox: BBox,
    keypoints: list[list[float]],
    score: float,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    short_side = min(float(bbox.w), float(bbox.h))
    long_side = max(float(bbox.w), float(bbox.h))
    aspect_ratio = float(bbox.w) / max(1.0, float(bbox.h))
    area = max(0.0, float(bbox.w) * float(bbox.h))
    image_area = max(1.0, float(image_width) * float(image_height))
    area_ratio = area / image_area
    visible_points = _valid_face_keypoints(keypoints, bbox)
    visible_features = _visible_face_features(len(visible_points))
    reasons: list[str] = []
    if short_side < 24.0:
        reasons.append("人脸区域过小，无法稳定识别具体个人")
    if area_ratio < 0.0003:
        reasons.append("人脸面积占比过低，隐私可识别性不足")
    if aspect_ratio < 0.55 or aspect_ratio > 1.8:
        reasons.append("人脸框长宽比异常，疑似手部、身体局部或背景误检")
    if short_side / max(1.0, long_side) < 0.42:
        reasons.append("人脸框过窄或过扁，形状不符合可识别人脸")
    if _face_bbox_too_clipped(bbox, image_width, image_height):
        reasons.append("人脸框严重贴边或残缺，无法稳定确认身份特征")
    if len(visible_points) < 3:
        reasons.append("可见五官关键点不足，可能是后脑、遮挡脸或极少侧脸")
    if float(score) < 0.45:
        reasons.append("人脸检测置信度不足")
    identifiable = not reasons
    confidence = min(0.99, max(0.0, float(score)) * (1.0 if identifiable else 0.45))
    return {
        "is_identifiable_face": identifiable,
        "identifiability_score": round(confidence, 4),
        "visible_face_features": visible_features,
        "face_filter_decision": "keep" if identifiable else "drop",
        "face_filter_reason": "五官关键点和尺寸满足可识别真实人脸要求" if identifiable else "；".join(reasons),
        "face_bbox_short_side": round(short_side, 2),
        "face_bbox_aspect_ratio": round(aspect_ratio, 4),
        "face_area_ratio": round(area_ratio, 6),
        "face_visible_keypoint_count": len(visible_points),
    }


def _face_bbox_too_clipped(bbox: BBox, image_width: int, image_height: int) -> bool:
    width = max(1.0, float(image_width))
    height = max(1.0, float(image_height))
    margin_x = max(2.0, float(bbox.w) * 0.08)
    margin_y = max(2.0, float(bbox.h) * 0.08)
    touches_x = float(bbox.x) <= margin_x or float(bbox.x + bbox.w) >= width - margin_x
    touches_y = float(bbox.y) <= margin_y or float(bbox.y + bbox.h) >= height - margin_y
    return bool((touches_x and float(bbox.w) < 48.0) or (touches_y and float(bbox.h) < 48.0))


def _valid_face_keypoints(keypoints: list[list[float]], bbox: BBox) -> list[list[float]]:
    valid: list[list[float]] = []
    pad_x = max(4.0, float(bbox.w) * 0.20)
    pad_y = max(4.0, float(bbox.h) * 0.20)
    left = float(bbox.x) - pad_x
    right = float(bbox.x) + float(bbox.w) + pad_x
    top = float(bbox.y) - pad_y
    bottom = float(bbox.y) + float(bbox.h) + pad_y
    for point in keypoints or []:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            continue
        if left <= x <= right and top <= y <= bottom:
            valid.append([x, y])
    return valid


def _visible_face_features(count: int) -> list[str]:
    labels = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right", "face_contour"]
    return labels[: max(0, min(int(count), len(labels)))]


def _scrfd_blob(image: np.ndarray, input_size: int) -> tuple[np.ndarray, float, int, int]:
    cv2 = _import_cv2()
    height, width = image.shape[:2]
    scale = min(float(input_size) / max(width, 1), float(input_size) / max(height, 1))
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    pad_left = 0
    pad_top = 0
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    blob = cv2.dnn.blobFromImage(
        canvas,
        scalefactor=1.0 / 128.0,
        size=(input_size, input_size),
        mean=(127.5, 127.5, 127.5),
        swapRB=True,
    )
    return blob.astype(np.float32), scale, pad_left, pad_top


def _decode_scrfd_outputs(
    outputs: list[Any],
    score_threshold: float,
    input_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    strides = (8, 16, 32)
    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_keypoints: list[np.ndarray] = []
    for level, stride in enumerate(strides):
        scores = np.asarray(outputs[level]).reshape(-1)
        bbox_preds = np.asarray(outputs[level + 3]).reshape(-1, 4) * stride
        kps_preds = np.asarray(outputs[level + 6]).reshape(-1, 10) * stride
        anchor_centers = _scrfd_anchor_centers(input_size, stride)
        count = min(len(scores), len(anchor_centers), len(bbox_preds), len(kps_preds))
        if count <= 0:
            continue
        scores = scores[:count]
        keep = np.where(scores >= score_threshold)[0]
        if keep.size == 0:
            continue
        anchors = anchor_centers[:count][keep]
        all_boxes.append(_distance_to_bbox(anchors, bbox_preds[:count][keep]))
        all_scores.append(scores[keep])
        all_keypoints.append(_distance_to_keypoints(anchors, kps_preds[:count][keep]))
    if not all_boxes:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0, 5, 2), dtype=np.float32),
        )
    return np.vstack(all_boxes), np.concatenate(all_scores), np.vstack(all_keypoints)


def _scrfd_anchor_centers(input_size: int, stride: int, num_anchors: int = 2) -> np.ndarray:
    height = input_size // stride
    width = input_size // stride
    y, x = np.mgrid[:height, :width]
    centers = np.stack((x, y), axis=-1).astype(np.float32)
    centers = (centers * stride).reshape(-1, 2)
    return np.repeat(centers, num_anchors, axis=0)


def _distance_to_bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance_to_keypoints(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    points = points[:, None, :]
    values = distance.reshape(-1, 5, 2)
    decoded = np.empty_like(values)
    decoded[:, :, 0] = points[:, :, 0] + values[:, :, 0]
    decoded[:, :, 1] = points[:, :, 1] + values[:, :, 1]
    return decoded


def _map_letterbox_box(
    box: np.ndarray,
    scale: float,
    pad_left: int,
    pad_top: int,
    image_width: int,
    image_height: int,
) -> BBox | None:
    x1 = (float(box[0]) - pad_left) / scale
    y1 = (float(box[1]) - pad_top) / scale
    x2 = (float(box[2]) - pad_left) / scale
    y2 = (float(box[3]) - pad_top) / scale
    x1 = max(0.0, min(x1, float(image_width)))
    y1 = max(0.0, min(y1, float(image_height)))
    x2 = max(0.0, min(x2, float(image_width)))
    y2 = max(0.0, min(y2, float(image_height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return BBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)


def _map_letterbox_keypoints(
    keypoints: np.ndarray,
    scale: float,
    pad_left: int,
    pad_top: int,
    image_width: int,
    image_height: int,
) -> list[list[float]]:
    mapped: list[list[float]] = []
    for point in keypoints:
        x = max(0.0, min((float(point[0]) - pad_left) / scale, float(image_width)))
        y = max(0.0, min((float(point[1]) - pad_top) / scale, float(image_height)))
        mapped.append([x, y])
    return mapped


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[index], x1[order[1:]])
        yy1 = np.maximum(y1[index], y1[order[1:]])
        xx2 = np.minimum(x2[index], x2[order[1:]])
        yy2 = np.minimum(y2[index], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[index] + areas[order[1:]] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = order[np.where(iou <= threshold)[0] + 1]
    return keep


def _dedupe_by_iou(findings: list[PictureFinding], threshold: float = 0.82) -> list[PictureFinding]:
    kept: list[PictureFinding] = []
    for finding in sorted(findings, key=lambda item: item.score, reverse=True):
        if finding.region is None:
            kept.append(finding)
            continue
        duplicate = False
        for existing in kept:
            if existing.region is None or existing.category != finding.category:
                continue
            if _bbox_iou(existing.region.bbox, finding.region.bbox) >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(finding)
    return kept


def _bbox_iou(left: BBox, right: BBox) -> float:
    left_x2 = left.x + left.w
    left_y2 = left.y + left.h
    right_x2 = right.x + right.w
    right_y2 = right.y + right.h
    inter_w = max(0.0, min(left_x2, right_x2) - max(left.x, right.x))
    inter_h = max(0.0, min(left_y2, right_y2) - max(left.y, right.y))
    intersection = inter_w * inter_h
    union = left.w * left.h + right.w * right.h - intersection
    if union <= 0:
        return 0.0
    return intersection / union
