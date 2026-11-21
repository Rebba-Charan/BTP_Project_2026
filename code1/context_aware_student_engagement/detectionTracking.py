from pathlib import Path

import cv2
import numpy as np
import torch
from engageConfig import *
import time


class _Track:
    def __init__(self, track_id, box, class_id, confidence):
        self.id = track_id
        self.history_observations = [box[:4].copy()]
        self.class_id = int(class_id)
        self.conf = float(confidence)


class SimpleByteTrack:
    """Small IoU tracker with the subset of BoxMOT's API used by this demo."""

    def __init__(self, iou_threshold=0.3, max_age=30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.active_tracks = []
        self.per_class_active_tracks = None
        self._next_id = 1
        self._ages = {}

    @staticmethod
    def _iou(left, right):
        x1 = max(left[0], right[0])
        y1 = max(left[1], right[1])
        x2 = min(left[2], right[2])
        y2 = min(left[3], right[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_left = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
        area_right = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
        union = area_left + area_right - intersection
        return intersection / union if union else 0.0

    def update(self, detections, frame):
        detections = np.asarray(detections, dtype=np.float32).reshape(-1, 6)
        matched = set()
        for track in self.active_tracks:
            best_index, best_iou = None, 0.0
            for index, detection in enumerate(detections):
                if index in matched:
                    continue
                score = self._iou(track.history_observations[-1], detection[:4])
                if score > best_iou:
                    best_index, best_iou = index, score
            if best_index is not None and best_iou >= self.iou_threshold:
                detection = detections[best_index]
                track.history_observations.append(detection[:4].copy())
                track.class_id = int(detection[5])
                track.conf = float(detection[4])
                self._ages[track.id] = 0
                matched.add(best_index)
            else:
                self._ages[track.id] += 1
        for index, detection in enumerate(detections):
            if index not in matched:
                track = _Track(self._next_id, detection, detection[5], detection[4])
                self._next_id += 1
                self.active_tracks.append(track)
                self._ages[track.id] = 0
        self.active_tracks = [
            track for track in self.active_tracks
            if self._ages[track.id] <= self.max_age
        ]
        return np.asarray(
            [[*track.history_observations[-1], track.id, track.conf, track.class_id]
             for track in self.active_tracks],
            dtype=np.float32,
        )

    def _infer_state(self, track):
        return "confirmed"

    def id_to_color(self, track_id, state="confirmed"):
        return ((37 * int(track_id)) % 255,
                (67 * int(track_id)) % 255,
                (97 * int(track_id)) % 255)

class Tracking:
    def __init__(self,objectDetector='fastercnn',tracker='ByteTrack'):
        # device = torch.device('cpu')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.objectDetector = objectDetector
        if objectDetector=='yolo':
            self.initializeYoloDetetcor()
        else:
            self.initializeDetector()
        self.initializeTracker()
    def initializeDetector(self):
        try:
            from torchvision.models.detection import (
                fasterrcnn_resnet50_fpn_v2,
                FasterRCNN_ResNet50_FPN_V2_Weights as Weights)
        except ImportError as exc:
            raise ImportError(
                "The Faster R-CNN detector requires torchvision. "
                "Install the torch/torchvision pair from requirements.txt."
            ) from exc
        # Load detector with pretrained weights and preprocessing transforms
        weights = Weights.DEFAULT
        self.detector = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
        self.detector.to(self.device).eval()
        self.transform = weights.transforms()
    def initializeYoloDetetcor(self):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "The optional YOLO detector requires ultralytics. "
                "Install it with `pip install ultralytics`."
            ) from exc
        model_path = yolox_model or "yolo11n.pt"
        self.detector = YOLO(model_path)
    def initializeTracker(self):
        self.tracker = SimpleByteTrack()

    # This preprocess differs from the current version of YOLOX preprocess, but ByteTrack uses it
    # https://github.com/ifzhang/ByteTrack/blob/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/yolox/data/data_augment.py#L189
    def yolox_preprocess(self,image,
                         input_size,
                         mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225),
                         swap=(2, 0, 1)):
        if len(image.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
        else:
            padded_img = np.ones(input_size) * 114.0

        img = np.array(image)
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR).astype(np.float32)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        padded_img = padded_img[:, :, ::-1]
        padded_img /= 255.0
        if mean is not None:
            padded_img -= mean

        if std is not None:
            padded_img /= std

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def detect(self,frame):
        if self.objectDetector=='yolo':
            result = self.detector.predict(frame, verbose=False, conf=0.5)[0]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                dets = np.empty((0, 6), dtype=np.float32)
            else:
                dets = np.column_stack((
                    boxes.xyxy.cpu().numpy(),
                    boxes.conf.cpu().numpy(),
                    boxes.cls.cpu().numpy(),
                )).astype(np.float32)
        else:
            # Convert frame to RGB and prepare for detector
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(torch.uint8)
            input_tensor = self.transform(tensor).to(self.device)

            # Run detection
            output = self.detector([input_tensor])[0]
            scores = output['scores'].detach().cpu().numpy()
            labels = output['labels'].detach().cpu().numpy()
            keep = labels == 1  # scores >= 0.5

            # Prepare detections for tracking
            boxes = output['boxes'][keep].detach().cpu().numpy()
            #===expand y axis===
            boxes[:,3] = boxes[:,3]*1.1
            labels = output['labels'][keep].detach().cpu().numpy()
            filtered_scores = scores[keep]
            dets = np.concatenate([boxes, filtered_scores[:, None], labels[:, None]], axis=1)

        #===sort it from left to right
        #dets = np.asarray(sorted(dets, key=lambda x: x[0]))
        sorted_indices = dets[:, 0].argsort() if len(dets) else []
        sorted_dets = dets[sorted_indices]
        return sorted_dets

    def track(self,dets,frame,show_lost=False):
        # Update tracker
        res = self.tracker.update(dets, frame)

        # =======trackers_data===#
        if self.tracker.per_class_active_tracks is None:  # dict
            active_tracks = self.tracker.active_tracks
        else:
            active_tracks = []
            for k in self.tracker.per_class_active_tracks.keys():
                active_tracks += self.tracker.per_class_active_tracks[k]

        trackers_data =[]
        for a in active_tracks:
            if not a.history_observations:
                continue
            if len(a.history_observations) < 3:
                continue
            box = a.history_observations[-1]

            state = self.tracker._infer_state(a)

            if state != 'confirmed':
                print(f"state {state}")
            t_id = a.id
            trackers_data.append((a.id,box,state))

        return trackers_data

    def plot_boxes_on_img(
            self,
            img: np.ndarray,
            trackersInfo,
            thickness: int = 2,
            fontscale: float = 0.5,
            state: str = "confirmed",
            style: str = "solid",  # "solid" | "dashed" (dashed only for AABB)
            ) -> np.ndarray:
        """
        Draws a bounding box with ID, confidence, and class information on an image.
        """

        for id,box in trackersInfo:
            color = self.tracker.id_to_color(id, state=state)
            x1, y1, x2, y2 = map(int, (box[0], box[1], box[2], box[3]))
            if style == "dashed":
                img = self._draw_dashed_rect(img, x1, y1, x2, y2, color, thickness)
            else:
                img = cv2.rectangle(img,(x1, y1),(x2, y2),color,thickness,)
            img = cv2.putText(
                img,f"id: {int(id)}",
                (x1, max(0, y1 - 10)),cv2.FONT_HERSHEY_SIMPLEX,fontscale,color,thickness,)
        return img
