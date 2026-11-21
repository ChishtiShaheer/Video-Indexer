"""YOLO11l detector module for video object detection and counting."""
from .common import (COCO_CLASS_IDS, CONF_THRESHOLD, IMGSZ, IOU_THRESHOLD,
                     TARGET_CLASSES)


class Detector:
    def __init__(
        self,
        weights="yolo11l.pt",
        device=None,
        half=True,
        imgsz=IMGSZ,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        classes=None,
    ):
        import numpy as np
        import torch
        from ultralytics import YOLO

        self.np = np
        self.torch = torch
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.classes = list(classes) if classes is not None else COCO_CLASS_IDS

        # Handle device resolution
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = str(device)
            if self.device == "cuda" and torch.cuda.is_available():
                self.device = "cuda:0"

        # Explicitly set current active CUDA device for PyTorch runtime
        if torch.cuda.is_available() and self.device.startswith("cuda:"):
            try:
                torch.cuda.set_device(self.device)
            except Exception:
                pass

        self.is_engine = str(weights).endswith(".engine")
        self.model = YOLO(weights)

        # TensorRT engines are built in FP16; PyTorch .pt uses half=True on CUDA
        self.half = half and (self.device != "cpu")

        # Warm up model to eliminate first-batch compilation latency
        dummy = self.np.zeros((self.imgsz, self.imgsz, 3), dtype=self.np.uint8)
        self.model.predict(
            dummy,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.classes,
            device=self.device,
            half=self.half,
            verbose=False,
        )

    def predict_batch(self, batch_images):
        """Run batched inference on BGR images.

        batch_images: np.ndarray of shape (B, H, W, 3) or list of images.
        Returns:
            list of 5-tuples: [(person, car, dog, laptop, phone), ...]
        """
        if batch_images is None or len(batch_images) == 0:
            return []

        # Ultralytics predict() on this version requires a list of (H, W, 3)
        # images, not a stacked 4-D numpy array (B, H, W, 3).
        if hasattr(batch_images, "ndim") and batch_images.ndim == 4:
            batch_images = [batch_images[i] for i in range(len(batch_images))]

        results = self.model.predict(
            batch_images,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.classes,
            device=self.device,
            half=self.half,
            verbose=False,
        )

        batch_counts = []
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                cls_ids = r.boxes.cls.int().cpu().numpy()
                p = int(self.np.count_nonzero(cls_ids == 0))
                c = int(self.np.count_nonzero(cls_ids == 2))
                d = int(self.np.count_nonzero(cls_ids == 16))
                l = int(self.np.count_nonzero(cls_ids == 63))
                ph = int(self.np.count_nonzero(cls_ids == 67))
                batch_counts.append((p, c, d, l, ph))
            else:
                batch_counts.append((0, 0, 0, 0, 0))

        return batch_counts
