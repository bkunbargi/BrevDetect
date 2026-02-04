import numpy as np
import torch
from PIL import Image

from insightface.app import FaceAnalysis


def _to_rgb_uint8(image):
    """
    Convert ComfyUI IMAGE -> uint8 RGB HxWx3
    ComfyUI usually: torch float32 [B,H,W,C] in 0..1
    """
    if isinstance(image, torch.Tensor):
        img = image
        if img.dim() == 4:
            img = img[0]  # first in batch
        img = img.detach().cpu().numpy()
    else:
        img = np.array(image)

    # Ensure HWC
    if img.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {img.shape}")

    # Some codebases accidentally produce CHW; handle that safely
    if img.shape[-1] != 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))

    if img.shape[-1] != 3:
        raise ValueError(f"Expected 3 channels, got shape {img.shape}")

    # If float image in 0..1, convert to 0..255 uint8
    if img.dtype != np.uint8:
        img = (img * 255.0).clip(0, 255).astype(np.uint8)

    return img


def _to_comfy_image(rgb_uint8):
    """
    uint8 RGB HxWx3 -> ComfyUI IMAGE torch float [1,H,W,C] in 0..1
    """
    out = rgb_uint8.astype(np.float32) / 255.0
    return torch.from_numpy(out).unsqueeze(0)


class FaceDetectResizeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "BrevDetect"

    # Cache InsightFace app across calls
    _app = None
    _app_cfg = None

    @classmethod
    def _get_app(cls):
        """
        Create FaceAnalysis once and reuse. This avoids repeated model init.
        """
        # If you want to tune detection size, do it here.
        det_size = (640, 640)

        # Pick GPU if available, else CPU
        ctx_id = 0 if torch.cuda.is_available() else -1

        cfg = (ctx_id, det_size)
        if cls._app is None or cls._app_cfg != cfg:
            app = FaceAnalysis(name="buffalo_l")
            app.prepare(ctx_id=ctx_id, det_size=det_size)
            cls._app = app
            cls._app_cfg = cfg

        return cls._app

    def detect_largest_face_box(self, image_rgb_uint8):
        """
        image_rgb_uint8: HxWx3 RGB uint8
        Return: {"box": [x, y, w, h]} or None
        """
        app = self._get_app()

        # InsightFace expects BGR
        image_bgr = image_rgb_uint8[:, :, ::-1].copy()

        faces = app.get(image_bgr)
        if not faces:
            return None

        # faces[i].bbox = [x1,y1,x2,y2]
        boxes = np.array([f.bbox for f in faces], dtype=np.float32)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        i = int(np.argmax(areas))

        x1, y1, x2, y2 = boxes[i]
        x = int(max(0, np.floor(x1)))
        y = int(max(0, np.floor(y1)))
        w = int(max(0, np.ceil(x2 - x1)))
        h = int(max(0, np.ceil(y2 - y1)))

        return {"box": [x, y, w, h]}

    def process_image(self, image):
        try:
            image_rgb = _to_rgb_uint8(image)

            face = self.detect_largest_face_box(image_rgb)

            min_face_size = 128
            max_face_size = 640

            pil_image = Image.fromarray(image_rgb)

            if face is not None:
                _, _, w, h = face["box"]
                largest_dim = max(w, h)

                if largest_dim <= 0:
                    scale_factor = 1.0
                elif largest_dim > max_face_size:
                    scale_factor = max_face_size / largest_dim
                elif largest_dim < min_face_size:
                    scale_factor = min_face_size / largest_dim
                else:
                    scale_factor = 1.0

                new_w = max(1, int(round(pil_image.width * scale_factor)))
                new_h = max(1, int(round(pil_image.height * scale_factor)))

                if (new_w, new_h) != pil_image.size:
                    pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)

            out_rgb = np.array(pil_image, dtype=np.uint8)
            return (_to_comfy_image(out_rgb),)

        except Exception as e:
            print(f"[BrevDetect] Error: {e}")
            import traceback
            traceback.print_exc()
            return (image,)


class BrevResize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize_image"
    CATEGORY = "BrevResize"

    def resize_image(self, image, width, height):
        try:
            image_rgb = _to_rgb_uint8(image)
            pil = Image.fromarray(image_rgb)
            pil = pil.resize((int(width), int(height)), Image.LANCZOS)
            out_rgb = np.array(pil, dtype=np.uint8)
            return (_to_comfy_image(out_rgb),)
        except Exception as e:
            print(f"[BrevResize] Error: {e}")
            import traceback
            traceback.print_exc()
            return (image,)


NODE_CLASS_MAPPINGS = {
    "FaceDetectResizeNode": FaceDetectResizeNode,
    "BrevResize": BrevResize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDetectResizeNode": "Face Detect and Resize",
    "BrevResize": "Brev Resize",
}
