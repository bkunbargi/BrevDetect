import numpy as np
import torch
from PIL import Image

from insightface.app import FaceAnalysis


def _to_rgb_uint8(image):
    if isinstance(image, torch.Tensor):
        img = image
        if img.dim() == 4:
            img = img[0]
        print(f"[BrevDetect] Input tensor shape: {tuple(img.shape)}, dtype={img.dtype}")
        img = img.detach().cpu().numpy()
    else:
        img = np.array(image)
        print(f"[BrevDetect] Input numpy shape: {img.shape}, dtype={img.dtype}")

    if img.ndim != 3:
        raise ValueError(f"Expected HWC image, got {img.shape}")

    if img.shape[-1] != 3 and img.shape[0] == 3:
        print("[BrevDetect] Transposing CHW → HWC")
        img = np.transpose(img, (1, 2, 0))

    if img.shape[-1] != 3:
        raise ValueError(f"Expected 3 channels, got {img.shape}")

    if img.dtype != np.uint8:
        print("[BrevDetect] Converting float image → uint8")
        img = (img * 255.0).clip(0, 255).astype(np.uint8)

    print(f"[BrevDetect] RGB uint8 image: {img.shape}")
    return img


def _to_comfy_image(rgb_uint8):
    out = rgb_uint8.astype(np.float32) / 255.0
    t = torch.from_numpy(out).unsqueeze(0)
    print(f"[BrevDetect] Output tensor shape: {tuple(t.shape)}, dtype={t.dtype}")
    return t


class FaceDetectResizeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "BrevDetect"

    _app = None
    _app_cfg = None

    @classmethod
    def _get_app(cls):
        det_size = (640, 640)
        ctx_id = 0 if torch.cuda.is_available() else -1

        if cls._app is None:
            print(f"[BrevDetect] Initializing InsightFace (ctx_id={ctx_id})")
            app = FaceAnalysis(name="buffalo_l")
            app.prepare(ctx_id=ctx_id, det_size=det_size)
            cls._app = app
            cls._app_cfg = (ctx_id, det_size)
        else:
            print("[BrevDetect] Reusing cached InsightFace instance")

        return cls._app

    def detect_largest_face_box(self, image_rgb_uint8):
        print("[BrevDetect] Running face detection…")
        app = self._get_app()

        image_bgr = image_rgb_uint8[:, :, ::-1].copy()
        faces = app.get(image_bgr)

        print(f"[BrevDetect] Faces detected: {len(faces)}")
        if not faces:
            return None

        boxes = np.array([f.bbox for f in faces], dtype=np.float32)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        i = int(np.argmax(areas))
        x1, y1, x2, y2 = boxes[i]

        x = int(max(0, np.floor(x1)))
        y = int(max(0, np.floor(y1)))
        w = int(max(0, np.ceil(x2 - x1)))
        h = int(max(0, np.ceil(y2 - y1)))

        print(f"[BrevDetect] Largest face box: x={x}, y={y}, w={w}, h={h}, area={w*h}")
        return {"box": [x, y, w, h]}

    def process_image(self, image):
        print("[BrevDetect] ==== FaceDetectResizeNode ====")
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

                print(f"[BrevDetect] Scale factor: {scale_factor:.4f}")

                new_w = max(1, int(round(pil_image.width * scale_factor)))
                new_h = max(1, int(round(pil_image.height * scale_factor)))

                if (new_w, new_h) != pil_image.size:
                    print(f"[BrevDetect] Resizing image → {new_w}×{new_h}")
                    pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
                else:
                    print("[BrevDetect] No resize needed")
            else:
                print("[BrevDetect] No face found — skipping resize")

            out_rgb = np.array(pil_image, dtype=np.uint8)
            return (_to_comfy_image(out_rgb),)

        except Exception as e:
            print(f"[BrevDetect] ERROR: {e}")
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
        print("[BrevResize] ==== Resize ====")
        try:
            image_rgb = _to_rgb_uint8(image)
            pil = Image.fromarray(image_rgb)
            pil = pil.resize((int(width), int(height)), Image.LANCZOS)
            out_rgb = np.array(pil, dtype=np.uint8)
            return (_to_comfy_image(out_rgb),)
        except Exception as e:
            print(f"[BrevResize] ERROR: {e}")
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
