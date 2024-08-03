import numpy as np
import torch
import cv2
from mtcnn import MTCNN

class FaceDetectResize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "min_face_size": ("INT", {"default": 128, "min": 64, "max": 1024, "step": 1}),
                "max_face_size": ("INT", {"default": 640, "min": 128, "max": 2048, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "BrevDetect"

    def process_image(self, image, min_face_size, max_face_size):
        # Convert from ComfyUI image format to numpy array
        image_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)

        # Convert to BGR (cv2 uses BGR by default)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Detect faces
        faces = self.detect_faces(image_bgr)

        # Process and resize the image
        if faces:
            largest_face_dimensions = max((result['box'][2], result['box'][3]) for result in faces)
            if max(largest_face_dimensions) > max_face_size:
                scale_factor = max_face_size / max(largest_face_dimensions)
            elif max(largest_face_dimensions) < min_face_size:
                scale_factor = min_face_size / max(largest_face_dimensions)
            else:
                scale_factor = 1
            new_width = int(image_bgr.shape[1] * scale_factor)
            new_height = int(image_bgr.shape[0] * scale_factor)
            scaled_image = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        else:
            scaled_image = image_bgr

        # Convert back to RGB for ComfyUI
        scaled_image_rgb = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)

        # Convert back to ComfyUI image format
        scaled_image_np = scaled_image_rgb.astype(np.float32) / 255.0
        scaled_image_tensor = torch.from_numpy(scaled_image_np).unsqueeze(0).permute(0, 3, 1, 2)

        return (scaled_image_tensor,)

    def detect_faces(self, image_bgr):
        detector = MTCNN()
        # MTCNN expects RGB input
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(image_rgb)
        if not results:
            return results
        largest_face = max(results, key=lambda x: x['box'][2] * x['box'][3])
        return [largest_face]

NODE_CLASS_MAPPINGS = {
    "FaceDetectResize": FaceDetectResize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDetectResize": "Face Detect and Resize"
}