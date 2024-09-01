import numpy as np
import torch
import cv2
from PIL import Image
from mtcnn import MTCNN

class FaceDetectResizeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "BrevDetect"

    def detect_faces(self, image_rgb):
        """Detect faces in the RGB image using MTCNN and keep only the largest face."""
        print(f"Input image shape: {image_rgb.shape}, dtype: {image_rgb.dtype}")
        detector = MTCNN()
        results = detector.detect_faces(image_rgb)
        if not results:
            print("No faces detected.")
            return None
        largest_face = max(results, key=lambda x: x['box'][2] * x['box'][3])
        print(f"Largest face detected at: {largest_face['box']}")
        return [largest_face]

    def process_image(self, image):
        print("Input image shape: ", image.shape)
        try:
            if isinstance(image, torch.Tensor):
                image_np = image.squeeze(0).permute(0,1,2).cpu().numpy()
            else:
                image_np = np.array(image)

            print("Image Shape: ", image_np.shape)
            if image_np.shape[2] != 3:
                image_np = np.transpose(image_np, (1,2,0))

            image_rgb = (image_np * 255).clip(0, 255).astype(np.uint8)
            print(f"Converted image shape: {image_rgb.shape}, dtype: {image_rgb.dtype}")

            faces = self.detect_faces(image_rgb)
            min_face_size = 128
            max_face_size = 640
            pil_image = Image.fromarray(image_rgb)
            print(f"Original PIL image size: {pil_image.size}")

            if faces:
                largest_face_dimensions = max((face['box'][2], face['box'][3]) for face in faces)
                print(f"Largest face dimensions: {largest_face_dimensions}")
                if max(largest_face_dimensions) > max_face_size:
                    scale_factor = max_face_size / max(largest_face_dimensions)
                elif max(largest_face_dimensions) < min_face_size:
                    scale_factor = min_face_size / max(largest_face_dimensions)
                else:
                    scale_factor = 1
                new_width = int(pil_image.width * scale_factor)
                new_height = int(pil_image.height * scale_factor)
                scaled_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
                print(f"Image resized to: {scaled_image.size}")
            else:
                scaled_image = pil_image
                print("No resizing needed. Image remains at original size.")

            # Convert back to ComfyUI image format
            output_image = np.array(scaled_image).astype(np.float32) / 255.0
            output_image = torch.from_numpy(output_image).unsqueeze(0)
            print(f"Output image shape: {output_image.shape}, type: {output_image.dtype}")

            return (output_image,)
        except Exception as e:
            print(f"Error in process_image: {str(e)}")
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
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize_image"
    CATEGORY = "BrevResize"

    def resize_image(self, image, width, height):
        print("Input image shape: ", image.shape)
        try:
            if isinstance(image, torch.Tensor):
                image_np = image.squeeze(0).permute(0,1,2).cpu().numpy()
            else:
                image_np = np.array(image)

            print("Image Shape: ", image_np.shape)
            if image_np.shape[2] != 3:
                image_np = np.transpose(image_np, (1,2,0))

            image_rgb = (image_np * 255).clip(0, 255).astype(np.uint8)
            print(f"Converted image shape: {image_rgb.shape}, dtype: {image_rgb.dtype}")

            pil_image = Image.fromarray(image_rgb)
            print(f"Original PIL image size: {pil_image.size}")

            resized_image = pil_image.resize((width, height), Image.LANCZOS)
            print(f"Image resized to: {resized_image.size}")

            # Convert back to ComfyUI image format
            output_image = np.array(resized_image).astype(np.float32) / 255.0
            output_image = torch.from_numpy(output_image).permute(2, 0, 1).unsqueeze(0)
            print(f"Output image shape: {output_image.shape}, type: {output_image.dtype}")

            return (output_image,)
        except Exception as e:
            print(f"Error in resize_image: {str(e)}")
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