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

class BrevResizeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "target_height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "maintain_aspect_ratio": (["True", "False"], {"default": "True"})
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize_image"
    CATEGORY = "BrevResize"

    def resize_image(self, image, target_width, target_height, maintain_aspect_ratio=True):
        try:
            print(f"Input image type: {type(image)}")
            print(f"Input image shape: {image.shape if hasattr(image, 'shape') else 'Not available'}")
            print(f"Target dimensions: {target_width}x{target_height}")
            print(f"Maintain aspect ratio: {maintain_aspect_ratio}")

            # Ensure the input tensor is in the correct format (B, C, H, W)
            if isinstance(image, torch.Tensor):
                if image.shape[1] == 3:  # If it's already in (B, C, H, W) format
                    image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                else:  # If it's in (B, H, W, C) format
                    image_np = image.squeeze(0).cpu().numpy()
                print(f"Converted to NumPy array. Shape: {image_np.shape}")
            else:
                image_np = np.array(image)
                print(f"Already NumPy array. Shape: {image_np.shape}")

            # Ensure the image is in the correct format for PIL (H, W, C)
            if image_np.shape[2] != 3:
                image_np = np.transpose(image_np, (1, 2, 0))
            
            # Convert image to RGB format if not already
            image_rgb = (image_np * 255).clip(0, 255).astype(np.uint8)
            print(f"Converted to RGB. Shape: {image_rgb.shape}")

            # Convert to PIL image
            pil_image = Image.fromarray(image_rgb)
            print(f"Converted to PIL Image. Size: {pil_image.size}")

            if maintain_aspect_ratio == "True":
                # Maintain aspect ratio
                pil_image.thumbnail((target_width, target_height), Image.LANCZOS)
                new_width, new_height = pil_image.size
                print(f"Resized with aspect ratio. New size: {new_width}x{new_height}")
            else:
                # Resize without maintaining aspect ratio
                new_width, new_height = target_width, target_height
                pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
                print(f"Resized without aspect ratio. New size: {new_width}x{new_height}")

            print(f"Final image size: {pil_image.size}")

            # Convert back to NumPy array and normalize to [0, 1] for ComfyUI
            output_image = np.array(pil_image).astype(np.float32) / 255.0
            output_image = torch.from_numpy(output_image).permute(2, 0, 1).unsqueeze(0)
            print(f"Output image shape: {output_image.shape}")

            return (output_image,)
        except Exception as e:
            print(f"Error in BrevResizeNode: {str(e)}")
            import traceback
            traceback.print_exc()
            return (image,)

NODE_CLASS_MAPPINGS = {
    "FaceDetectResizeNode": FaceDetectResizeNode,
    "BrevResizeNode": BrevResizeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDetectResizeNode": "Face Detect and Resize",
    "BrevResizeNode": "Brev Resize",
}