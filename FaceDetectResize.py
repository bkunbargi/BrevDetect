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
                "min_face_size": ("INT", {"default": 128, "min": 64, "max": 1024, "step": 1}),
                "max_face_size": ("INT", {"default": 640, "min": 128, "max": 2048, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "BrevDetect"

    def process_image(self, image, min_face_size, max_face_size):
        # Debug: Print input image shape and type
        print(f"Input image shape: {image.shape}, type: {image.dtype}")
        
        image_rgb = self.comfy_to_cv2(image)
        
        # Debug: Print converted image shape and type
        print(f"Converted image shape: {image_rgb.shape}, type: {image_rgb.dtype}")

        faces = self.detect_faces(image_rgb)
        
        # Debug: Print number of detected faces
        print(f"Number of detected faces: {len(faces) if faces else 0}")

        pil_image = Image.fromarray(image_rgb)

        if faces:
            largest_face_dimensions = max((face['box'][2], face['box'][3]) for face in faces)
            # Debug: Print largest face dimensions
            print(f"Largest face dimensions: {largest_face_dimensions}")
            
            if max(largest_face_dimensions) > max_face_size:
                scale_factor = max_face_size / max(largest_face_dimensions)
            elif max(largest_face_dimensions) < min_face_size:
                scale_factor = min_face_size / max(largest_face_dimensions)
            else:
                scale_factor = 1
            
            # Debug: Print scale factor
            print(f"Scale factor: {scale_factor}")
            
            new_width = int(pil_image.width * scale_factor)
            new_height = int(pil_image.height * scale_factor)
            scaled_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        else:
            scaled_image = pil_image
            # Debug: No resizing needed
            print("No resizing needed.")

        scaled_image_array = np.array(scaled_image)
        
        # Debug: Print scaled image shape and dtype
        print(f"Scaled image shape: {scaled_image_array.shape}, dtype: {scaled_image_array.dtype}")

        output_image = self.cv2_to_comfy(scaled_image_array)
        
        # Debug: Print output image shape and type
        print(f"Output image shape: {output_image.shape}, type: {output_image.dtype}")

        return (output_image,)

    def detect_faces(self, image_rgb):
        print("Detecting faces...")
        detector = MTCNN()
        results = detector.detect_faces(image_rgb)
        if not results:
            print("No faces detected.")
            return results
        largest_face = max(results, key=lambda x: x['box'][2] * x['box'][3])
        print(f"Largest face detected at: {largest_face['box']}")
        return [largest_face]

    def comfy_to_cv2(self, comfy_image):
        # Assuming comfy_image is a torch Tensor with shape [C, H, W]
        image_np = comfy_image.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    def cv2_to_comfy(self, cv2_image):
        # Convert a cv2 image back to a comfyui format
        image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        image_np = image_rgb.astype(np.float32) / 255.0
        return torch.from_numpy(image_np).unsqueeze(0).permute(0, 3, 1, 2)

NODE_CLASS_MAPPINGS = {
    "FaceDetectResizeNode": FaceDetectResizeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDetectResizeNode": "Face Detect and Resize"
}

# import numpy as np
# import torch
# import cv2
# from PIL import Image
# from mtcnn import MTCNN

# class FaceDetectResizeNode:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "image": ("IMAGE",),
#                 "min_face_size": ("INT", {"default": 128, "min": 64, "max": 1024, "step": 1}),
#                 "max_face_size": ("INT", {"default": 640, "min": 128, "max": 2048, "step": 1}),
#             },
#         }

#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "process_image"
#     CATEGORY = "BrevDetect"

#     def process_image(self, image, min_face_size, max_face_size):
#         # Debug: Print input image shape and type
#         print(f"Input image shape: {image.shape}, type: {image.dtype}")
        
#         image_rgb = self.comfy_to_cv2(image)
        
#         # Debug: Print converted image shape and type
#         print(f"Converted image shape: {image_rgb.shape}, type: {image_rgb.dtype}")

#         faces = self.detect_faces(image_rgb)
        
#         # Debug: Print number of detected faces
#         print(f"Number of detected faces: {len(faces)}")

#         pil_image = Image.fromarray(image_rgb)

#         if faces:
#             largest_face_dimensions = max((result['box'][2], result['box'][3]) for result in faces)
#             if max(largest_face_dimensions) > max_face_size:
#                 scale_factor = max_face_size / max(largest_face_dimensions)
#             elif max(largest_face_dimensions) < min_face_size:
#                 scale_factor = min_face_size / max(largest_face_dimensions)
#             else:
#                 scale_factor = 1
            
#             # Debug: Print scale factor
#             print(f"Scale factor: {scale_factor}")
            
#             new_width = int(pil_image.width * scale_factor)
#             new_height = int(pil_image.height * scale_factor)
#             scaled_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
#         else:
#             scaled_image = pil_image

#         scaled_image_array = np.array(scaled_image)
        
#         # Debug: Print scaled image shape
#         print(f"Scaled image shape: {scaled_image_array.shape}")

#         output_image = self.cv2_to_comfy(scaled_image_array)
        
#         # Debug: Print output image shape and type
#         print(f"Output image shape: {output_image.shape}, type: {output_image.dtype}")

#         return (output_image,)

#     def detect_faces(self, image_rgb):
#         detector = MTCNN()
#         results = detector.detect_faces(image_rgb)
#         if not results:
#             return results
#         largest_face = max(results, key=lambda x: x['box'][2] * x['box'][3])
#         return [largest_face]

#     def comfy_to_cv2(self, comfy_image):
#         image_np = comfy_image.squeeze().permute(1, 2, 0).cpu().numpy()
#         image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
#         return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

#     def cv2_to_comfy(self, cv2_image):
#         image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
#         image_np = image_rgb.astype(np.float32) / 255.0
#         return torch.from_numpy(image_np).unsqueeze(0).permute(0, 3, 1, 2)

# NODE_CLASS_MAPPINGS = {
#     "FaceDetectResizeNode": FaceDetectResizeNode
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "FaceDetectResizeNode": "Face Detect and Resize"
# }
