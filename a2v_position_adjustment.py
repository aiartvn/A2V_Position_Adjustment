import torch
import numpy as np
from PIL import Image
import cv2
import time

class A2V_Multi_Image_Composite:
    def __init__(self):
        self.last_update_time = 0
        self.update_interval = 0.016  # ~60 FPS
        
        self.preview_window = "A2V Multi Image Composite"
        self.control_window = "Control Panel"
        
        self.dragging = False
        self.selected_image = 0
        self.current_images = []
        self.current_bg = None
        self.image_params = []
        
        self.scale_mode = False
        self.rotate_mode = False
        self.show_grid = False
        
        self.cached_images = []
        self.cached_params = []
        self.drag_start_pos = None
        self.drag_start_window = None
        
        cv2.setNumThreads(8)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background": ("IMAGE",),
                "images": ("IMAGE",),
                "positions": ("INT", {"default": [0, 0], "min": -4096, "max": 4096, "step": 1}),
                "scales": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "rotations": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "blend_modes": (["normal", "multiply", "screen", "overlay"],),
                "opacities": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "enable_preview": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite_images"
    CATEGORY = "A2V/Image"
    OUTPUT_NODE = True

    def create_windows(self):
        try:
            cv2.namedWindow(self.preview_window, cv2.WINDOW_NORMAL)
            cv2.namedWindow(self.control_window, cv2.WINDOW_NORMAL)

            bg_h, bg_w = self.current_bg.shape[:2]
            win_w = min(1024, bg_w)
            win_h = int(win_w * (bg_h / bg_w))
            
            cv2.resizeWindow(self.preview_window, win_w, win_h)
            cv2.resizeWindow(self.control_window, 400, 200)
            
            cv2.moveWindow(self.preview_window, 100, 100)
            cv2.moveWindow(self.control_window, 100 + win_w + 20, 100)

            # Create trackbars for selected image
            cv2.createTrackbar('Image', self.control_window,
                             self.selected_image, len(self.current_images)-1,
                             lambda x: self.select_image(x))
            cv2.createTrackbar('Scale %', self.control_window,
                             int(self.image_params[self.selected_image]['scale']*100), 1000,
                             lambda x: self.update_param('scale', x/100))
            cv2.createTrackbar('Rotate°', self.control_window,
                             int(self.image_params[self.selected_image]['rotation']+180), 360,
                             lambda x: self.update_param('rotation', x-180))
            cv2.createTrackbar('Opacity %', self.control_window,
                             int(self.image_params[self.selected_image]['opacity']*100), 100,
                             lambda x: self.update_param('opacity', x/100))
            cv2.createTrackbar('Grid', self.control_window,
                             int(self.show_grid), 1,
                             lambda x: self.toggle_grid(x))

            cv2.setMouseCallback(self.preview_window, self.mouse_callback)
            
        except Exception as e:
            print(f"Warning: Control panel creation failed: {str(e)}")

    def draw_grid(self, img):
        h, w = img.shape[:2]
        overlay = np.zeros_like(img)
        
        grid_color = (128, 128, 128)
        center_color = (0, 255, 0)
        
        for x in range(0, w, 50):
            cv2.line(overlay, (x, 0), (x, h), grid_color, 1)
        for y in range(0, h, 50):
            cv2.line(overlay, (0, y), (w, y), grid_color, 1)

        cv2.line(overlay, (w//2, 0), (w//2, h), center_color, 1)
        cv2.line(overlay, (0, h//2), (w, h//2), center_color, 1)

        return cv2.addWeighted(overlay, 0.3, img, 1.0, 0)

    def create_status_overlay(self, img):
        params = self.image_params[self.selected_image]
        status = f"Image {self.selected_image+1}/{len(self.current_images)} | " \
                f"Pos: ({params['x_pos']}, {params['y_pos']}) | " \
                f"Scale: {params['scale']:.2f}x | " \
                f"Rot: {params['rotation']}° | " \
                f"Opacity: {params['opacity']:.2f} | " \
                f"Blend: {params['blend_mode']} | " \
                f"{'Scale' if self.scale_mode else 'Rotate' if self.rotate_mode else 'Move'} Mode"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        
        (text_w, text_h), baseline = cv2.getTextSize(status, font, font_scale, thickness)
        
        cv2.rectangle(img, (8, 8), (text_w + 12, text_h + 12), (0, 0, 0), -1)
        cv2.putText(img, status, (10, text_h + 10), font, font_scale, (255, 255, 255), thickness)
        
        return img

    def apply_blend_mode(self, background, foreground, mode, opacity):
        if mode == "normal":
            return background * (1 - opacity) + foreground * opacity
        elif mode == "multiply":
            return background * foreground * opacity + background * (1 - opacity)
        elif mode == "screen":
            return 1 - (1 - background) * (1 - foreground * opacity)
        elif mode == "overlay":
            mask = background >= 0.5
            result = np.zeros_like(background)
            result[mask] = 1 - 2 * (1 - background[mask]) * (1 - foreground[mask])
            result[~mask] = 2 * background[~mask] * foreground[~mask]
            return result * opacity + background * (1 - opacity)
        return background

    def transform_image(self, img, params):
        h, w = img.shape[:2]
        center = (w/2, h/2)
        
        M = cv2.getRotationMatrix2D(center, params['rotation'], params['scale'])
        
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        has_alpha = img.shape[2] == 4
        if has_alpha:
            rgb_img = img[:, :, :3]
            alpha_img = img[:, :, 3]
            
            rgb_transformed = cv2.warpAffine(
                rgb_img, M, (new_w, new_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
            
            alpha_transformed = cv2.warpAffine(
                alpha_img, M, (new_w, new_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            result = np.dstack((rgb_transformed, alpha_transformed))
        else:
            result = cv2.warpAffine(
                img, M, (new_w, new_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
            
        return result

    def update_preview(self):
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return

        if not self.current_images or self.current_bg is None:
            return

        try:
            result = self.current_bg.copy()
            
            # Composite all images
            for idx, (img, params) in enumerate(zip(self.current_images, self.image_params)):
                transformed = self.transform_image(img, params)
                
                x_pos = params['x_pos']
                y_pos = params['y_pos']
                h_img, w_img = transformed.shape[:2]
                h_bg, w_bg = result.shape[:2]
                
                x1 = max(0, x_pos)
                y1 = max(0, y_pos)
                x2 = min(w_bg, x_pos + w_img)
                y2 = min(h_bg, y_pos + h_img)
                x1_img = max(0, -x_pos)
                y1_img = max(0, -y_pos)
                
                if x1 < w_bg and y1 < h_bg and x2 > 0 and y2 > 0:
                    img_roi = transformed[y1_img:y1_img + (y2-y1), x1_img:x1_img + (x2-x1)]
                    
                    if transformed.shape[2] == 4:
                        mask = img_roi[:, :, 3:4] / 255.0
                        bg_region = result[y1:y2, x1:x2]
                        fg_region = img_roi[:, :, :3]
                        
                        blended = self.apply_blend_mode(
                            bg_region, fg_region,
                            params['blend_mode'],
                            params['opacity'] * mask
                        )
                        result[y1:y2, x1:x2] = blended
                    else:
                        blended = self.apply_blend_mode(
                            result[y1:y2, x1:x2],
                            img_roi,
                            params['blend_mode'],
                            params['opacity']
                        )
                        result[y1:y2, x1:x2] = blended

            if self.show_grid:
                result = self.draw_grid(result)
            result = self.create_status_overlay(result)
            
            cv2.imshow(self.preview_window, result)
            self.last_update_time = current_time
            
        except Exception as e:
            print(f"Warning: Preview update error: {str(e)}")

    def select_image(self, idx):
        self.selected_image = idx
        params = self.image_params[idx]
        
        cv2.setTrackbarPos('Scale %', self.control_window, int(params['scale']*100))
        cv2.setTrackbarPos('Rotate°', self.control_window, int(params['rotation']+180))
        cv2.setTrackbarPos('Opacity %', self.control_window, int(params['opacity']*100))
        
        self.update_preview()

    def mouse_callback(self, event, x, y, flags, param):
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return

        params = self.image_params[self.selected_image]
        shift_pressed = flags & cv2.EVENT_FLAG_SHIFTKEY
        ctrl_pressed = flags & cv2.EVENT_FLAG_CTRLKEY

        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start_pos = (params['x_pos'], params['y_pos'])
            self.drag_start_window = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            dx = x - self.drag_start_window[0]
            dy = y - self.drag_start_window[1]
            
            if shift_pressed:
                dx = dx // 5
                dy = dy // 5
            elif ctrl_pressed:
                dx = dx // 10
                dy = dy // 10
            
            params['x_pos'] = self.drag_start_pos[0] + dx
            params['y_pos'] = self.drag_start_pos[1] + dy
            self.update_preview()
            self.last_update_time = current_time
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        
        elif event == cv2.EVENT_MOUSEWHEEL:
            if self.rotate_mode:
                delta = 1 if flags > 0 else -1
                if shift_pressed:
                    delta /= 5
                elif ctrl_pressed:
                    delta /= 10
                    
                new_rotation = (params['rotation'] + delta) % 360
                params['rotation'] = new_rotation
                cv2.setTrackbarPos('Rotate°', self.control_window, int(new_rotation + 180))
            else:
                scale_factor = 1.1 if flags > 0 else 0.9
                if ctrl_pressed:
                    scale_factor = 1.01 if flags > 0 else 0.99
                elif shift_pressed:
                    scale_factor = 1.05 if flags > 0 else 0.95
                
                new_scale = params['scale'] * scale_factor
                new_scale = max(0.1, min(10.0, new_scale))
                
                params['scale'] = new_scale
                cv2.setTrackbarPos('Scale %', self.control_window, int(new_scale * 100))
            
            self.update_preview()

    def toggle_grid(self, value):
        self.show_grid = bool(value)
        self.update_preview()

    def update_param(self, name, value):
        params = self.image_params[self.selected_image]
        old_value = params.get(name)
        params[name] = value
        self.update_preview()

    def cycle_blend_mode(self):
        params = self.image_params[self.selected_image]
        blend_modes = ["normal", "multiply", "screen", "overlay"]
        current_idx = blend_modes.index(params['blend_mode'])
        next_idx = (current_idx + 1) % len(blend_modes)
        params['blend_mode'] = blend_modes[next_idx]
        self.update_preview()

    def initialize_image_params(self, num_images, positions, scales, rotations, blend_modes, opacities):
        self.image_params = []
        for i in range(num_images):
            pos_idx = min(i * 2, len(positions) - 2)
            self.image_params.append({
                'x_pos': positions[pos_idx],
                'y_pos': positions[pos_idx + 1],
                'scale': scales[min(i, len(scales) - 1)],
                'rotation': rotations[min(i, len(rotations) - 1)],
                'blend_mode': blend_modes[min(i, len(blend_modes) - 1)],
                'opacity': opacities[min(i, len(opacities) - 1)]
            })

    def composite_images(self, background, images, positions, scales, rotations, blend_modes, opacities, enable_preview):
        try:
            # Convert tensor inputs to numpy arrays
            if isinstance(images, torch.Tensor):
                images = [img for img in images]
            
            bg = (background[0].cpu().numpy() * 255).astype(np.uint8)
            imgs = [(img.cpu().numpy() * 255).astype(np.uint8) for img in images]
            
            # Initialize parameters
            self.current_bg = bg.copy()
            self.current_images = imgs
            self.initialize_image_params(len(imgs), positions, scales, rotations, blend_modes, opacities)
            self.selected_image = 0

            if enable_preview:
                try:
                    self.create_windows()
                    self.update_preview()
                    
                    while True:
                        key = cv2.waitKey(1) & 0xFF
                        
                        if key == 27 or key == 13 or key == ord('q'):  # ESC, ENTER or Q
                            break
                        elif key == ord('b'):  # Cycle blend modes
                            self.cycle_blend_mode()
                        elif key == ord('g'):  # Toggle grid
                            self.show_grid = not self.show_grid
                            cv2.setTrackbarPos('Grid', self.control_window, int(self.show_grid))
                            self.update_preview()
                        elif key == ord('r'):  # Reset current image
                            params = self.image_params[self.selected_image]
                            params.update({
                                'x_pos': 0, 'y_pos': 0,
                                'scale': 1.0, 'rotation': 0.0,
                                'opacity': 1.0, 'blend_mode': 'normal'
                            })
                            cv2.setTrackbarPos('Scale %', self.control_window, 100)
                            cv2.setTrackbarPos('Rotate°', self.control_window, 180)
                            cv2.setTrackbarPos('Opacity %', self.control_window, 100)
                            self.update_preview()
                        elif key == ord('s'):  # Toggle scale mode
                            self.scale_mode = not self.scale_mode
                            self.rotate_mode = False
                            self.update_preview()
                        elif key == ord('t'):  # Toggle rotate mode
                            self.rotate_mode = not self.rotate_mode
                            self.scale_mode = False
                            self.update_preview()
                        elif key == ord('c'):  # Center current image
                            params = self.image_params[self.selected_image]
                            img = self.current_images[self.selected_image]
                            img_h, img_w = img.shape[:2]
                            bg_h, bg_w = self.current_bg.shape[:2]
                            params['x_pos'] = (bg_w - img_w) // 2
                            params['y_pos'] = (bg_h - img_h) // 2
                            self.update_preview()
                        elif key == ord('n'):  # Next image
                            next_idx = (self.selected_image + 1) % len(self.current_images)
                            self.select_image(next_idx)
                        elif key == ord('p'):  # Previous image
                            prev_idx = (self.selected_image - 1) % len(self.current_images)
                            self.select_image(prev_idx)
                            
                finally:
                    cv2.destroyAllWindows()
                    for _ in range(4):
                        cv2.waitKey(1)

            # Create final composite
            result = self.current_bg.copy()
            
            for img, params in zip(self.current_images, self.image_params):
                transformed = self.transform_image(img, params)
                
                # Convert to PIL for composition
                if transformed.shape[2] == 4:
                    img_pil = Image.fromarray(transformed, mode='RGBA')
                else:
                    img_pil = Image.fromarray(transformed, mode='RGB')
                
                if result.shape[2] == 4:
                    result_pil = Image.fromarray(result, mode='RGBA')
                else:
                    result_pil = Image.fromarray(result, mode='RGB')
                
                # Composite
                if transformed.shape[2] == 4:
                    mask = img_pil.split()[3]
                    result_pil.paste(img_pil, (params['x_pos'], params['y_pos']), mask)
                else:
                    result_pil.paste(img_pil, (params['x_pos'], params['y_pos']))
                
                result = np.array(result_pil)

            # Convert to tensor
            result_array = result.astype(np.float32) / 255.0
            result_tensor = torch.from_numpy(result_array)[None,]
            
            # Clear resources
            self.current_images = []
            self.current_bg = None
            self.image_params = []
            
            return (result_tensor,)
            
        except Exception as e:
            print(f"Error in composite_images: {str(e)}")
            cv2.destroyAllWindows()
            return (background,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "A2V_Multi_Image_Composite": A2V_Multi_Image_Composite
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "A2V_Multi_Image_Composite": "A2V Multi Image Composite"
}