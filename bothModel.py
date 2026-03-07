import cv2
import torch
import numpy as np
import ssl
import time
from ultralytics import YOLO

# Fix for SSL: CERTIFICATE_VERIFY_FAILED error in PyTorch Hub downloads
ssl._create_default_https_context = ssl._create_unverified_context

def main():
    # 1. Load the MiDaS model (MiDaS_small is best for real-time webcam use)
    # Use the high-accuracy model
    model_type = "DPT_Hybrid" 
    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    # Move to Mac GPU (MPS) for speed
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    midas.to(device)
    midas.eval()

    # Use the DPT transform for the Large model
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform

    # 2. Load the lightest YOLO26 segmentation model (nano)
    try:
        yolo_model = YOLO('yolo26n-seg.pt')
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # Define urban environment classes (COCO dataset indices)
    # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck, 9: traffic light, 10: fire hydrant, 11: stop sign, 12: street sign, 13: bench
    urban_classes = [0, 1, 2, 3, 5, 7, 9, 10, 11, 12, 13, 15, 16]

    # 3. Open Webcam
    # `1` (or higher) is usually the iPhone Continuity Camera if connected/enabled.
    # `0` represents the default built-in Mac webcam.
    camera_index = 1
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened() or not cap.read()[0]:
        print(f"Warning: Could not open camera at index {camera_index}. Falling back to index 0.")
        cap.release()
        camera_index = 0
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting webcam... Press 'q' to quit the application.")

    danger_start_time = None
    obstacle_printed = False
    class_first_seen = {}  # class_name -> first_seen_time
    class_printed = set()  # classes already printed for current streak

    while cap.isOpened():
        success, frame = cap.read()
        
        # If the frame is empty, don't crash! Just wait a millisecond and try again.
        if not success:
            print("Waiting for stream...")
            cv2.waitKey(100) 
            continue

        # --- MiDaS Depth Estimation ---
        # Convert BGR (OpenCV default) to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(img).to(device)
        
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
        output = prediction.cpu().numpy()
        output_norm = cv2.normalize(output, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        output_color = cv2.applyColorMap(output_norm, cv2.COLORMAP_MAGMA)

        # --- OBSTACLE DETECTION LOGIC ---
        h, w = output_norm.shape

        # 1. Define the "Danger Zone" (The central column of the screen)
        # Looking from slightly above the center (h*0.4) to just above your feet (h*0.85)
        zone_top, zone_bottom = 0, int(h * 0.7)
        zone_left, zone_right = int(w * 0.3), int(w * 0.7)

        # 2. Extract the data from this zone
        danger_zone = output_norm[zone_top:zone_bottom, zone_left:zone_right]
        max_val = np.max(danger_zone)
        min_val = np.min(danger_zone)
        range_val = max_val - min_val
        print(f"Min: {min_val}, Max: {max_val}, Range: {range_val}")


        _, obstacle_mask = cv2.threshold(danger_zone, min_val + 0.8 * range_val, 255, cv2.THRESH_BINARY)
        obstacle_pixel_count = np.count_nonzero(obstacle_mask)
        if obstacle_pixel_count > 0.1 * danger_zone.size and max_val >= 30:
            danger = True
        else:
            danger = False

        # Print OBSTACLE if danger is continuous for 1 second (once per danger episode)
        if danger:
            if danger_start_time is None:
                danger_start_time = time.time()
            elif not obstacle_printed and time.time() - danger_start_time >= 0.75:
                cv2.putText(output_color, "OBSTACLE", (w//4, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                obstacle_printed = True
        else:
            danger_start_time = None
            obstacle_printed = False

        # 5. Visual Feedback
        # Draw the rectangle on the original frame (Green if clear, Red if blocked)
        box_color = (0, 255, 0) # Green
        if danger: # If more than 10% of the box is blocked
            box_color = (0, 0, 255) # Red

            # Additionally, draw a square around the obstacle region within the danger zone
            ys, xs = np.where(obstacle_mask > 0)
            if xs.size > 0 and ys.size > 0:
                # Bounding box in danger_zone coordinates
                local_x1, local_x2 = int(xs.min()), int(xs.max())
                local_y1, local_y2 = int(ys.min()), int(ys.max())

                # Convert to full-frame coordinates
                x1_full = zone_left + local_x1
                x2_full = zone_left + local_x2
                y1_full = zone_top + local_y1
                y2_full = zone_top + local_y2

                # Make the box roughly square with a small padding
                width = x2_full - x1_full
                height = y2_full - y1_full
                half_side = int(max(width, height) / 2)
                half_side = int(half_side * 1.1)  # small padding

                cx = (x1_full + x2_full) // 2
                cy = (y1_full + y2_full) // 2

                sq_x1 = max(0, cx - half_side)
                sq_y1 = max(0, cy - half_side)
                sq_x2 = min(w - 1, cx + half_side)
                sq_y2 = min(h - 1, cy + half_side)

                # Draw the generalized obstacle square in yellow
                cv2.rectangle(output_color, (sq_x1, sq_y1), (sq_x2, sq_y2), (0, 255, 255), 2)
            # cv2.putText(frame, str(avg_gradient), (w//4, h//2), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            # Optional: Add audio alert
            # import os; os.system('say "Object" &')

        cv2.rectangle(output_color, (zone_left, zone_top), (zone_right, zone_bottom), box_color, 3)

        # --- YOLO26 Segmentation & Classification ---
        # Run YOLO inference
        results = yolo_model.predict(source=frame, conf=0.45, classes=urban_classes, imgsz=320, show=False)
        
        # We start with the original frame instead of the pre-annotated one so we can customize the text
        annotated_frame = frame.copy()

        # Iterate through the detections
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            detected_classes = set()

            # Draw segmentation masks if they exist
            if results[0].masks is not None:
                annotated_frame = results[0].plot(boxes=False, labels=False)

            for i, box in enumerate(boxes):
                # 1. Get Class Name
                cls_id = int(box.cls[0].item())
                class_name = yolo_model.names[cls_id]
                detected_classes.add(class_name)

                # 2. Get Bounding Box Coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # 3. Calculate Average Depth within the Bounding Box
                # Ensure coordinates are within image bounds
                h, w = output.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Extract the depth region for this object
                depth_region = output_norm[y1:y2, x1:x2]
                
                if depth_region.size > 0:
                    # MiDaS outputs relative inverse depth. 
                    # We invert it to get a pseudo-distance measurement.
                    avg_inverse_depth = np.mean(depth_region)
                    # Add a small epsilon to prevent division by zero
                    pseudo_distance = 1.0 / (avg_inverse_depth + 1e-6) 
                    
                    # Scale for readability (arbitrary scaling factor for display purposes)
                    display_distance = pseudo_distance * 1000

                    # 4. Draw Custom Label (Class + Distance)
                    label = f"{class_name}: {display_distance:.1f} units"
                    
                    # Draw text background
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 0), cv2.FILLED)
                    
                    # Draw text
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Print class name if instance in frame for > 0.5 seconds (once per streak)
            for cls in detected_classes:
                if cls not in class_first_seen:
                    class_first_seen[cls] = time.time()
                elif cls not in class_printed and time.time() - class_first_seen[cls] >= 0.75:
                    print(cls)
                    class_printed.add(cls)
            # Draw confirmed class names on the depth view (output_color)
            y_offset = 30
            for cls in (class_printed & detected_classes):
                cv2.putText(output_color, cls, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                y_offset += 30
            for cls in list(class_first_seen.keys()):
                if cls not in detected_classes:
                    del class_first_seen[cls]
                    class_printed.discard(cls)
        else:
            class_first_seen.clear()
            class_printed.clear()

        # --- Display the results ---
        # Combine the two frames horizontally into a single window
        # Both should be the same size (640x480)
        combined_frame = np.hstack((annotated_frame, output_color))

        cv2.imshow("WalkAssist: YOLO26 Segmentation (Left) + MiDaS Depth (Right)", combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()