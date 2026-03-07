import cv2
import torch
import numpy as np
import ssl
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
        if obstacle_pixel_count > 0.1 * danger_zone.size:
            danger = True
        else:
            danger = False
        

        # 5. Visual Feedback
        # Draw the rectangle on the original frame (Green if clear, Red if blocked)
        box_color = (0, 255, 0) # Green
        if danger: # If more than 10% of the box is blocked
            box_color = (0, 0, 255) # Red
            cv2.putText(frame, "!!! OBSTACLE !!!", (w//4, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            # Optional: Add audio alert
            # import os; os.system('say "Object" &')

        cv2.rectangle(frame, (zone_left, zone_top), (zone_right, zone_bottom), box_color, 3)

        # --- YOLO26 Segmentation & Classification ---
        # Run YOLO inference
        results = yolo_model.predict(source=frame, conf=0.45, classes=urban_classes, imgsz=320, show=False)
        
        # We start with the original frame instead of the pre-annotated one so we can customize the text
        annotated_frame = frame.copy()

        # Iterate through the detections
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            # Draw segmentation masks if they exist
            if results[0].masks is not None:
                annotated_frame = results[0].plot(boxes=False, labels=False)
            
            for i, box in enumerate(boxes):
                # 1. Get Class Name
                cls_id = int(box.cls[0].item())
                class_name = yolo_model.names[cls_id]

                # 2. Get Bounding Box Coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # 3. Calculate Average Depth within the Bounding Box
                # Ensure coordinates are within image bounds
                h, w = output.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Extract the depth region for this object
                depth_region = output[y1:y2, x1:x2]
                
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