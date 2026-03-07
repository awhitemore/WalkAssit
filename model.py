import cv2
import torch
import numpy as np

# 1. Load the MiDaS model (MiDaS_small is best for real-time webcam use)
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# 2. Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# 3. Load transforms to resize and normalize the image
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform if model_type == "MiDaS_small" else midas_transforms.dpt_transform

# 4. Open Webcam
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    success, frame = cap.read()
    
    # If the frame is empty, don't crash! Just wait a millisecond and try again.
    if not success:
        print("Waiting for iPhone stream...")
        cv2.waitKey(100) 
        continue
    if not success:
        break

    # Convert BGR (OpenCV default) to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply transforms and move to device
    input_batch = transform(img).to(device)

    # Perform Inference (Predict Depth)
    with torch.no_grad():
        prediction = midas(input_batch)

        # Resize to original resolution
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()

    # Normalize the output for visualization (0 to 255)
    output_norm = cv2.normalize(output, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply a colormap for better visibility (MAGMA makes depth intuitive)
    output_color = cv2.applyColorMap(output_norm, cv2.COLORMAP_MAGMA)

    # Display the result
    cv2.imshow('Webcam Feed', frame)
    cv2.imshow('Depth Map', output_color)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()