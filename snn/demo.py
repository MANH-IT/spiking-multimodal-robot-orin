import torch
import cv2
import numpy as np
from snn.snn_model import SpikingMotionDetector
from spikingjelly.activation_based import functional

def run_snn_demo():
    # Initialize SNN Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpikingMotionDetector().to(device)
    functional.set_step_mode(model, step_mode='m') # Multi-step mode

    cap = cv2.VideoCapture(0)
    print("Running SNN Motion Demo... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame for SNN [T, C, H, W]
        # Simulate time steps by repeating the frame or using a sequence
        img = cv2.resize(frame, (64, 64))
        img = img.transpose(2, 0, 1) / 255.0
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(device) # [1, C, H, W]
        
        # Create a time sequence
        sequence = img_tensor.repeat(4, 1, 1, 1) # 4 time steps

        # Inference
        with torch.no_grad():
            output_spikes = model(sequence)
            functional.reset_net(model) # Crucial for SNN

        # Basic visualization of spikes
        spike_viz = output_spikes[-1].cpu().numpy()[0, 0] # First channel of last step
        spike_viz = (spike_viz * 255).astype(np.uint8)
        spike_viz = cv2.resize(spike_viz, (frame.shape[1], frame.shape[0]))

        cv2.imshow("SNN Spike Output (Motion Features)", spike_viz)
        cv2.imshow("Original", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_snn_demo()
