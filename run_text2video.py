import torch
from model import Model

# Initialize the model
model = Model(device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16)

# Define the text prompt and parameters
prompt = "A horse galloping on a street"
params = {
    "t0": 44,
    "t1": 47,
    "motion_field_strength_x": 12,
    "motion_field_strength_y": 12,
    "video_length": 8
}

out_path, fps = f"./text2video_{prompt.replace(' ','_')}.mp4", 4

# Generate video
model.process_text2video(prompt, fps=fps, path=out_path, **params)
