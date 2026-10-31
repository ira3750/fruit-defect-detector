"""Apple defect checker Gradio demo using the final ResNet18 model.

Commands: 
    pip install -r requirements.txt
    python app.py
"""
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import gradio as gr
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "final_model.pt")
UNSURE_BELOW = 0.80      # below this confidence the app says "not sure" (decided using validation table)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("final_model.pt not found next to app.py. "
                            "Copy it from fruit_project/final/ on Drive into this folder.")

# ---- loads the model when the app starts -------------------------------------------------
ckpt = torch.load(MODEL_PATH, map_location=device)
class_names = ckpt["class_names"]

model = models.resnet18(weights=None)            # the weights come from the saved checkpoint
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(device).eval()
for q in model.parameters():
    q.requires_grad = True                       
cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def diagnose(img):
    if img is None:
        return None, "Upload a photo of an apple.", None
    img = img.convert("RGB")
    inp = PREPROCESS(img).unsqueeze(0).to(device)

    probs = torch.softmax(model(inp), dim=1)[0].detach().cpu()
    top = int(probs.argmax())
    conf = float(probs[top])

    heat = cam(input_tensor=inp, targets=[ClassifierOutputTarget(top)])[0]
    rgb = np.asarray(img.resize((224, 224)), dtype=np.float32) / 255.0
    overlay = show_cam_on_image(rgb, heat, use_rgb=True)

    if conf < UNSURE_BELOW:
        verdict = f"Not sure (best guess: {class_names[top]}, {conf:.0%}). Please ask an expert."
    else:
        verdict = f"{class_names[top]} ({conf:.0%} confident)"
    scores = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    return scores, verdict, overlay


examples_dir = os.path.join(HERE, "examples")
examples = None
if os.path.isdir(examples_dir):
    found = sorted(os.path.join(examples_dir, f) for f in os.listdir(examples_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    examples = [[p] for p in found] or None

demo = gr.Interface(
    fn=diagnose,
    inputs=gr.Image(type="pil", label="Apple photo"),
    outputs=[gr.Label(num_top_classes=len(class_names), label="Prediction"),
             gr.Textbox(label="Verdict"),
             gr.Image(label="Grad-CAM: where the model looked")],
    title="Apple defect checker (prototype)",
    description=("Classifies an apple photo as Healthy, Black Rot, Black Pox or Powdery Mildew and shows heatmap to explain decision. \n "
                 "Trained only on images of Manalagi apples from Poncokusumo area, Malang Regency, Indonesia. "),
    examples=examples,
    cache_examples=False,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
