# app1.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ===============================
# MODEL (must match training)
# ===============================
class SmallCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SmallCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [16,64,64]

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [32,32,32]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [64,16,16]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ===============================
# LOAD MODEL
# ===============================
device = torch.device("cpu")
model = SmallCNN(num_classes=2).to(device)
model.load_state_dict(torch.load("cnn_chest_xray_light.pth", map_location=device))
model.eval()

# ===============================
# TRANSFORM (128×128)
# ===============================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = ["NORMAL", "PNEUMONIA"]

# ===============================
# STREAMLIT UI
# ===============================
st.title("🫁 Chest X-Ray Pneumonia Detection")

uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_class = torch.argmax(probs).item()

    # Show result
    st.subheader("Prediction Result")
    st.write(f"**Predicted Class:** {class_names[pred_class]}")
    st.write(f"**Confidence:** {probs[pred_class].item():.2f}")

    # Probabilities
    st.progress(float(probs[pred_class]))
    st.write({class_names[i]: f"{probs[i].item():.2f}" for i in range(len(class_names))})
