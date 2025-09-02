import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pickle

# ===============================
# LOAD MODEL
# ===============================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ===============================
# TRANSFORM
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

classes = ["NORMAL", "PNEUMONIA"]

# ===============================
# STREAMLIT UI
# ===============================
st.title("🩺 Chest X-Ray Pneumonia Classifier")
st.write("Upload a chest X-ray image, and the model will predict whether it is **Normal** or **Pneumonia**.")

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = outputs.max(1)
        prediction = classes[predicted.item()]

    st.success(f"### Prediction: {prediction}")
