import os
import torch
from torchvision import transforms
from PIL import Image
from django.shortcuts import render
from .forms import ImageUploadForm
from .model import CancerNet   # 👈 we will create model.py next

# Path to your trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cancer_detector.pth")

# Load the model
model = CancerNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()   # set to evaluation mode

# Transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def classify_image(request):
    prediction = None
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Get uploaded image
            image = Image.open(request.FILES['image']).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)  # add batch dimension

            # Run through the model
            with torch.no_grad():
                output = model(image_tensor)
                _, predicted = torch.max(output, 1)

            # ⚠️ Adjust labels according to your training mapping
            if predicted.item() == 0:
                prediction = "Cancer"
            else:
                prediction = "Normal"
    else:
        form = ImageUploadForm()

    return render(request, "classifier/result.html", {"form": form, "prediction": prediction})
