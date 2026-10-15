import torch
from torchvision import transforms

from helper_lib.model import get_model


CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


device = torch.device("cpu")


model = get_model("CNN")

checkpoint = torch.load(
    "models/cnn_model.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


def classify_image(image):
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    predicted_class = CLASSES[predicted.item()]

    return predicted_class