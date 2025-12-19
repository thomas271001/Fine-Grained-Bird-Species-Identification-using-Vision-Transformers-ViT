import torch
import torch.nn as nn
import timm  # PyTorch Image Models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Data Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Load Data from your unzipped folders
base_path = '/content/drive/MyDrive/bird_data'
train_ds = datasets.ImageFolder(f'{base_path}/train', transform=transform)
valid_ds = datasets.ImageFolder(f'{base_path}/valid', transform=transform)
test_ds  = datasets.ImageFolder(f'{base_path}/test',  transform=transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)

print(f"Classes found: {len(train_ds.classes)}")
print(f"Training samples: {len(train_ds)}")
# Initialize a pre-trained Tiny Vision Transformer with 20 output classes
model = timm.create_model('deit_tiny_patch16_224', pretrained=True, num_classes=20)
model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
def train_and_validate(epochs=10):
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss, train_correct = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)

        # --- Validation Phase ---
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)

        train_acc = train_correct.double() / len(train_ds)
        val_acc = val_correct.double() / len(valid_ds)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss/len(train_ds):.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss/len(valid_ds):.4f}   | Acc: {val_acc:.4f}")
        print("-" * 30)

# Start training
train_and_validate(epochs=10)
model.eval()
test_correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        test_correct += torch.sum(preds == labels.data)

print(f"Final Test Accuracy: {test_correct.double() / len(test_ds):.4f}")
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate Final Accuracy
test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"Final Test Accuracy: {test_acc * 100:.2f}%")

# Plot Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=train_ds.classes, yticklabels=train_ds.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Bird Species Confusion Matrix')
plt.show()

# Detailed Report
print("\nDetailed Classification Report:")
print(classification_report(all_labels, all_preds, target_names=train_ds.classes))
import ipywidgets as widgets
from IPython.display import display, clear_output
import io
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# 1. Configuration
# If confidence is below this, the model will flag it as "Unknown"
THRESHOLD = 0.80

# 2. UI Components
uploader = widgets.FileUpload(accept='image/*', multiple=False)
output = widgets.Output()

def on_upload_change(change):
    with output:
        clear_output()
        if not uploader.value:
            return

        # Robust extraction for both ipywidgets 7.x and 8.x
        try:
            if isinstance(uploader.value, dict):
                # Older format: {filename: {content: ...}}
                first_file_name = list(uploader.value.keys())[0]
                file_info = uploader.value[first_file_name]
            else:
                # Newer format: list/tuple of dicts
                file_info = uploader.value[0]

            content = file_info['content']
        except (IndexError, KeyError):
            print("Error: Could not access the uploaded file.")
            return

        # Load and preprocess
        img = Image.open(io.BytesIO(content)).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)

        # Inference
        model.eval()
        with torch.no_grad():
            logits = model(img_t)
            # Standard Softmax to get probability distribution
            probs = F.softmax(logits, dim=1)
            # Get Top 3 most likely species
            top_probs, top_idxs = torch.topk(probs, 3)

        # Primary Result
        best_prob = top_probs[0][0].item()
        best_idx = top_idxs[0][0].item()
        class_name = train_ds.classes[best_idx]

        # Apply Confidence Logic
        if best_prob < THRESHOLD:
            print(f"⚠️ LOW CONFIDENCE: {best_prob*100:.2f}%")
            print(f"Prediction: {class_name} (Possible Out-of-Distribution Image)")
            title_color = 'red'
        else:
            print(f"✅ Prediction: {class_name}")
            print(f"Confidence: {best_prob*100:.2f}%")
            title_color = 'green'

        # Show Top 3 for comparison
        print("\n--- Top 3 Matches ---")
        for i in range(3):
            p = top_probs[0][i].item()
            idx = top_idxs[0][i].item()
            name = train_ds.classes[idx]
            print(f"{i+1}. {name}: {p*100:.2f}%")

        # Visual Output
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"Target: {class_name}", color=title_color, fontsize=14)
        plt.axis('off')
        plt.show()

# Link and Display
uploader.observe(on_upload_change, names='value')
print("Upload a bird image for Fine-Grained Identification:")
display(uploader, output)
