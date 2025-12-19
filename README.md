# Fine-Grained Bird Species Identification using Vision Transformers (ViT)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-100%25-brightgreen)

## 📌 Project Overview
This project addresses the challenge of **Fine-Grained Visual Categorization (FGVC)** by identifying 20 specific bird species with high precision. Unlike general classification, fine-grained identification requires the model to distinguish between sub-categories that share extremely similar physical traits. 

We utilize a **Vision Transformer (ViT)** architecture, specifically **DeiT-Tiny**, to capture the intricate localized features necessary for this task.



## 🚀 Key Features
* **Transformer-Based Learning:** Leverages self-attention mechanisms to focus on discriminative bird features like beak shape and wing patterns.
* **State-of-the-Art Performance:** Achieved a perfect **100% Accuracy, Precision, and Recall** on the test set.
* **OOD Robustness:** Implements a **80% Confidence Threshold** to detect Out-of-Distribution (OOD) images, such as illustrations or species not present in the training data.
* **Interactive Inference:** Includes an integrated Google Colab widget for instant image uploads and classification.

## 📊 Dataset
The model was trained and validated on the **Birds 20 Species - Image Classification** dataset:
* **Total Images:** 3,413 natural photographs.
* **Categories:** 20 distinct species (e.g., Abbotts Booby, African Emerald Cuckoo, American Kestrel).
* **Split:** 3,208 Training images, 100 Validation images, and 100 Test images.

## 🛠️ Technical Implementation
* **Model:** `deit_tiny_patch16_224` (via `timm` library).
* **Optimizer:** AdamW with a learning rate of $1e-4$.
* **Loss Function:** Cross-Entropy Loss.
* **Input Size:** 224 x 224 pixels.

## 📈 Performance Results
The model reached near-total convergence within 10 epochs.

| Metric | Test Set Result |
| :--- | :--- |
| **Accuracy** | 100% |
| **Precision** | 1.00 |
| **Recall** | 1.00 |
| **F1-Score** | 1.00 |



### **Observations on Robustness**
* **Natural Photos:** High confidence scores (>99%).
* **Illustrations:** Lower confidence (~65%), correctly identifying domain shift.
* **Unknown Species:** Flagged correctly via thresholding when the model is presented with non-dataset species.

## 💻 How to Use
1.  **Clone the Repo:**
    ```bash
    git clone [https://github.com/thomas271001/bird-vit-classification.git](https://github.com/thomas271001/bird-vit-classification.git)
    ```
2.  **Install Dependencies:**
    ```bash
    pip install torch torchvision timm matplotlib pillow ipywidgets
    ```
3.  **Run Inference:** Open the Colab notebook and use the upload widget to test your own bird images.

## 📜 License
This project is licensed under the MIT License.
