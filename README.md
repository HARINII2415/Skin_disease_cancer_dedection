# 🧬 Skin Cancer Detection using CNN

This project focuses on detecting skin cancer using **Convolutional Neural Networks (CNN)** in Python with TensorFlow and Keras. It classifies images of skin lesions as either **benign or malignant**, supporting early diagnosis and better treatment decisions.

---

## 📌 Features

- Image classification using deep learning
- Image preprocessing and data augmentation
- CNN architecture using TensorFlow/Keras
- Model performance visualization
- Evaluation using confusion matrix & classification report

---

## 🧠 Tech Stack Used

- **Python**
- **TensorFlow & Keras**
- **NumPy & Pandas**
- **Matplotlib & Seaborn**
- **Scikit-learn**
- **OpenCV** (optional)

---

## 📁 Dataset Information

- Dataset: **ISIC (International Skin Imaging Collaboration) Archive**
- Type: Labeled image dataset of skin lesions
- Classes: Benign and Malignant
- Download from: [ISIC Archive](https://www.isic-archive.com)

---

## 🛠️ Installation & Setup

1. **Clone the repository**

git clone https://github.com/your-username/skin-cancer-detection.git
cd skin-cancer-detection
Install the required libraries

bash
Copy
Edit
pip install -r requirements.txt
Run the script

bash
Copy
Edit
python skin_cancer_detection.py
🧬 CNN Model Overview
Input Layer: Image data

Conv2D + MaxPooling Layers (Feature extraction)

Dropout Layers (To reduce overfitting)

Flatten + Dense Layers

Output Layer: Sigmoid activation (for binary classification)

📊 Results
Training Accuracy: ~95%

Validation Accuracy: ~92%

Confusion Matrix and Classification Report show high precision and recall

Graphs for accuracy & loss curves during training included

Model Output: Malignant
Confidence: 94.2%
💡 Future Improvements
Streamlit-based web deployment

Multi-class classification support

Model optimization (Learning rate, Epoch tuning)

Use of Transfer Learning (e.g., ResNet, MobileNet)

🤝 Contributions
Contributions, issues, and feature requests are welcome!
Feel free to fork this repository and submit a pull request.





Made with ❤️ by Harini A

