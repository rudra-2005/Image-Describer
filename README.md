# 🏋️‍♂️ Human Pose Estimation & Classification using Deep Learning and Machine Learning  

This project focuses on **human pose estimation and classification** using **CNN, MediaPipe landmarks, and various machine learning algorithms**. The goal is to classify poses and determine their correctness based on extracted body landmarks.  

---

## 📌 Project Overview  

✅ **Data Analysis**  
- Performed **image-based data analysis** (visuals available in the code).  

✅ **Convolutional Neural Network (CNN)**  
- **Preprocessed and normalized** image data.  
- Achieved **81.91% accuracy** with a standard CNN.  
- Performance graphs for CNN are included in the code.  

✅ **Landmark-Based Classification (Using MediaPipe)**  
- Extracted **body landmarks** (face, hip, shoulder, etc.) using **Google's MediaPipe**.  
- Applied multiple machine learning models for classification:  
  - **Support Vector Machine (SVM)** → **81.15% accuracy**  
  - **K-Nearest Neighbors (KNN)** → **74.77% accuracy**  
  - **Random Forest** → **87.04% accuracy**  

✅ **Pre-trained Model: VGG16**  
- Implemented **VGG16 (pre-trained CNN)** for pose classification.  
- After **50 epochs**, achieved:  
  - **Training Accuracy**: 93.06%  
  - **Validation Accuracy**: 88.37%  

✅ **Pose Correctness Analysis**  
- Increased **Random Forest** tree count **from 100 to 150**.  
- **Normalized values** based on **shoulder distance** and **hip center**.  
- Achieved **98% accuracy** in determining **how correct a pose is**.  
- **F1-score, support, precision, and recall** are provided in the code.  
- Implemented a **confidence score** system to quantify pose correctness.  

---

## 📊 Results  

| Model | Accuracy |
|--------|---------|
| **CNN (Custom)** | 81.91% |
| **SVM** | 81.15% |
| **KNN** | 74.77% |
| **Random Forest** | 87.04% |
| **VGG16 (Pre-trained CNN)** | 88.37% (Validation) |
| **Pose Correctness (Random Forest, 150 trees)** | 98% |

---

## 🖥️ Tech Stack  

- **Deep Learning**: TensorFlow, Keras (CNN, VGG16)  
- **Machine Learning**: Scikit-Learn (SVM, KNN, Random Forest)  
- **Pose Estimation**: MediaPipe  
- **Data Processing**: OpenCV, NumPy, Pandas  
- **Visualization**: Matplotlib, Seaborn  



