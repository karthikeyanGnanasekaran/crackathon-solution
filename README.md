# Road Damage Detection (Crackathon Solution)

This repository contains the source code and trained model for the **Crackathon Road Damage Detection Challenge**. 

Our solution utilizes the **YOLOv8-Medium** architecture to automatically detect and classify road anomalies with high speed and accuracy, suitable for real-time infrastructure monitoring.

## 📊 Performance Highlights

- **Model:** YOLOv8m (Medium)
- **mAP@50 (Accuracy):** 62.9%
- **Inference Speed:** ~8.2ms per image (Tesla T4)
- **Framework:** Ultralytics YOLO / PyTorch

## 🎯 Detection Classes

The model is trained to identify 5 specific types of road damage:

1. **Longitudinal Crack**
2. **Transverse Crack**
3. **Alligator Crack**
4. **Pothole**
5. **Other Corruption**

## 📂 Repository Structure

- `Crackathon_Solution.ipynb`: The complete training and inference pipeline (Jupyter Notebook).
- `road_damage_model.pt`: The best performing weights (Save these to your local machine to run predictions).
- `requirements.txt`: List of Python dependencies required to run the code.

## 🚀 Quick Start

### 1. Installation
Clone the repository and install the dependencies:

```bash
git clone [https://github.com/karthikeyanGnanasekaran/crackathon-solution.git](https://github.com/karthikeyanGnanasekaran/crackathon-solution.git)
cd crackathon-solution
pip install -r requirements.txt
