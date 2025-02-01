## **Face Mask Detection Model using CNNs**  

### **Overview**  
This project implements a **Convolutional Neural Network (CNN)** to detect whether a person is wearing a face mask or not. The model is trained on a labeled dataset of masked and unmasked faces.  

### **Dataset**  
- The dataset is sourced from Kaggle: [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset).  
- It contains images categorized as **with mask** and **without mask**.  

### **Model Architecture**  
- **Convolutional Layers**: Extract spatial features from images.  
- **Batch Normalization & Dropout**: Prevent overfitting and improve generalization.  
- **Fully Connected Layers**: Perform classification into mask/no-mask categories.  
- **Activation Function**: ReLU (hidden layers) & Softmax (output layer).  
- **Loss Function**: Categorical Crossentropy.  
- **Optimizer**: Adam.  

### **Training Details**  
- **Batch Size**: 32  
- **Epochs**: 30 (with Early Stopping based on validation loss)  
- **Validation Split**: 20% of the dataset  

### **Results**  
**Classification Report:**  
| Class          | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| **Without Mask** | 0.89      | 0.93   | 0.91     | 397     |
| **With Mask**    | 0.92      | 0.87   | 0.90     | 366     |
| **Accuracy**     | **0.90**  | -      | -        | 763     |
| **Macro Avg**    | 0.91      | 0.90   | 0.90     | 763     |
| **Weighted Avg** | 0.91      | 0.90   | 0.90     | 763     |

![image](https://github.com/user-attachments/assets/f15a7579-800d-4e28-888c-adf0edfb6340)

**Accuracy & Loss Plots:** 

![image](https://github.com/user-attachments/assets/330ea467-7030-4f21-b6ae-8dc09046d912)

![image](https://github.com/user-attachments/assets/918aa0ac-f4dc-44f9-840b-18f8cdf23370)


### **Usage**  
1. **Install Dependencies**  
   ```bash
   pip install tensorflow opencv-python pandas numpy matplotlib
   ```
2. **Run the Model Training**  
   ```python
   python train.py
   ```
3. **Test the Model on New Images**  
   ```python
   python predict.py --image path/to/image.jpg
   ```

### **Future Improvements**  
- Train on a **larger dataset** for better generalization.  
- Implement **real-time detection** using OpenCV and a webcam.  
- Optimize for **mobile deployment** using TensorFlow Lite.  

