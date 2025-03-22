# Traffic-Sing-DenseNet-IncetionV3
Enhancing Traffic Sign Detection Using a Hybrid Model of DenseNet121 and InceptionV3 with Optimized dense layer


This project demonstrates a hybrid deep learning approach for traffic sign detection using DenseNet121 and InceptionV3 architectures, integrated with an optimized dense layer for final classification. The model is trained on a balanced version of the ICTS Cropped Dataset to ensure accurate detection across all traffic sign classes.

### Project Overview
The research work involves preprocessing and augmentation of traffic sign images, balancing of training data through oversampling, transfer learning using DenseNet121 and InceptionV3, feature fusion via concatenation, and construction of a custom dense classification layer. The model is trained and evaluated using multiple performance metrics including accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC curves.

### Dataset
The dataset consists of categorized traffic sign images from the ICTS Cropped Dataset.
Data is loaded via CSV metadata and image directories, then split into training, validation, and test sets.
Augmentation and preprocessing are applied using Keras ImageDataGenerator. You can download the dataset from here: https://drive.google.com/file/d/1n5uDIKwxy6Wd7hRMX86EiWgi0QW_gBcN/view?usp=sharing

![Trafic signs sample images](https://github.com/user-attachments/assets/31ca8fa6-6bc6-4a14-9bf3-6ed55e730da4)



### Technologies Used
Python, TensorFlow / Keras, NumPy, Pandas, Matplotlib & Seaborn (for visualization), Scikit-learn (for evaluation metrics)

### Model Architecture
Feature extraction using pre-trained DenseNet121 and InceptionV3, Feature fusion by flattening and concatenating base model outputs, Optimized fully connected (dense) layer with 256 neurons and ReLU activation, Final classification layer using softmax activation for multi-class output, Training includes class balancing, data augmentation, and callback mechanisms like early stopping and learning rate reduction.

![model_methodology (2)](https://github.com/user-attachments/assets/87c133b9-444c-4730-8189-17f8d03167ef)


### Results and Evaluation Metrices
Our Proposed work achieved high classification accuracy on validation and test sets, Effective generalization on unseen traffic sign images, Balanced training improved robustness across minority classes, Confusion matrix and ROC-AUC curves indicate strong model performance and class separability.

### Accuracy Curve:
![accuracy_curve](https://github.com/user-attachments/assets/2c1f9867-b362-4534-bcce-1ca2eb1ac9a8)


### Loss Curve
![loss_curve](https://github.com/user-attachments/assets/74af2fe0-4cec-4f64-b732-f32790ebf822)


### Confusion Matrix:
![confusion_matrix](https://github.com/user-attachments/assets/becd9b39-a9cf-49c4-be9a-417e70a4b006)

### ROC Curve:
![roc_curve](https://github.com/user-attachments/assets/95d89c7e-f830-4393-b2ee-f0588815a7e2)


