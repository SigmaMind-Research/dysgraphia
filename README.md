# Dysgraphia

# Classification Models

## Overview

This project involves the development and evaluation of several machine learning classifiers for a classification task. The task involves predicting the label based on a set of features. The classifiers implemented include Random Forest, Logistic Regression, k-Nearest Neighbors, and Decision Tree classifiers.

## Dataset

- **`feature_100_true.csv`**: Dataset containing the features and labels for the classification task.

## Usage

1. **Prepare Data**: Place your dataset file (`feature_100_true.csv`) in the project directory.
2. **Data Exploration**: Explore the dataset using the provided script. This script will visualize the data and perform necessary preprocessing steps.
3. **Model Training and Evaluation**:

   - The script trains various classifiers on the data using the training set.
   - For each classifier, the training process involves:
     - Loading the training data.
     - Initializing the classifier (e.g., Random Forest, Logistic Regression, etc.).
     - Training the classifier on the training data.
     - Evaluating the trained classifier on the testing data.
     - Calculating performance metrics such as accuracy, precision, recall, and F1-score.
   - Results including performance metrics, confusion matrices, and visualizations will be generated.

## Results

- The performance metrics (accuracy, precision, recall, F1-score) of each classifier are displayed in the console output.
- Confusion matrices and classification reports are generated for each classifier.
- Visualizations, including scatter plots and correlation heatmaps, are saved in the `/figures` directory for data exploration.

# Imagenet 16

## Overview

This project involves the development and evaluation of several machine learning classifiers for a classification task. The task involves predicting the label based on a set of features. The classifiers implemented include Random Forest, Logistic Regression, k-Nearest Neighbors, and Decision Tree classifiers.

## Dataset

- **Data Source**: The dataset used in this project is stored in the `/content/drive/MyDrive/data` directory.
- **Data Format**: The dataset consists of images in various formats, labeled based on certain criteria.

## Script Overview

- **Script**: The script provided in this project is used to preprocess the images, train a deep learning model using transfer learning (VGG16), and evaluate its performance.
- **Preprocessing**: The script preprocesses the images using various techniques such as resizing, enhancing sharpness, and splitting into smaller pieces.
- **Model Training**: The script utilizes transfer learning with the VGG16 architecture pretrained on ImageNet. It adds custom dense layers on top of the VGG16 base and trains the model on the preprocessed image data.
- **Model Evaluation**: After training, the script evaluates the trained model using a separate validation dataset and computes various performance metrics such as accuracy, precision, recall, and F1-score.
- **Results Visualization**: The script generates visualizations of training and validation accuracy and loss over epochs, as well as a confusion matrix to visualize the model's performance.

## Usage

1. **Prepare Data**: Place your dataset in the `/content/drive/MyDrive/data` directory.
2. **Review Results**: After running the script, review the generated visualizations and performance metrics to analyze the model's performance.

## Results

- The script outputs various performance metrics including accuracy, precision, recall, and F1-score.
- Visualizations are generated to show the training and validation accuracy and loss over epochs, as well as a confusion matrix to visualize the model's performance.

## Additional Notes

- This script assumes that the dataset is stored in the specified directory and follows a specific format. Make sure to adjust the code accordingly if your dataset differs.
- Experiment with different hyperparameters, preprocessing techniques, and model architectures to optimize performance for your specific task.

# Imagenet 19

## Overview

This project involves training a deep learning model using transfer learning with the VGG19 architecture pretrained on ImageNet. The objective is to classify images based on a specific criterion. The script preprocesses the images, trains the model, and evaluates its performance using various metrics.

## Dataset

- **Data Source**: The dataset used in this project is stored in the `/content/drive/MyDrive/data` directory.
- **Data Format**: The dataset consists of images in various formats, labeled based on certain criteria.

## Script Overview

- **Preprocessing**: The script preprocesses the images using techniques such as resizing and enhancing sharpness. It then splits each image into smaller pieces for training.
- **Model Training**: The script utilizes transfer learning with the VGG19 architecture pretrained on ImageNet. It adds custom dense layers on top of the VGG19 base and trains the model on the preprocessed image data.
- **Model Evaluation**: After training, the script evaluates the trained model using a separate test dataset and computes various performance metrics such as accuracy, precision, recall, and F1-score.
- **Results Visualization**: The script generates visualizations of training accuracy and loss over epochs, as well as a confusion matrix to visualize the model's performance.

## Usage

1. **Prepare Data**: Place your dataset in the `/content/drive/MyDrive/data` directory.
2. **Review Results**: After running the script, review the generated visualizations and performance metrics to analyze the model's performance.

## Results

- The script outputs various performance metrics including accuracy, precision, recall, and F1-score.
- Visualizations are generated to show the training accuracy and loss over epochs, as well as a confusion matrix to visualize the model's performance.
- Additionally, the Receiver Operating Characteristic (ROC) curve is plotted to assess the model's predictive performance.

## Additional Notes

- This script assumes that the dataset is stored in the specified directory and follows a specific format. Make sure to adjust the code accordingly if your dataset differs.
- Experiment with different hyperparameters, preprocessing techniques, and model architectures to optimize performance for your specific task.

# InceptionV3

## Overview

This script involves training a deep learning model using transfer learning with the InceptionV3 architecture pretrained on ImageNet. The objective is to classify images based on a specific criterion. The script preprocesses the images, trains the model, and evaluates its performance using various metrics.

## Setup

1. **Environment**: The script is designed to run in a Google Colab environment. Ensure that you have access to a GPU runtime.
2. **Dataset**: The dataset should be stored in the `/content/drive/MyDrive/data` directory.

## Script Overview

- **Preprocessing**: The script preprocesses the images by resizing them to (512, 512) and enhancing their sharpness. It then splits each image into smaller pieces for training.
- **Model Training**: The script utilizes transfer learning with the InceptionV3 architecture pretrained on ImageNet. It adds custom dense layers on top of the InceptionV3 base and trains the model on the preprocessed image data.
- **Model Evaluation**: After training, the script evaluates the trained model using a separate test dataset and computes various performance metrics such as accuracy, precision, recall, and F1-score.
- **Results Visualization**: Visualizations are generated to show the training accuracy and loss over epochs, as well as a confusion matrix to visualize the model's performance. Additionally, the Receiver Operating Characteristic (ROC) curve is plotted to assess the model's predictive performance.

## Usage

1. **Run the Script**: Execute the script in a Google Colab environment.
2. **Review Results**: After running the script, review the generated visualizations and performance metrics to analyze the model's performance.

## Results

- The script outputs various performance metrics including accuracy, precision, recall, and F1-score.
- Visualizations are generated to show the training accuracy and loss over epochs, as well as a confusion matrix to visualize the model's performance.
- Additionally, the Receiver Operating Characteristic (ROC) curve is plotted to assess the model's predictive performance.

## Additional Notes

- Ensure that the dataset is stored in the specified directory and follows a specific format. Adjust the code accordingly if your dataset differs.
- Experiment with different hyperparameters, preprocessing techniques, and model architectures to optimize performance for your specific task.

Here's the README for the provided script:

# ResNet50

## Overview

This script involves training a deep learning model using transfer learning with the ResNet50 architecture pretrained on ImageNet. The objective is to classify images based on a specific criterion. The script preprocesses the images, trains the model, and evaluates its performance using various metrics.

## Setup

1. **Environment**: The script is designed to run in a Google Colab environment. Ensure that you have access to a GPU runtime.
2. **Dataset**: The dataset should be stored in the `/content/drive/MyDrive/data` directory.

## Script Overview

- **Preprocessing**: The script preprocesses the images by resizing them to (512, 512) and enhancing their sharpness. It then splits each image into smaller pieces for training.
- **Model Training**: The script utilizes transfer learning with the ResNet50 architecture pretrained on ImageNet. It adds custom dense layers on top of the ResNet50 base and trains the model on the preprocessed image data.
- **Model Evaluation**: After training, the script evaluates the trained model using a separate test dataset and computes various performance metrics such as accuracy, precision, recall, and F1-score.
- **Results Visualization**: Visualizations are generated to show the training accuracy and loss over epochs, as well as a confusion matrix to visualize the model's performance. Additionally, the Receiver Operating Characteristic (ROC) curve is plotted to assess the model's predictive performance.

## Usage

1. **Run the Script**: Execute the script in a Google Colab environment.
2. **Review Results**: After running the script, review the generated visualizations and performance metrics to analyze the model's performance.

## Results

- The script outputs various performance metrics including accuracy, precision, recall, and F1-score.
- Visualizations are generated to show the training accuracy and loss over epochs, as well as a confusion matrix to visualize the model's performance.
- Additionally, the Receiver Operating Characteristic (ROC) curve is plotted to assess the model's predictive performance.

## Additional Notes

- Ensure that the dataset is stored in the specified directory and follows a specific format. Adjust the code accordingly if your dataset differs.
- Experiment with different hyperparameters, preprocessing techniques, and model architectures to optimize performance for your specific task.

# Convolutional Neural Network (CNN)

## Overview

This script involves training a Convolutional Neural Network (CNN) model for image classification using the Keras framework with TensorFlow backend. The objective is to classify images based on a specific criterion. The script preprocesses the images, trains the CNN model, and evaluates its performance using various metrics.

## Setup

1. **Environment**: The script is designed to run in a Google Colab environment. Ensure that you have access to a GPU runtime.
2. **Dataset**: The dataset should be stored in the `/content/drive/MyDrive/data` directory.

## Script Overview

- **Preprocessing**: The script preprocesses the images by resizing them to (256, 256) and enhancing their sharpness. It then splits each image into smaller pieces for training.
- **Model Training**: The script defines a CNN model architecture using the Sequential API from Keras. It comprises convolutional layers, max-pooling layers, dropout layers, and dense layers. The model is trained on the preprocessed image data.
- **Model Evaluation**: After training, the script evaluates the trained model using a separate test dataset and computes various performance metrics such as accuracy, precision, recall, and F1-score. It also generates a confusion matrix to visualize the model's performance.
- **Results Visualization**: Visualizations are generated to show the training accuracy and loss over epochs.

## Usage

1. **Run the Script**: Execute the script in a Google Colab environment.
2. **Review Results**: After running the script, review the generated visualizations and performance metrics to analyze the model's performance.

## Results

- The script outputs various performance metrics including accuracy, precision, recall, and F1-score.
- A confusion matrix is generated to visualize the model's performance on the test dataset.
- Visualizations are generated to show the training accuracy and loss over epochs.

## Additional Notes

- Ensure that the dataset is stored in the specified directory and follows a specific format. Adjust the code accordingly if your dataset differs.
- Experiment with different hyperparameters, preprocessing techniques, and model architectures to optimize performance for your specific task.

# Vision Transformer (ViT)

## Overview

This script implements a Vision Transformer (ViT) model for image classification using TensorFlow and Keras. The Vision Transformer is a transformer-based architecture originally designed for natural language processing tasks but adapted for computer vision tasks with impressive results. This script specifically focuses on classifying medical images based on a given criterion.

## Setup

1. **Environment**: The script is designed to run in a Google Colab environment with access to a GPU runtime for faster training.
2. **Dataset**: Ensure that your medical image dataset is stored in the `/content/drive/MyDrive/data` directory.

## Script Overview

- **Data Preprocessing**: The script reads and preprocesses the medical images, including resizing them to (512, 512) and enhancing their sharpness using ImageEnhance from PIL. It splits each image into smaller pieces for training.
- **Model Architecture**: The script defines a Vision Transformer (ViT) model architecture using TensorFlow and Keras. The model consists of multiple Transformer blocks, each containing multi-head self-attention mechanisms and feed-forward neural networks.
- **Model Training**: The ViT model is trained on the preprocessed image data using the AdamW optimizer and sparse categorical cross-entropy loss function. The training progress is monitored, and performance metrics are recorded.
- **Model Evaluation**: After training, the script evaluates the trained ViT model on a separate test dataset. It computes various performance metrics such as accuracy, precision, recall, and F1-score. Additionally, it generates a confusion matrix to visualize the model's performance.
- **Results Visualization**: Visualizations are generated to illustrate the training accuracy and loss over epochs.

## Usage

1. **Run the Script**: Execute the script in a Google Colab environment with GPU runtime enabled.
2. **Review Results**: After running the script, review the generated visualizations and performance metrics to analyze the ViT model's performance on your medical image classification task.

## Results

- The script outputs various performance metrics, including accuracy, precision, recall, and F1-score, for evaluating the ViT model.
- A confusion matrix is generated to provide a visual representation of the model's performance on the test dataset.
- Visualizations depicting the training accuracy and loss over epochs are provided to track the training progress.

## Additional Notes

- Ensure that your medical image dataset is stored in the specified directory and follows the required format. Adjust the code accordingly if your dataset structure differs.
- Experiment with different hyperparameters, preprocessing techniques, and ViT model configurations to optimize performance for your specific medical image classification task.

# Voting Classifier

## Overview

This script performs the evaluation of various machine learning classifiers on a dataset using features extracted from medical images. The classifiers considered include Logistic Regression, k-Nearest Neighbors (kNN), Random Forest, Decision Tree, and ensemble methods (Voting Classifier). The objective is to classify medical images into appropriate categories based on the extracted features.

## Setup

1. **Environment**: This script is designed to run in a Google Colab environment. Make sure to mount your Google Drive to access the dataset stored there.
2. **Dataset**: The dataset containing extracted features should be stored in a CSV file named `feature_100_true.csv` located in the `/content/drive/MyDrive/` directory.

## Script Overview

- **Data Loading and Preprocessing**: The script reads the dataset from the CSV file and preprocesses it by separating features and labels.
- **Data Visualization**: Visualizations such as scatter plots and correlation heatmaps are generated to explore the relationship between different features and labels.
- **Model Training and Evaluation**: Various classifiers are trained on the dataset, and their performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. The evaluated classifiers include Logistic Regression, k-Nearest Neighbors, Random Forest, Decision Tree, and ensemble methods (Voting Classifier).

## Usage

1. **Run the Script**: Execute the script in a Google Colab environment.
2. **Review Results**: After running the script, review the performance metrics and visualizations generated to analyze the effectiveness of different machine learning classifiers on the medical image classification task.

## Results

- The script provides the accuracy scores of each classifier on the test dataset.
- Detailed classification reports are generated for each classifier, including precision, recall, F1-score, and support for each class.
- Confusion matrices are displayed to visualize the classification results and identify any misclassifications made by the classifiers.

## Additional Notes

- Experiment with different classifiers and hyperparameters to optimize performance for your specific medical image classification task.
- Adjust the code as needed to accommodate different dataset formats or preprocessing requirements.

# Feature extraction

## Overview

This script is designed for image classification tasks using various machine learning models and techniques. Its primary focus is on feature extraction from the images, which are then used for training different models and evaluating their performance.

## Setup

1. **Environment**: The script is intended to run in a Google Colab environment with access to a GPU for faster processing.
2. **Dataset**: Ensure that your image dataset is stored in the `/content/drive/MyDrive/datad` directory.

## Feature Extraction

- **Pen Pressure**: The script calculates the pen pressure feature by analyzing the intensity and variance of pixels in the image.
- **Line Spacing**: Line spacing is computed by examining the vertical projection profile of the image and detecting spaces between lines of text.
- **Slant Angle**: The slant angle of the text in the image is determined by analyzing the tilt of characters using image processing techniques.
- **Baseline Angle**: Baseline angle extraction involves identifying the orientation of the text baseline in the image.
- **Imagenet weights**: Weights from last second layer of trained imagenet 16 model.

## Usage

1. **Run the Script**: Execute the script in a Google Colab environment with GPU runtime enabled.
2. **Review Results**: After running the script, review the generated parametes in csv file
