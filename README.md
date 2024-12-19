# Arabic-Hand-Written-Classifier

Welcome to the Arabic Handwritten Classifier project! This repository contains all the resources, code, and documentation needed to train and evaluate a deep learning model for recognizing Arabic handwritten text. The project leverages modern machine learning techniques to classify Arabic alphabets, words, and paragraphs.

Project Overview

Arabic handwriting presents a unique challenge due to the cursive nature of the script and its variations depending on the position within a word. This project aims to:

Develop a robust neural network model to classify Arabic handwritten characters.

Support multi-level classification (isolated alphabets, words, and paragraphs).

Utilize a diverse dataset to ensure generalizability across different handwriting styles.

Key Features:

Dataset: Over 53,000 labeled images of Arabic characters.

Model: Convolutional Neural Network (CNN) built using TensorFlow and Keras.

Performance Optimization: Data augmentation, regularization, and hyperparameter tuning.

Applications: Handwriting recognition systems, educational tools, and digital archival of Arabic texts.

Dataset Description

The dataset consists of:

Isolated Alphabets: 65 classes representing all variations of Arabic letters (initial, medial, final, and isolated forms).

Words: A collection of 10 Arabic words that include all Arabic alphabets.

Paragraphs: Handwritten paragraphs contributed by 82 individuals to capture diverse writing styles.

Dataset Highlights:

Isolated Alphabets:

Total Samples: 53,199 images

Organized into 65 directories, each representing a unique class.

Words and Paragraphs:

Includes 10 words and paragraph-level data for model evaluation and testing.

Technical Details

Model Architecture

The Arabic Handwritten Classifier employs a CNN model designed for image classification tasks.

Input Layer: Processes grayscale images of handwritten text.

Convolutional Layers: Extract spatial features from the input images.

Pooling Layers: Reduce dimensionality while retaining essential features.

Dense Layers: Perform the final classification into 65 classes.

Activation Functions:

ReLU for hidden layers.

Softmax for the output layer to generate class probabilities.
