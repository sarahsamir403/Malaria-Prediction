# Malaria Detection Model

This repository contains code for building and training a machine learning model to classify malaria cell images. The goal is to automatically distinguish between parasitized and uninfected cells from microscopic images.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

The project uses a convolutional neural network (CNN) to detect malaria in blood cell images. The model is trained on the "Malaria Cell Images Dataset," which includes labeled images of parasitized and uninfected cells.

## Dataset

The dataset used in this project is available on [Kaggle: Malaria Cell Images Dataset](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria). The dataset contains over 27,000 images with two categories:
- Parasitized: Infected with malaria.
- Uninfected: Healthy cells.

Make sure to download and extract the dataset before running the model.

## Model Architecture

The model leverages the following structure:
- **Input layer**: Images resized to 64x64 pixels and normalized.
- **Convolutional layers**: To extract spatial features from images.
- **Pooling layers**: To reduce spatial dimensions.
- **Fully connected layers**: For classification.

The model was trained using the `TensorFlow` library with the following hyperparameters:
- Optimizer: `Adam`
- Loss function: `categorical_crossentropy`
- Metrics: `accuracy`

# Malaria Classification App

This is a simple web-based application built using [Streamlit](https://streamlit.io/) to classify whether a blood smear image is infected with malaria or not. The model used is a Convolutional Neural Network (CNN) trained to detect malaria from images.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Model Information](#model-information)
- [Notes](#notes)

## Project Overview
The app allows users to upload an image of a blood smear and predicts whether it is infected with malaria or uninfected using a pre-trained model.

### Features
- Upload an image (in `.jpg`, `.png`, or `.jpeg` format).
- The image is pre-processed and passed to a deep learning model.
- Displays the uploaded image and prediction result (Infected or Uninfected).
- User-friendly interface using Streamlit.

## Setup Instructions
### Prerequisites
- Python 3.x
- TensorFlow
- OpenCV
- Streamlit
- Pillow


