# skin-lesion-classification-GAN
Skin disease classification using GAN-based synthetic images
# Skin Lesion Classification using GAN

## Overview
This project focuses on classifying skin diseases using deep learning.
To overcome limited medical image data, a **Generative Adversarial Network (GAN)**
is used to generate synthetic skin lesion images.

## Technologies Used
- Python
- GAN (Generator + Discriminator)
- Streamlit (Frontend)
- Flask (Backend API)
- Roboflow Inference API

## Features
- Synthetic image generation using GAN
- Skin lesion classification
- Web-based UI using Streamlit
- REST API using Flask

## Project Architecture
GAN → Synthetic Images → Classifier → Web UI

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
