\# Skin Lesion Classification using GAN-based Synthetic Images



\## Overview

This project focuses on automated skin lesion classification using deep learning.

To address limited medical image availability and class imbalance, synthetic

skin lesion images were generated using a Generative Adversarial Network (GAN).



\## Project Pipeline

GAN → Synthetic Images → CNN Classifier → Web Application



\## Technologies Used

\- Python

\- TensorFlow / Keras

\- GAN (Generator \& Discriminator)

\- CNN

\- Streamlit (Frontend)

\- Flask (Backend)



\## Synthetic Data Generation

A GAN was trained offline to generate synthetic skin lesion images.

These images were used to augment the training dataset for improved

classification performance.



Due to computational and storage constraints, GAN training scripts

and pretrained weights are not included in this repository.



\## Repository Structure

\- `gan/` – GAN architecture documentation

\- `notebooks/` – CNN training notebooks

\- `sample\_synthetic\_images/` – Example generated images

\- `app.py` – Streamlit application

\- `flask\_app.py` – Backend API



\## How to Run

```bash

pip install -r requirements.txt

streamlit run app.py



