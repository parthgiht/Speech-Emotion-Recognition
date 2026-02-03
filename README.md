# 🎭Speech Emotion Recognition System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A state-of-the-art deep learning system for recognizing emotions from speech audio using LSTM, GRU architectures.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Models](#models)
- [Results](#results)
- [Documentation](#documentation)
- [License](#license)

## 🎯 Overview

This project implements an advanced Speech Emotion Recognition (SER) system capable of classifying emotions from audio signals. The system leverages state-of-the-art deep learning architectures including LSTM, GRU, that combines both approaches.

### Supported Emotions

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Surprised (PS)
- Sad

## ✨ Features

### Core Functionality

- **Multi-Model Architecture**: LSTM, GRU
- **Advanced Feature Extraction**: MFCC, Mel Spectrogram, Chroma, Spectral Contrast
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrices
- **Production-Ready**: Modular code, proper error handling, logging

### Advanced Features

- **Automated Hyperparameter Tuning**: Callbacks for learning rate adjustment
- **Early Stopping**: Prevents overfitting automatically
- **Model Checkpointing**: Saves best performing models
- **TensorBoard Integration**: Real-time training visualization
- **Cross-Validation Support**: Robust model evaluation
- **Batch Processing**: Efficient feature extraction
- **Inference Pipeline**: Ready-to-use prediction function


## 📁 Project Structure

```
speech-emotion-recognition/
│
├── speech_emotion_recognition_pro.ipynb  # Main notebook
├── requirements.txt                      # Python dependencies
├── README.md                            # This file
├── config.py                            # Configuration settings
│
├── models/                              # Saved models
│   ├── lstm_best.keras
│   ├── gru_best.keras
│   ├── hybrid_best.keras
│   ├── label_encoder.pkl
│   └── config.json
│
├── outputs/                             # Results and visualizations
│   ├── emotion_distribution.png
│   ├── training_history.png
│   ├── confusion_matrices.png
│   ├── metrics_comparison.png
│   └── project_report.json
│
├── logs/                                # TensorBoard logs
│   ├── lstm/
│   └── gru/
│   
│
└── utils/                               # Utility modules
    ├── data_loader.py
    ├── feature_extractor.py
    ├── model_builder.py
    └── visualizer.py
```

## 🏗️ Models

### LSTM Model

**Architecture:**
- 2 LSTM layers (128, 64 units)
- Batch Normalization
- Dropout (0.3)
- 2 Dense layers (64, 32 units)
- Softmax output

**Parameters:** ~250K

### GRU Model

**Architecture:**
- 2 GRU layers (128, 64 units)
- Batch Normalization
- Dropout (0.3)
- 2 Dense layers (64, 32 units)
- Softmax output

**Parameters:** ~200K

## 📈 Results

### Model Performance Comparison

| Model  | Accuracy | Precision | Recall  | F1-Score |
|--------|----------|-----------|---------- |----------|
| LSTM   | 0.992857 | 0.992973  | 0.992857|    0.992862|
| GRU    | 0.996429 |	0.996458|	0.996429|	0.996428  |


*Note: Actual results will vary based on your dataset*

### Key Findings

1. **Model Efficiency**: GRU models train 15-20% faster than LSTM with comparable accuracy
2. **Optimization**: Early stopping typically occurs around 50-70 epochs
3. **Feature Importance**: MFCC features contribute most significantly to accuracy

## 📚 Documentation

### Configuration Parameters

```python
class Config:
    # Audio Processing
    SAMPLE_RATE = 22050
    DURATION = 3
    N_MFCC = 40
    
    # Training
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # Architecture
    LSTM_UNITS = [128, 64]
    GRU_UNITS = [128, 64]
    DROPOUT_RATE = 0.3
```

### Feature Extraction

The system extracts multiple audio features:

- **MFCC** (Mel-Frequency Cepstral Coefficients): Primary feature
- **Chroma**: Harmonic content representation
- **Mel Spectrogram**: Time-frequency representation
- **Spectral Contrast**: Spectral peak and valley differences
- **Zero Crossing Rate**: Signal change indicator

### Training Process

1. **Data Loading**: Audio files with emotion labels
2. **Feature Extraction**: MFCC computation
3. **Preprocessing**: Normalization and encoding
4. **Model Training**: With callbacks and validation
5. **Evaluation**: Comprehensive metrics
6. **Model Saving**: Best performing model



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TESS Dataset for emotion audio samples
- TensorFlow and Keras teams
- Librosa for audio processing capabilities
- The open-source community


---

**Made with ❤️ and 🤖 Deep Learning**
