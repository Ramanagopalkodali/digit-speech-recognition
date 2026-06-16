# Speaker-Independent Spoken Digit Recognition from Scratch

In this reporitpty the precording got cut so , I will keep the link of complete files so use that download that and keep in same folder and then run the maim.py to get accrate results , if you do this directly we get wrong results -link: https://drive.google.com/drive/folders/1f5eGtxSqxV8bJ8d7e9Oxy8JFPg2wH-Be?usp=sharing

This repository contains a complete, from-scratch implementation of a spoken digit recognition system using probabilistic graphical and mixture models. 
The system processes raw audio data, extracts acoustic features, and classifies spoken digits (0-9) across unseen speakers using custom-built statistical classifiers.

## 🌟 Key Innovations & Engineering Highlights
* **Zero-Library Algorithmic Implementation:** Built the Expectation-Maximization (EM) algorithm for Gaussian Mixture Models (GMM) and sequence evaluation/training for Hidden Markov Models (HMM) strictly using core mathematical and linear algebra operations (NumPy/SciPy math utilities)—no `scikit-learn` or `hmmlearn`.
* **K-Means Accelerated GMM Initialization:** To circumvent the local optima vulnerabilities of the EM algorithm, a custom K-Means clustering algorithm was engineered to seed initial cluster centroids, significantly accelerating log-likelihood convergence.
* **Speaker-Independent Rigor:** Designed the validation pipeline to strictly isolate speakers 1–5 for training/validation, reserving speaker 6 purely for test inference, ensuring true generalization evaluation.

## 📊 Performance & Metrics
* **Top Test Accuracy:** `81.3%` achieved via the GMM framework with K-Means initialization.
* **Evaluation Framework:** Full script implementation to generate cross-speaker validation configurations, comprehensive confusion matrices, and isolated per-digit precision/recall arrays.

## How to run the repository:
The rep

## 🛠️ Repository Structure
```text
├── src/
│   ├── feature_extraction.py  # MFCC feature processing pipeline
│   ├── gmm.py                 # Custom GMM implementation (EM & K-Means)
│   ├── hmm.py                 # Custom Discrete HMM implementation
│   └── inference.py           # Single-file audio inference engine
├── data/                      # Dataset organization (Train: Speakers 1-5 | Test: Speaker 6)
├── Report.pdf                 # Detailed mathematical derivations & metrics analysis
├── Details.txt                # Student metadata
└── README.md                  # Project documentation



