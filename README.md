# AI-BioScan: Respiratory Disease Detection from Cough & Voice

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

AI-BioScan is a deep learning project that analyzes **cough and voice recordings** to detect respiratory conditions such as **Asthma, Pneumonia, and Healthy**.  
It leverages **Mel-Spectrograms + CNNs** for audio classification and provides a **Streamlit web interface** for real-time predictions.

---

## Features
-  **Audio Preprocessing** (MFCC / Mel-Spectrograms)  
-  **CNN-based classifier** with data balancing  
-  **Training pipeline** with configurable hyperparameters (`config.yaml`)  
-  **Streamlit web app** for real-time disease prediction  
-  **Early stopping & regularization** to handle class imbalance  

---

##  Project Structure
```
AI-BioScan/
│── config.yaml           # Centralized config
│── requirements.txt      # Dependencies
│── README.md             # Project documentation
│── app_streamlit.py      # Streamlit web app
│── src/
│    ├── train.py         # Training script
│    ├── infer.py         # Inference script
│    ├── model.py         # Model architecture
│    ├── dataset.py       # Data loading & preprocessing
│    ├── audio_utils.py   # Audio feature extraction
│── models/               # Saved models (ignored in Git)
│── data/                 # Dataset (ignored in Git)
│── notebooks/            # Optional Jupyter experiments
```

---

##  Installation

```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/AI-BioScan.git
cd AI-BioScan

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Training the Model
Prepare your dataset in the structure:
```
data/
│── Asthma/
│   ├── asthma_01.wav
│   ├── asthma_02.wav
│── Pneumonia/
│   ├── pneu_01.wav
│── Healthy/
│   ├── healthy_01.wav
```

Run training:
```bash
python -m src.train --data_dir data --epochs 40 --batch_size 32
```

Models will be saved in the `models/` directory.

---

##  Inference
Run prediction on a sample audio file:
```bash
python -m src.infer --audio path/to/sample.wav
```

Output:
```
Predicted: Asthma (92% confidence)
```

---

##  Run Streamlit App
Start the web app:
```bash
streamlit run app_streamlit.py
```

Upload a `.wav` file → Get instant disease prediction.

---

##  Configuration
All hyperparameters and settings are controlled via **config.yaml**:
```yaml
sample_rate: 16000
duration_sec: 3.0
n_mfcc: 40
batch_size: 32
epochs: 20
learning_rate: 0.0005
classes:
  - Pneumonia
  - Healthy
  - Asthma
```

---

##  Results (Example)
| Class     | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Asthma    | 0.89      | 0.85   | 0.87     |
| Pneumonia | 0.93      | 0.95   | 0.94     |
| Healthy   | 0.91      | 0.90   | 0.90     |

---

##  Contributing
Pull requests and suggestions are welcome! For major changes, please open an issue first.

---

##  License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
