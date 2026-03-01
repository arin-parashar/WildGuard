# WildGuard
A human–wildlife co-occurrence alert and alarm system.

WildGuard detects situations where humans and wild animals appear together in the same scene and triggers alerts accordingly.

It uses:

MegaDetector for object detection

SpeciesNet for animal species classification

ESP32 serial communication for buzzer and light alerts

# Project Structure

WildGuard/

│

├── src/

├── models/

│ └── md/

│ └── md_v5a.0.0.pt

│

├── external/

│ └── MegaDetector/

│

├── requirements.txt

└── README.md


# Installation

Step 1 – Clone Repository

git clone <your-repository-url>
cd WildGuard

Step 2 – Create Virtual Environment

python -m venv venv
venv\Scripts\activate

Step 3 – Install Dependencies

pip install -r requirements.txt

Step 4 – Install PyTorch

If using GPU (CUDA 11.8):

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

If using CPU only:

pip install torch torchvision

# Model Setup

# MegaDetector Model

Download MegaDetector v5a from:
https://lila.science/public/megadetector/md_v5a.0.0.pt

Create this folder if it does not exist:
models/md/

Place the downloaded file inside:
models/md/md_v5a.0.0.pt

MegaDetector Repository

Inside the project directory:

cd external
git clone https://github.com/microsoft/CameraTraps.git
 MegaDetector

# SpeciesNet Setup

Step 1 – Install Kaggle CLI

pip install kaggle

Step 2 – Download Kaggle API Token

Go to:
https://www.kaggle.com/account

Click “Create New API Token”

Download kaggle.json

Place it inside:
C:\Users<your-username>.kaggle\

SpeciesNet will automatically download the required model on first run.

Running the Project

From the project root directory:

python src/main.py

#ESP32 Setup (Optional)

If using the hardware buzzer/light system, update the COM port inside the code:

serial.Serial("COM3", 115200)

Replace "COM3" with your ESP32 device port.

# Features

Human detection

Animal detection

Species classification

Real-time threat detection

GUI-based monitoring interface

ESP32 serial alert triggering
