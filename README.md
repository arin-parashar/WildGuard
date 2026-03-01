# WildGuard
A human and wildlife co-occurrence alert and alarm system.
WildGuard detects situations where humans and wild animals appear together in the same scene.
It uses:

MegaDetector (object detection)

SpeciesNet (animal classification)

ESP32 serial communication for buzzer and light alerts


Project Structure
WildGuard/
│
├── src/
├── models/
│   └── md/
│       └── md_v5a.0.0.pt
│
├── external/
│   └── MegaDetector/
│
├── requirements.txt
└── README.md

Installation

1️⃣ Clone Repository
git clone <your-repo-url>
cd WildGuard
2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Install PyTorch
If using GPU (CUDA 11.8):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
If CPU only:
pip install torch torchvision


Model Setup
🔹 MegaDetector
Go to:
https://lila.science/public/megadetector/md_v5a.0.0.pt
Download the file
Place it in:
models/md/md_v5a.0.0.pt
🔹 MegaDetector Repository
Inside project:
cd external
git clone https://github.com/microsoft/CameraTraps.git MegaDetector


🔹 SpeciesNet
Install Kaggle CLI:
pip install kaggle
Download Kaggle API token from:
https://www.kaggle.com/account
Place kaggle.json inside:
C:\Users\<your-username>\.kaggle\
SpeciesNet model will download automatically on first run.


Running the Project
From project root.
python src/main.py


ESP32 Setup (Optional)
If using hardware buzzer:
Update COM port inside code:
serial.Serial("COM3", 115200)
Change "COM3" to your device port.
