# 🐾 Animal Skin Disease Classifier API

This project is a **FastAPI-based backend** that classifies common skin conditions in **dogs, cats, and cows** using pre-trained TensorFlow/Keras models.  
It also provides **medicine suggestions** for the predicted disease using **Gemini (via `med.py`)**.

---

## 🚀 Features
- Image upload support (`.jpg`, `.png`, `.jpeg`)
- Lazy model loading for efficient memory usage
- Supported animals:
  - 🐶 Dogs → Demodicosis, Dermatitis, Fungal Infection, Hypersensitivity, Ringworm, Healthy
  - 🐱 Cats → Flea Allergy, Ringworm, Scabies, Healthy
  - 🐄 Cows → Lumpy Skin, Healthy
- Normalizes inconsistent labels
- Fetches medicine suggestions using Gemini

---

## 📂 Project Structure
```plaintext
.
├── app.py              # Main FastAPI application
├── model_loader.py     # Handles lazy loading of Keras models
├── utils.py            # Utility functions (e.g., image preprocessing)
├── med.py              # Gemini API integration for medicine suggestions
├── Dog.keras           # Pre-trained dog model
├── cat.keras           # Pre-trained cat model
├── cow.keras           # Pre-trained cow model
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation


---

## ⚙️ Installation & Running

```bash
git clone https://github.com/ankitkr1375/animal_classifier.git
cd animal_classifier

python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt

uvicorn app:app --reload
