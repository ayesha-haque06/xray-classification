## 🚀 Features
* **Deep Learning Classification:** Utilizes a fine-tuned VGG16 convolutional neural network architecture.
* **Interactive UI:** Clean, user-friendly web interface built with Streamlit for seamless image uploads.
* **Real-Time Predictions:** Instantly outputs the predicted medical class and confidence score.
* **Local Execution:** Runs completely offline on a local development server for privacy and ease of testing.

---

## 🛠️ Tech Stack
* **Python** (Core language)
* **TensorFlow / Keras** (Deep learning model framework)
* **Streamlit** (Interactive web application framework)
* **Pillow & NumPy** (Image processing and numerical manipulation)

---

## ⚠️ Model Limitations & Scope
* **Binary Classification:** The model is strictly trained on two specific classes (`NORMAL` and `COVID19`). 
* **Out-of-Distribution Inputs:** Providing alternative pathologies (such as Pneumonia or other lung conditions) that the model was not trained on can lead to high-confidence misclassifications, as the network is forced to choose between its known output labels.
