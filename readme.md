## Next Word Prediction using LSTM

This project is an **AI-based Next Word Predictor** built using **TensorFlow (LSTM)** and **Streamlit**.
It predicts the next possible word in a given sentence based on a trained language model.
The model was trained in **Google Colab**, tested locally, and can be deployed to any cloud platform.

---

### Features

✅ Trained using LSTM Neural Network
✅ Tokenizer and Model Architecture stored as JSON
✅ Weight file saved separately (`model.weights.h5`)
✅ Interactive Streamlit Web App
✅ Ready for Local or Cloud Deployment

---

### Project Structure

```
next-word-prediction-lstm/
│
├── data/                         # (Optional) Dataset used for training
├── notebooks/
│   └── train_model.ipynb          # Google Colab training notebook
├── models/
│   ├── model_architecture.json    # Model structure
│   ├── model.weights.h5           # Trained model weights
│   └── tokenizer.json             # Tokenizer data
├── app/
│   └── streamlit_app.py           # Streamlit interface
│
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview
└── .gitignore
```

---

### ech Stack

| Tool                   | Purpose                        |
| ---------------------- | ------------------------------ |
| **TensorFlow / Keras** | Build and train the LSTM model |
| **Google Colab**       | Training environment           |
| **Streamlit**          | Web app deployment             |
| **Python**             | Core programming language      |

---

### Training (Google Colab)

1. Open `notebooks/train_model.ipynb` in **Google Colab**
2. Upload your dataset
3. Run preprocessing, tokenization, and model training cells
4. Save outputs:

   * `model_architecture.json`
   * `model.weights.h5`
   * `tokenizer.json`
5. Download these 3 files to your local machine.

---

### 💻 Local Testing (Streamlit)

1. Clone this repository:

```bash
git clone https://github.com/your-username/next-word-prediction-lstm.git
cd next-word-prediction-lstm
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

4. Enter any sentence in the input box and get the next word prediction.

---

### ☁️ Deployment Options

You can deploy this app easily on:

* **Streamlit Cloud** → (Free & easiest)
* **Render** or **Hugging Face Spaces**
* **AWS / Google Cloud / Azure**

---

### 🧾 Example

**Input:**

> I love to

**Output:**

> cook 🍳

---

### Requirements

Add this in your `requirements.txt` file:

```
tensorflow==2.15.0
streamlit
numpy
```

---

### 🧑‍💻 Author

**Pervez Hasan**
AI & Machine Learning Engineer | Founder, AsuX AI
🔗 [pervezhasan.com](https://pervezhasan.com)
