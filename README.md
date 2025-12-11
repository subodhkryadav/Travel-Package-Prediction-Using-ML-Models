# Travel-Package-Prediction-Using-ML-Models

## 📸 Screenshots

### 🖼️ Screenshot 1
![Screenshot 1](https://raw.githubusercontent.com/subodhkryadav/Travel-Package-Prediction-Using-ML-Models/main/Screenshot%202025-12-11%20080003.png)

### 🖼️ Screenshot 2
![Screenshot 2](https://raw.githubusercontent.com/subodhkryadav/Travel-Package-Prediction-Using-ML-Models/main/Screenshot%202025-12-11%20080225.png)


A machine-learning–powered web application built with **Flask**, designed to predict whether a customer will purchase a travel package.  
This project uses **7 ML models**, an interactive UI, and clean structured data for evaluation and deployment.

---

## 📂 Project Structure

```
Travel-Package-Prediction-Using-ML-Models/
│
├── app.py                           # Flask backend application
│
├── templates/
│   └── index.html                   # Frontend UI (HTML + Jinja)
│
├── static/
│   └── style.css                    # Styling for UI
│
├── models/                          # Serialized ML models (Joblib)
│   ├── LogisticRegression_model.pkl
│   ├── DecisionTreeModel.pkl
│   ├── RandomForestModel.pkl
│   ├── knn_model.pkl
│   ├── Bagging_DecisionTree_model.pkl
│   ├── SVC.pkl
│   └── scaler.pkl                   # StandardScaler for Logistic Regression
│
├── Travel.csv                       # Raw dataset
├── clean_data.csv                   # Cleaned & preprocessed dataset
│
└── Travel-Package-Prediction-Using-ML-Models.ipynb   # Jupyter Notebook (training, EDA, model building)
```

---

## 🧠 Machine Learning Models Used

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **KNN Classifier**
5. **Bagging Classifier (Decision Tree)**
6. **Support Vector Classifier (SVC)**
7. **Baseline Rule (if applicable)**

Each model generates:
- **Prediction (YES/NO)**
- **Confidence score (%)**

The application automatically highlights the **best model** based on highest probability.

---

## 🚀 Features

- Clean and modern UI built using **HTML + CSS + Jinja**
- User-friendly structured input form
- Validations & placeholder hints
- Probabilities visualized with progress bars
- Error-handling for missing/unavailable models
- Dynamic model comparison
- Full Flask-based backend

---

## ▶️ How to Run Locally

### **1. Clone the Repository**
```bash
git clone https://github.com/subodhkryadav/Travel-Package-Prediction-Using-ML-Models.git
cd Travel-Package-Prediction-Using-ML-Models
```

### **2. Install Dependencies**
```bash
install-s
```
```
Flask
numpy
pandas
scikit-learn
joblib
```

### **3. Run the Application**
```bash
python app.py
```

### **4. Open in Browser**
```
http://127.0.0.1:5000/
```

---

## 📝 Requirements File

```
Flask
numpy
pandas
scikit-learn
joblib
```

---

## 📊 Dataset Description

You will include:

- **Travel.csv** → Original dataset  
- **clean_data.csv** → Cleaned dataset after preprocessing  
- Notebook includes:
  - EDA  
  - Missing value treatment  
  - Encoding  
  - Model training  
  - Cross-validation  
  - Saving models with Joblib  

---


## 👨‍💻 Author

**Subodh Kumar Yadav**  
B.Tech CSE • 3rd Year  
Jagannath University, Jaipur

---

## ⭐ Contribute

If you'd like to enhance the project:
- Improve UI
- Add more models
- Add visual analytics
- Write unit tests

PRs are welcome!

---

## 📄 License

This project is open-source under the **MIT License**.
