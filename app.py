from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# ----------------- Load models -----------------
MODEL_FILES = {
    "Logistic Regression": "models/LogisticRegression_model.pkl",
    "Decision Tree": "models/DecisionTreeModel.pkl",
    "Random Forest": "models/RandomForestModel.pkl",
    "KNN": "models/knn_model.pkl",
    "Bagging (Decision Tree)": "models/Bagging_DecisionTree_model.pkl",
    "SVC": "models/SVC.pkl"
}

MODELS = {}
for name, path in MODEL_FILES.items():
    try:
        MODELS[name] = joblib.load(path)
        print("Loaded:", name)
    except Exception as e:
        MODELS[name] = None
        print(f"Could not load {name}: {e}")

# Load scaler for logistic regression
SCALER = None
scaler_path = "models/scaler.pkl"
if os.path.exists(scaler_path):
    SCALER = joblib.load(scaler_path)
    print("Loaded scaler")

# ----------------- Features -----------------
FEATURE_ORDER = [
    "Age","CityTier","DurationOfPitch","NumberOfPersonVisiting","NumberOfFollowups",
    "ProductPitched","PreferredPropertyStar","NumberOfTrips","Passport",
    "PitchSatisfactionScore","OwnCar","NumberOfChildrenVisiting","Designation",
    "MonthlyIncome","TypeofContact_SelfEnquiry","Occupation_LargeBusiness",
    "Occupation_Salaried","Occupation_SmallBusiness","Gender_Male",
    "MaritalStatus_Married","MaritalStatus_Single","MaritalStatus_Unmarried"
]

VALIDATION = {
    "Age": (1, 100),
    "CityTier": (1, 3),
    "DurationOfPitch": (1, 120),
    "NumberOfPersonVisiting": (1, 10),
    "NumberOfFollowups": (0, 20),
    "ProductPitched": (1, 10),
    "PreferredPropertyStar": (1, 5),
    "NumberOfTrips": (0, 50),
    "Passport": (0, 1),
    "PitchSatisfactionScore": (1, 5),
    "OwnCar": (0, 1),
    "NumberOfChildrenVisiting": (0, 10),
    "Designation": (0, 20),
    "MonthlyIncome": (0, 10000000),
    "TypeofContact_SelfEnquiry": (0,1),
    "Occupation_LargeBusiness": (0,1),
    "Occupation_Salaried": (0,1),
    "Occupation_SmallBusiness": (0,1),
    "Gender_Male": (0,1),
    "MaritalStatus_Married": (0,1),
    "MaritalStatus_Single": (0,1),
    "MaritalStatus_Unmarried": (0,1)
}

def safe_float(x, default=0.0):
    try: return float(x)
    except: return default

def clamp(v, lo, hi): return max(lo, min(hi, v))

@app.route("/", methods=["GET","POST"])
def index():
    errors = []
    result = None
    best_model = None
    user_input = {f:"" for f in FEATURE_ORDER}

    if request.method == "POST":
        # collect and validate input
        for f in FEATURE_ORDER:
            raw = request.form.get(f,"").strip()
            val = safe_float(raw)
            if f in VALIDATION:
                lo, hi = VALIDATION[f]
                if val<lo or val>hi:
                    val=clamp(val,lo,hi)
                    errors.append(f"{f} adjusted to {val} (allowed {lo}-{hi})")
            user_input[f]=str(val)

        X=np.array([float(user_input[f]) for f in FEATURE_ORDER]).reshape(1,-1)

        # predict for all models
        results={}
        best_prob=-1.0
        for name, model in MODELS.items():
            if model is None:
                results[name]={"error":"Model not available"}
                continue
            X_used=X.copy()
            if name=="Logistic Regression" and SCALER:
                X_used=SCALER.transform(X_used)
            try:
                if hasattr(model,"predict_proba"):
                    prob=float(model.predict_proba(X_used)[0][1])
                elif hasattr(model,"decision_function"):
                    df=float(model.decision_function(X_used)[0])
                    prob=1.0/(1.0+np.exp(-df))
                else:
                    pred=int(model.predict(X_used)[0])
                    prob=float(pred)
                pred_class=int(prob>=0.5)
                results[name]={"prediction":pred_class,"probability":round(prob,4)}
                if prob>best_prob:
                    best_prob=prob
                    best_model=name
            except Exception as e:
                results[name]={"error":str(e)}
        result=results

    return render_template("index.html",
                           result=result,
                           best_model=best_model,
                           user_input=user_input,
                           errors=errors)

if __name__=="__main__":
    app.run(debug=True)
