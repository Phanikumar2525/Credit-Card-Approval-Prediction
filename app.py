from flask import Flask, render_template, request
import pickle
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:

        # ---------------- Read Form Values ----------------

        gender = int(request.form["gender"])
        car = int(request.form["car"])
        realty = int(request.form["realty"])

        children = float(request.form["children"])
        income = float(request.form["income"])

        income_type = int(request.form["income_type"])
        education = int(request.form["education"])
        family = int(request.form["family_status"])
        housing = int(request.form["housing"])

        age = float(request.form["age"])
        employment = float(request.form["employment"])
        family_members = float(request.form["family_members"])

        # ---------------- Input Validation ----------------

        if income <= 0:
            return render_template(
                "result.html",
                prediction_text="Invalid Input",
                error="Annual Income must be greater than 0."
            )

        if age < 18:
            return render_template(
                "result.html",
                prediction_text="Invalid Input",
                error="Applicant age must be at least 18 years."
            )

        if employment > age:
            return render_template(
                "result.html",
                prediction_text="Invalid Input",
                error="Employment years cannot exceed age."
            )

        if family_members < children:
            return render_template(
                "result.html",
                prediction_text="Invalid Input",
                error="Family members cannot be less than children."
            )

        # ---------------- Convert Years into Days ----------------

        days_birth = -(age * 365)
        days_employed = -(employment * 365)

        # ---------------- Create DataFrame ----------------

        features = pd.DataFrame([[

            gender,
            car,
            realty,
            children,
            income,
            income_type,
            education,
            family,
            housing,
            days_birth,
            days_employed,
            family_members

        ]], columns=[

            "CODE_GENDER",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "CNT_CHILDREN",
            "AMT_INCOME_TOTAL",
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "CNT_FAM_MEMBERS"

        ])

        # ---------------- Prediction ----------------

        prediction = model.predict(features)

        probability = model.predict_proba(features)

        confidence = round(max(probability[0]) * 100, 2)

        if prediction[0] == 1:
            result = "Credit Card Approved"
        else:
            result = "Credit Card Rejected"

        # ---------------- Dictionaries ----------------

        income_type_names = {
            0: "Commercial Associate",
            1: "Pensioner",
            2: "State Servant",
            3: "Student",
            4: "Working"
        }

        education_names = {
            0: "Academic Degree",
            1: "Higher Education",
            2: "Incomplete Higher",
            3: "Lower Secondary",
            4: "Secondary / Secondary Special"
        }

        family_names = {
            0: "Civil Marriage",
            1: "Married",
            2: "Separated",
            3: "Single / Not Married",
            4: "Widow"
        }

        housing_names = {
            0: "Co-op Apartment",
            1: "House / Apartment",
            2: "Municipal Apartment",
            3: "Office Apartment",
            4: "Rented Apartment",
            5: "With Parents"
        }

        # ---------------- Applicant Summary ----------------

        summary = {

            "Gender": "Male" if gender == 1 else "Female",

            "Own Car": "Yes" if car == 1 else "No",

            "Own Real Estate": "Yes" if realty == 1 else "No",

            "Annual Income": f"₹{income:,.0f}",

            "Income Type": income_type_names[income_type],

            "Education": education_names[education],

            "Family Status": family_names[family],

            "Housing Type": housing_names[housing],

            "Children": int(children),

            "Family Members": int(family_members),

            "Age": f"{int(age)} Years",

            "Employment": f"{int(employment)} Years"

        }

        # ---------------- Prediction Time ----------------

        prediction_time = datetime.now().strftime("%d-%b-%Y %I:%M %p")

        return render_template(

            "result.html",

            prediction_text=result,

            confidence=confidence,

            summary=summary,

            prediction_time=prediction_time

        )

    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    app.run(debug=True)