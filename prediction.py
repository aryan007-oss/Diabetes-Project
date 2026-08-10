from joblib import load
import pandas as pd

loaded_model = load("D:\project_\model_dir\diabetes_model.joblib")

data_dict = pd.DataFrame(
    {
        #"Pregnancies":[6],
        #"Glucose":[148],
        #"BloodPressure":[72],         {YES✔️}
        #"SkinThickness":[35],
        #"Insulin":[155],
        #"BMI":[33.6],
        #"DiabetesPedigreeFunction":[0.627],
        #"Age":[50]
        
        #"Pregnancies":[1],
        #"Glucose":[85],
        #"BloodPressure":[66],        {NO❌}
        #"SkinThickness":[29],
        #"Insulin":[155],
        #"BMI":[26.6],
        #"DiabetesPedigreeFunction":[0.351],
        #"Age":[31]
        
        "Pregnancies":[8],
        "Glucose":[183],
        "BloodPressure":[64],
        "SkinThickness":[29.15342],
        "Insulin":[155.548223],
        "BMI":[23.3],
        "DiabetesPedigreeFunction":[0.672],
        "Age":[32]
    }
)
# *Task
#The Control statement is used to check wheather the patient is DIABETIC or NOT
if loaded_model.predict(data_dict) == 1:
    print("The Patient is Diabetic..💀💀!!!")
else:
    print("The Patient has no syntom of diabetic..🫡!!")
#print(loaded_model.predict(data_dict))