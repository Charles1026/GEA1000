import csv 
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lin

def pearson_correlation(X,Y):
    if len(X)==len(Y):
        Sum_xy = sum((X-X.mean())*(Y-Y.mean()))
        Sum_x_squared = sum((X-X.mean())**2)
        Sum_y_squared = sum((Y-Y.mean())**2)       
        corr = Sum_xy / np.sqrt(Sum_x_squared * Sum_y_squared)
    return corr

if __name__ == "__main__":
  Hours_Studied = []
  Attendance = []
  Sleep_Hours = []
  Midterm_Score = []
  Consultation_Sessions = []
  Physical_Activity = []
  Final_Score = []
  
  with open("./Timothy_TE69_ICQ3.csv", 'r') as file:
    reader = csv.reader(file)
    
    next(reader)
    for row in reader:
      Hours_Studied.append(float(row[0]))
      Attendance.append(float(row[1]))
      Sleep_Hours.append(float(row[2]))
      Midterm_Score.append(float(row[3]))
      Consultation_Sessions.append(int(row[4]))
      Physical_Activity.append(float(row[5]))
      Final_Score.append(float(row[6]))
      
  Hours_Studied = np.asarray(Hours_Studied)
  Attendance = np.asarray(Attendance)
  Sleep_Hours = np.asarray(Sleep_Hours)
  Midterm_Score = np.asarray(Midterm_Score)
  Consultation_Sessions = np.asarray(Consultation_Sessions)
  Physical_Activity = np.asarray(Physical_Activity)
  Final_Score = np.asarray(Final_Score)
  
  print(f"1. r for attendance and final score {pearson_correlation(Attendance, Final_Score):.2f}")
  
  reg = lin.LinearRegression().fit(Attendance.reshape((Attendance.size, 1)), Final_Score)
  
  print(f"3. Regression {reg.coef_[0]:.3f}x + {reg.intercept_}")
  
  print(f"4. Prediction {reg.predict(np.asarray([[93.4]]))[0]:.3f}")