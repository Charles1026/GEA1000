import csv 
import numpy as np
from statistics import NormalDist
import scipy.stats as stats
# import matplotlib.pyplot as plt
import sklearn.linear_model as lin

def pearson_correlation(X,Y):
    if len(X)==len(Y):
        Sum_xy = sum((X-X.mean())*(Y-Y.mean()))
        Sum_x_squared = sum((X-X.mean())**2)
        Sum_y_squared = sum((Y-Y.mean())**2)       
        corr = Sum_xy / np.sqrt(Sum_x_squared * Sum_y_squared)
    return corr

if __name__ == "__main__":
  yes_no_map = {"No": 0, "Yes": 1}
  
  genders = [] # 0 = female 1 male
  gender_map = {"Female" : 0, "Male": 1}
  
  ages = []
  academic_pressure = []
  cgpa = []
  study_satisfaction = []
  sleep_duration = []
  sleep_duration_map = {"Less than 5 hours": 0, "5-6 hours": 1, "6-7 hours": 2, "7-8 hours": 3, "More than 8 hours": 4}
  dietary_habits = []
  dietary_habits_map = {"Unhealthy": 0, "Moderate": 1, "Healthy": 2}
  suicidal_thoughts = []
  work_study_hours = []
  financial_stress = []
  fam_history_of_mi = []
  
  has_gt_8_hrs_sleep = []
  has_depression = []
  
  with open("./group1.csv", 'r') as file:
    reader = csv.reader(file)
    
    next(reader)
    rowNum = 0
    try:
      for row in reader:
        rowNum += 1
        hasInvalid = False
        for value in row:
          if value == "Others" or value == "NA":
            hasInvalid = True
            break
        if row[4] != "Student" or row[7] == 0 or hasInvalid:
          continue
        
        genders.append(gender_map[row[1]])
        
        ages.append(int(row[2]))
        academic_pressure.append(int(row[5]))
        cgpa.append(float(row[7]))
        study_satisfaction.append(int(row[8]))
        sleep_duration.append(sleep_duration_map[row[10]])
        dietary_habits.append(dietary_habits_map[row[11]])
        suicidal_thoughts.append(yes_no_map[row[13]])
        work_study_hours.append(int(row[14]))
        financial_stress.append(int(row[15]))
        fam_history_of_mi.append(yes_no_map[row[16]])
        
        has_gt_8_hrs_sleep.append(1 if row[10] == "More than 8 hours" else 0)
        has_depression.append(int(row[17]))
    except Exception as err:
      print(f"Error at row {rowNum}: {err}")
  
  
  genders = np.asarray(genders)
  has_gt_8_hrs_sleep = np.asarray(has_gt_8_hrs_sleep)
  has_depression = np.asarray(has_depression)
  sample_size = has_depression.shape[0]
  
  # B.6
  depression_rate = np.count_nonzero(has_depression) / sample_size
  margin_of_error = NormalDist().inv_cdf((1 + 0.95) / 2) * np.sqrt(depression_rate * (1 - depression_rate) / sample_size)
  print(f"B.6 95% Confidence Interval [{depression_rate - margin_of_error:.3f}, {depression_rate + margin_of_error:.3f}]")
  
  
  # B.7
  gt_8_hrs_sleep_rate = np.count_nonzero(has_gt_8_hrs_sleep) / sample_size
  
  expected_rates = np.asarray([gt_8_hrs_sleep_rate * depression_rate, 
                               (1 - gt_8_hrs_sleep_rate) * depression_rate, 
                               gt_8_hrs_sleep_rate * (1 - depression_rate), 
                               (1 - gt_8_hrs_sleep_rate) * (1 - depression_rate)])
  
  observed_rates = np.asarray([np.count_nonzero(has_gt_8_hrs_sleep * has_depression) / sample_size,
                               np.count_nonzero((1 - has_gt_8_hrs_sleep) * has_depression) / sample_size,
                               np.count_nonzero(has_gt_8_hrs_sleep * (1 - has_depression)) / sample_size,
                               np.count_nonzero((1 - has_gt_8_hrs_sleep) * (1 - has_depression)) / sample_size])
  
  chi2_statistic = np.sum(((observed_rates - expected_rates) ** 2) / expected_rates)
  print(f"B.7 Chi Square Test p-Value: {stats.chi2.sf(chi2_statistic, 1):.3f}")
  
  
  # B.8
  num_males = np.count_nonzero(genders)
  
  depression_rate_in_females = np.count_nonzero(has_depression * (1 - genders)) / (sample_size - num_males)
  
  depression_rate_in_males = np.count_nonzero(has_depression * (genders)) / num_males
  
  pooled_rate = depression_rate
 
  z_score = (depression_rate_in_females - depression_rate_in_males) / np.sqrt(pooled_rate * (1 - pooled_rate) * ((1 / (num_males)) + (1 / (sample_size - num_males))))
  
  print(f"B.8 {z_score:.3f}")
  
  
  # Linear Regression Model
  X = np.asarray([genders, ages, academic_pressure, cgpa, study_satisfaction, sleep_duration, dietary_habits, suicidal_thoughts, work_study_hours, financial_stress, fam_history_of_mi]).transpose()
  y = has_depression
  
  print(X.shape, y.shape)
  
  classifier = lin.LogisticRegression().fit(X, y)
  print(classifier.coef_, classifier.intercept_)