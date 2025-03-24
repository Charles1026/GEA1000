import csv 
import numpy as np
from statistics import NormalDist
import scipy.stats as stats
# import matplotlib.pyplot as plt
import sklearn.linear_model as lin
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score

def pearson_correlation(X,Y):
    if len(X)==len(Y):
        Sum_xy = sum((X-X.mean())*(Y-Y.mean()))
        Sum_x_squared = sum((X-X.mean())**2)
        Sum_y_squared = sum((Y-Y.mean())**2)       
        corr = Sum_xy / np.sqrt(Sum_x_squared * Sum_y_squared)
    return corr
  
def standardise(arr: np.ndarray, axis = None) -> np.ndarray:
  arr = arr
  return (arr - arr.mean(axis = axis, keepdims = True)) / arr.std(axis = axis, keepdims = True)

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
  degrees = []
  degree_map = {"B.Tech": 0, "LLB": 1, "M.Tech": 2, "B.Com": 3, "B.Ed": 4, "B.Arch": 5, "Class 12": 6, "BHM": 7, "M.Ed": 8, 
                "M.Pharm": 9,"MSc": 10, "BSc": 11, "B.Pharm": 12, "M.Com": 13, "MHM": 14, "BBA": 15, "PhD": 16, "MA": 17,
                "MBBS": 18, "LLM": 19, "MBA": 20, "MCA": 21, "BCA": 22, "BA": 23, "BE": 24, "MD": 25, "ME": 26}
  
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
  print(f"B.6 95% Confidence Interval for depression rate [{depression_rate - margin_of_error:.3f}, {depression_rate + margin_of_error:.3f}], with p = {depression_rate:.3f}\n")
  
  
  # B.7 
  observed_counts = np.asarray([np.count_nonzero(has_gt_8_hrs_sleep * has_depression),
                               np.count_nonzero((1 - has_gt_8_hrs_sleep) * has_depression),
                               np.count_nonzero(has_gt_8_hrs_sleep * (1 - has_depression)),
                               np.count_nonzero((1 - has_gt_8_hrs_sleep) * (1 - has_depression))]).reshape((2, 2))
  
  row_totals = observed_counts.sum(axis = 1, keepdims = True)
  col_totals = observed_counts.sum(axis = 0, keepdims = True)
  expected_counts = (row_totals @ col_totals) / sample_size
  
  chi2_statistic = np.sum(((observed_counts - expected_counts) ** 2) / expected_counts)
  print(f"B.7 Chi Square Test for independence statistic: {chi2_statistic:.3f}, p-Value: {stats.chi2.sf(chi2_statistic, 1):.3f}\n")
  
  
  # B.8
  num_males = np.count_nonzero(genders)
  
  depression_rate_in_females = np.count_nonzero(has_depression * (1 - genders)) / (sample_size - num_males)
  
  depression_rate_in_males = np.count_nonzero(has_depression * (genders)) / num_males
  
  pooled_rate = depression_rate
 
  z_score = (depression_rate_in_females - depression_rate_in_males) / np.sqrt(pooled_rate * (1 - pooled_rate) * ((1 / (num_males)) + (1 / (sample_size - num_males))))
  
  print(f"B.8 2 sample t-test for proportion of depressed males against females z-value: {z_score:.3f}\n")
  
  
  # Linear Regression Model, standardise all non binary variables
  X = standardise(np.asarray([genders, ages, academic_pressure, cgpa, study_satisfaction, sleep_duration, dietary_habits, 
                  suicidal_thoughts, work_study_hours, financial_stress, fam_history_of_mi]), axis = 1).transpose()
  y = has_depression
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 69420) 
  
  print(f"Split into {y_train.size} train samples and {y_test.size} test samples")
  
  classifier = lin.LogisticRegression().fit(X_train, y_train)
  # print(classifier.coef_)
  
  # evaluate model
  y_pred = classifier.predict_proba(X_test)[:,1]
  print(f"Base Model log loss: {log_loss(y_test, y_pred):.3f}, AUC-ROC: {roc_auc_score(y_test, y_pred):.3f}")
  
  # perform significance test on 
  p = classifier.predict_proba(X_train).transpose() # predict returns p and 1 - p
  W = np.diag(p[0] * p[1]) # p * 1-p
  V = np.linalg.inv(X_train.transpose() @ W @ X_train)
  std_errs = np.sqrt(np.diag(V))
  z_stat = classifier.coef_ / std_errs
  print(z_stat)
  
  filter = (np.abs(z_stat) > NormalDist().inv_cdf((1 + 0.95) / 2)).flatten()
  
  new_X_train = X_train[:, filter]
  new_X_test = X_test[:, filter]
  
  #
  new_classifier = lin.LogisticRegression().fit(new_X_train, y_train)
  # print(new_classifier.coef_)
  
  # evaluate model
  new_y_pred = new_classifier.predict_proba(new_X_test)[:,1]
  print(f"Reduced Model log loss: {log_loss(y_test, new_y_pred):.3f}, AUC-ROC: {roc_auc_score(y_test, new_y_pred):.3f}")