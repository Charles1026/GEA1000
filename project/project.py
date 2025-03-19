import csv 
import numpy as np
from statistics import NormalDist
import scipy.stats as stats
# import matplotlib.pyplot as plt
# import sklearn.linear_model as lin

def pearson_correlation(X,Y):
    if len(X)==len(Y):
        Sum_xy = sum((X-X.mean())*(Y-Y.mean()))
        Sum_x_squared = sum((X-X.mean())**2)
        Sum_y_squared = sum((Y-Y.mean())**2)       
        corr = Sum_xy / np.sqrt(Sum_x_squared * Sum_y_squared)
    return corr

if __name__ == "__main__": 
  genders = [] # 0 = female 1 male
  gender_map = {"Female" : 0, "Male": 1}
  
  
  has_gt_8_hrs_sleep = []
  has_depression = []
  
  with open("./group1.csv", 'r') as file:
    reader = csv.reader(file)
    
    next(reader)
    for row in reader:
      if row[4] != "Student":
        continue
      
      genders.append(gender_map[row[1]])
      has_gt_8_hrs_sleep.append(1 if row[10] == "More than 8 hours" else 0)
      has_depression.append(int(row[17]))
  
  
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
  
  
  
  