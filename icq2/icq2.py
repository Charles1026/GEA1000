import csv 
import numpy as np

if __name__ == "__main__":
  job_levels = []
  salary = []
  remote_ratio = []
  
  with open("./icq2.csv", 'r') as file:
    reader = csv.reader(file)
    
    next(reader)
    
    for row in reader:
      job_level = 0 if row[0] == "EN" or row[0] == "EX" else 1
      job_levels.append(job_level)
      salary.append(int(float(row[3])))
      remote_ratio.append(int(row[4]))
      
      
  job_levels = np.asarray(job_levels)
  salary = np.asarray(salary)
  remote_ratio = np.asarray(remote_ratio)
  
  total_entries = job_levels.shape[0]
  
  
  print(f"Percentage of Entry Jobs: {((total_entries - np.count_nonzero(job_levels)) / total_entries) * 100 :.2f}%")
  
  no_remote_n_high_salary = np.count_nonzero((salary >= 100000) & (remote_ratio == 0))
  no_remote_given_high_salary = no_remote_n_high_salary / (salary >= 100000).shape[0]
  print(f"Rate(No Remote | Salary >= $100k): {no_remote_given_high_salary * 100:.2f}%")
  
  no_remote_n_low_salary = np.count_nonzero((salary < 100000) & (remote_ratio == 0))
  no_remote_given_low_salary = no_remote_n_low_salary / (salary < 100000).shape[0]
  print(f"Rate(No Remote | Salary < $100k): {no_remote_given_low_salary * 100:.2f}%")