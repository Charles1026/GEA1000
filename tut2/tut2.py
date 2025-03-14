import csv
import numpy as np

if __name__ == "__main__":
  
  total_monthly_use = []
  total_monthly_recycle = []
  
  with open("./recycling_survey.csv", 'r') as file:
    reader = csv.reader(file)
    
    reader.__next__()
    
    for row in reader:
      monthly_use = 0 if row[1] == "Rarely" or row[1] == "Seldom" else 1
      total_monthly_use.append(monthly_use)
      
      monthly_recycle = int(row[2])
      total_monthly_recycle.append(monthly_recycle)
      
  total_monthly_use = np.asarray(total_monthly_use)
  total_monthly_recycle = np.asarray(total_monthly_recycle)
  
  low_use_rate = np.sum(total_monthly_use) / total_monthly_use.shape[0]
  print(f"Low Use Rate {low_use_rate}")
  
  print(np.count_nonzero(total_monthly_recycle > 2))
  
  recycle_and_low_use = np.count_nonzero((total_monthly_use == 0) & (total_monthly_recycle > 2))
  recycle_and_high_use = np.count_nonzero((total_monthly_use == 1) & (total_monthly_recycle > 2))
  
  print(f"Recycle & Low Use: {recycle_and_low_use}, Recycle & High Use: {recycle_and_high_use}")
  print(f"Recycle & Low Use: {recycle_and_low_use / low_use_rate}, Recycle & High Use: {recycle_and_high_use / (1 - low_use_rate)}")