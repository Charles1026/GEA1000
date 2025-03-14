import csv 
import numpy as np
import matplotlib.pyplot as plt

def pearson_correlation(X,Y):
    if len(X)==len(Y):
        Sum_xy = sum((X-X.mean())*(Y-Y.mean()))
        Sum_x_squared = sum((X-X.mean())**2)
        Sum_y_squared = sum((Y-Y.mean())**2)       
        corr = Sum_xy / np.sqrt(Sum_x_squared * Sum_y_squared)
    return corr

if __name__ == "__main__":
  Birth_Rate_per1000 = []
  Death_Rate_per1000 = []
  GDP_per_capita = []
  Median_Age = []
  Population = []
  
  with open("./population_age.csv", 'r') as file:
    reader = csv.reader(file)
    
    next(reader)
    for row in reader:
      if (float(row[3]) < 15):
        continue
      
      Birth_Rate_per1000.append(float(row[0]))
      Death_Rate_per1000.append(float(row[1]))
      GDP_per_capita.append(float(row[2]))
      Median_Age.append(float(row[3]))
      Population.append(float(row[4]))
      
  Birth_Rate_per1000 = np.asarray(Birth_Rate_per1000)
  Death_Rate_per1000 = np.asarray(Death_Rate_per1000)
  GDP_per_capita = np.asarray(GDP_per_capita)
  Median_Age = np.asarray(Median_Age)
  Population = np.asarray(Population)
  
  
  fig = plt.figure()
  ax1 = fig.add_subplot(121)
  ax1.scatter(Median_Age, Population)
  ax1.set_title(f"Population vs Median Age (R = {pearson_correlation(Median_Age, Population):.2f})")
  
  ax2=fig.add_subplot(122)
  ax2.scatter(Median_Age, np.log(Population))
  ax2.set_title(f"ln(Population) vs Median Age (R = {pearson_correlation(Median_Age, np.log(Population)):.2f})")
  
  plt.show()