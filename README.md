# Champlain Alumni Salary Prediction

## Overview
This project is a statistical analysis of post-graduation career salaries for Champlain College alumni.

The model uses three different machine learning algorithms:
1. Linear regression
2. Random forest
3. K Nearest Neighbors

The goal of this project is to analyze how much Champlain graduates make at their first jobs out of college and to determine
which factors are the best predictors of salary. We hope the school's career services can make use of our findings.

All points in the original dataset were collected by the Champlain College Career Collaborative. Every year, the college
sends out an "exit survey" to all Champlain alumni who graduated the year prior. This survey collects detailed data such
as major, current employment status, annual salary, number of internships completed while attending college, and
cumulative GPA. The oldest data is from the graduating class of 2016 and the most recent data is from the graduating
class of 2023.

All data was anonymized by Champlain College administration before being distributed to us (all 
potentially identifying information was removed from the set).

## Instructions for Running the Models
The code in this repository currently allows the user to run a linear regression, random forest, and KNN analysis on the
data for three majors:
1. Computer Science
2. Computer Networking & Cybersecurity
3. Computer & Digital Forensics

To run the program, run the 'main.py' file. This will run all the models back-to-back. Otherwise, you can run each model 
separately in their respective file (i.e. run the linear regression model by running 'linear_regression.py', etc.). To
generate the 3D plots for linear regression, uncomment the 'plot3d' line in the 'run_regression_model_on_major()'
function. These plots can also be found in the 'LinearRegression_ProjectFigures' folder in the repo.

### Random Forest
In order to edit the parameters of the Random Forest Regressor, simply change, remove, or add arguments to the RandomForestRegressor function on line 64 of random_forest_regressor.py

## Acknowledgements
Data collected by Champlain College is property of the college.

Special thanks to David Kopec for bringing us the idea and supporting us throughout the process.

Additionally, special thanks to Kerry Shackett and the rest of the Career Collaborative for making this possible.
