# -*- coding: utf-8 -*-
"""
#Purpose: Simple Linear Regression to evaluate if housing lot area affects a homes sale price. 

#Feature Selection Candidates
0	PID
1	Lot_Area
2	House_Style
3	Overall_Qual
4	Overall_Cond
5	Year_Built
6	Heating_QC
7	Central_Air
8	Gr_Liv_Area
9	Bedroom_AbvGr
10	Fireplaces
11	Garage_Area
12	Mo_Sold
13	Yr_Sold
14	SalePrice
15	Basement_Area
16	Full_Bathroom
17	Half_Bathroom
18	Total_Bathroom
19	Deck_Porch_Area
20	Age_Sold
21	Season_Sold
22	Garage_Type_2
23	Foundation_2
24	Masonry_Veneer
25	Lot_Shape_2
26	House_Style2
27	Overall_Qual2
28	Overall_Cond2
29	Log_Price
30	Bonus
31	score

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Dataset

dataset=pd.read_excel('housing_data.xlsx')
X = dataset.iloc[:,[1]].values
y= dataset.iloc[:,-1].values

#Splitting the dataset into the Training set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Fitting MLR to the Training Set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression() #regressor is the object for the class of LinearRegression
regressor.fit(X_train, y_train)

#Predicting the Training set results
y_pred = regressor.predict(X_test)

#Visualising the Training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Lot Area vs. Sale Price (Training Set)')
plt.xlabel('Lot Area')
plt.ylabel('Sale Price')
plt.show()

#Visualising the Test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Lot Area vs. Sale Price (Test Set)')
plt.xlabel('Lot Area')
plt.ylabel('Sale Price')
plt.show()


#Conclusion: Test SLR data is representative of the training model and indicates similiar
#regression patterns. Outliers are present in both the training/test models indicating data
#normalization should be executed to refine the analysis. 









































