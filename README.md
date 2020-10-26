# House-Price-Prediction

• This repository consists of files required to deploy a Machine Learning Web App created with Flask on Heroku platform.

• If you want to view the deployed model, click on the following link:
Deployed at:https://homepriceprediction-api.herokuapp.com/

• If you are searching for Code. Click the link mentioned below for the same:
https://github.com/Niraj1995/House-Price-Prediction/blob/main/Model.py

• Data Description

Data Features With Categories - Data Code
1. Neighborhood:-
NAmes-8
CollgCr-14
OldTown-4
Edwards-5
Somerst-18
Gilbert-13
NridgHt-21
Sawyer-6
NWAmes-12
SawyerW-10
BrkSide-3
Crawfor-16
Mitchel-9
NoRidge-22
Timber-19
IDOTRR-0
ClearCr-17
SWISU-15
StoneBr-20
Blmngtn-11
MeadowV-15
BrDale-1

2. OverallQual
Range Between 1 to 10

3. YearRemodAdd
Range Between 0 to 60

4. RoofStyle:-
Gable - 0
Hip - 1
Flat/Gambrel/Mansard/Shed - 2

5. BsmtQual:-
TA - 2
Gd - 3
Ex - 4
Fa - 1

6. BsmtExposure:-
No - 1 
Av - 3 
Gd - 4
Mn - 2

7. KitchenQual:-
TA - 1
Gd - 2
Ex - 3
Fa - 0

8. Fireplaces:-
Range 0 - 3

9. FireplaceQu:-
Gd - 4
TA - 3
Fa - 2
Ex - 5
Po - 0

10. GarageType:-
Attchd - 4
Detchd - 2
BuiltIn	- 5
Basment - 3
CarPort/2Types - 1

11. GarageFinish:-
Unf - 1
RFn - 2
Fin - 3

12. GarageCars:-
Range 0 - 4

13.CentralAir:-
Y - 1
N - 0

MODEL PROCESSES:-
This project includes all lifecycle in these data science project.

1. Data Analysis.
2. Feature Engineering.
3. Feature Selection.
4. Model Building.
5. Model Deployment.

1. Data Analysis:-

	In Data Analysis Process the all the variables have plotted in Bar/Box/Scatter Plot to observe the outliers/relationship
	of variables with the target variables/skewness if present in the numerical varibles.
	
2. Feature Engineering:-
	i. Feature Creation:- The Temporary Variable like the Year Variable. 3 New Variable were generated .
		The Age of the House from the time the house was build till the time it was sold.
		The Age of the House from the time the house was modified till the time it was sold.
		The Age of the House from the time the Garage was build till the time it was sold.
		
	ii. Missing Value Treatment:
		For Numerical Variable:- The missing value was replaced with Median or Mode incase of numerical variable because of outlier present.
		For Categorial Variable:- The missing value was categorized as a separate catgory incase of Categorial variable.
		
	iii. Skewness:
		For Skewness the numerical variable was performed log transformation to handle the skewness.
		
	iv. Handling Rare categories features:
		Categories containing less than 1 percent observation are categorized as Rare variables
		
	v. Feature scaling:
		All variables are scaled using min max scalar so that all variables have equal magnitude.
		
3. Feature Selection
	For Feature Selection we used Lasso regression as the Lasso regression penalizes variables to zero 
	So only the variables that helps to predict the independent variables are left.
	Out of the 81 variables using lasso regression 15 best variables are used for prediction.
	
4. 	Model Building
	Random Forest and Multiple Linear Regression was the model chosen for model building. Both the model was build on the 15 variables.
	Random Forest was chosen as the best model because Random Forest gave a good accuracy between the two.

5. Model Validation:
	Model Validation was calculated using Accuracy which was done on the test data which gave accuracy of more than 95%.
	Also for more safety the model was tested using k fold validation with the accuracy measure as mean absolute error which gave error as 0.107 
	after 10 folds.
	
6. Model Deployment:
	Finally The Model was Deployed using the Flask libraries in the Heroku App.
