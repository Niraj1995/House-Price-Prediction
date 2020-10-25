import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns

### Display all columns
pd.pandas.set_option('display.max_columns',None)

dataset = pd.read_csv("D:\KrishNaikSessions\House_Price_Prediction_Train.csv")
dataset.shape

dataset.head()

##### MISSING VALUES
feature_with_na = [features for features in dataset.columns if dataset[features].isnull().sum()>1]    

for features in feature_with_na:
    print(features,np.round(dataset[features].isnull().mean(),4),'% missing values')
    
#### SINCE THEY ARE MISSING VALUES, WE NEED TO FIND THE RELATIONSHIP BETWEEN MISSING VALUES AND SALES PRICE
for features in feature_with_na:
    data = dataset.copy()
    
    ### Let's make a variable that indicates 1 if the observation was missing or 0
    data[features] = np.where(data[features].isnull(),1,0)
    
    ### Let's calculate the mean SalePrice where the information is missing or pre
    data.groupby(features)['SalePrice'].median().plot.bar()
    plt.title(features)
    plt.show()
    
### Here with the relation between the missing values and the dependent variable is clearly visible. So we need to replace these nan with something meaningful which we will do in the feature Engineering section
print("Id of house {}".format(len(dataset.Id)))


### NUMERICAL VARIABLES
#List of numerical variables
numerical_features = [features for features in dataset.columns if dataset[features].dtypes != 'O']

print("Number of numerical features",len(numerical_features))

### visualize the numerical features
dataset[numerical_features].head()

# List of variables that contain year information
year_feature = [feature for feature in numerical_features if 'Yr' in feature or "Year" in feature]
year_feature


## Let's explore the content of these year variables
for year in year_feature:
    print(year,dataset[year].unique())
    
### Lets analysis datetime variable

dataset.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel("Year Sold")
plt.ylabel("SalePrice")
plt.title("House Price vs Sale Price")


### we will compare the difference between all year feature with sale price

for features in year_feature:
    if features != 'YrSold':
        data = dataset.copy()
   #### We will capture the difference between year variable and the hou
        data[features] = data['YrSold'] - data[features]
    
        plt.scatter(data[features],data['SalePrice'])
        plt.xlabel(features)
        plt.ylabel("SalePrice")
        plt.show()
        
    
### Numerical variables are of 2 types continuous and discrete
discrete_features = [features for features in numerical_features if len(dataset[features].unique())<25 and features not in year_feature+['Id']]
print("discrete variable {}".format(len(discrete_features)))

discrete_features

dataset[discrete_features].head()

### Lets find the relationship between them and Sale Price
for features in discrete_features:
    data = dataset.copy()
    data.groupby(features)['SalePrice'].median().plot.bar()
    plt.xlabel(features)
    plt.ylabel("SalePrice")
    plt.title(features)
    plt.show()

### There is a relationship between variable number and sale price    
### CONTINUOUS VARIABLE
Continuous_feature = [feature for feature in numerical_features if feature not in discrete_features+year_feature+['Id']]
print("Continuous features {}".format(len(Continuous_feature)))

#Lets analysis the continuous variables by creating histograms to understand 
for feature in Continuous_feature:
    data= dataset.copy()
    data[feature].hist(bins = 25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()


##### Logarithmic transformation
for feature in Continuous_feature:
    data = dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data['SalePrice'] = np.log(data['SalePrice'])
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel("Sale price")
        plt.title(feature + " VS Sale Price")
        plt.show()
        
        
### OUTLIER
for feature in Continuous_feature:
    data = dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data.boxplot(column = feature)
        plt.ylabel("Sale price")
        plt.title(feature + " VS Sale Price")
        plt.show()
        
        
### Categorial Variables
Categorial_feature = [feature for feature in dataset.columns if dataset[feature].dtype == "O"]
Categorial_feature


for feature in Categorial_feature:
    print("The feature is {} and count of unique is {}".format(feature,len(dataset[feature].unique())))


#### Find out the relationship between categorial variable and sale price
for feature in Categorial_feature:
    data = dataset.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel("SalePrice")
    plt.title(feature)
    plt.show()


#### Missing Values are present in Categorial Variables

features_nan = [feature for feature in dataset.columns if dataset[feature].isnull().sum() > 1 and dataset[feature].dtypes == "O"]

for feature in features_nan:
    print("{} : {} missing values".format(feature,np.round(dataset[feature].isnull().mean(),4)))
    

##### Replace the  missing values with a new label
def replace_cat_feature(dataset,features_nan):
    data = dataset.copy()
    data[features_nan] = data[features_nan].fillna("Missing")
    return data

dataset = replace_cat_feature(dataset,features_nan)

dataset[features_nan].isnull().sum()


dataset.head()

##### Numerical Variables with Missing Values
numerical_nan = [feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtypes != "O"]

for feature in numerical_nan:
    print("{} : {} missing values".format(feature,np.round(dataset[feature].isnull().mean(),4)))


##### Replacing the Null of numerical variable with Median or Mode because there are lot of Outliers Present
for feature in numerical_nan:
    median_value = dataset[feature].median()
    
    #### create a new feature to capture the nan values
    dataset[feature+'nan'] = np.where(dataset[feature].isnull(),1,0)
    dataset[feature].fillna(median_value,inplace= True)
    
dataset[numerical_nan].isnull().sum()        

dataset.head(50)
    
#### Temporal Variables
for feature in ["YearBuilt","YearRemodAdd","GarageYrBlt"]:
    dataset[feature] = dataset["YrSold"] - dataset[feature]


#### Logarithmic transformation on Numerical Variables
num_features = ["LotFrontage","LotArea","1stFlrSF","GrLivArea","SalePrice"]

for feature in num_features:
    dataset[feature] = np.log(dataset[feature])

dataset.head()

#### HANDLING RARE CATEGORIAL FEATURES
#### Removing categories containing less than 1 percent of the observations
categorial_features = [feature for feature in dataset.columns if dataset[feature].dtypes == "O"]

categorial_features
    
    
for feature in  categorial_features:
    temp = dataset.groupby(feature)['SalePrice'].count()/len(dataset)
    temp_df = temp[temp>0.01].index
    dataset[feature] = np.where(dataset[feature].isin(temp_df),dataset[feature],'Rare_var')
    
dataset.head(50)

#### Categorial Variable into the numerical values
for features in categorial_features:
    labels_ordered = dataset.groupby(features)['SalePrice'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    dataset[features]=dataset[features].map(labels_ordered)


dataset.head(100)

#### Feature Scaling
feature_scale = [feature for feature in dataset.columns if feature not in ['Id','SalePrice']]


from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()
scalar.fit(dataset[feature_scale])

data = pd.concat([dataset[['Id','SalePrice']].reset_index(drop = True),pd.DataFrame(scalar.transform(dataset[feature_scale]),columns = feature_scale)],axis = 1)

data.head()

#### Feature Selection
y_train = data[["SalePrice"]]

X_train = data.drop(['Id','SalePrice','SaleCondition'],axis=1)

#### Apply feature selection
# first I specify Lasso Regression model and I select suitable alpha value
# the bigger the alpha value the less features will be selected.

# then i will use the selectFromModel object from sklearn, which will select the features which coefficient are non zero

from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

feature_sel_model = SelectFromModel(Lasso(alpha = 0.008,random_state = 0))
feature_sel_model.fit(X_train,y_train)

feature_sel_model.get_support()

### Lets print the number of total and selected columns
Selected_feat = X_train.columns[(feature_sel_model.get_support())]
Selected_feat

X_train = X_train[Selected_feat]


#### Model building
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()

regressor.fit(X_train,y_train)

Prediction = regressor.predict(X_train)
Prediction = np.expm1(Prediction)

# evaluate model
cv = KFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(regressor, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# convert scores to positive
scores = np.absolute(scores)
# summarize the result
s_mean = np.mean(scores)
print('Mean MAE: %.3f' % (s_mean))

from sklearn.metrics import r2_score

Score = r2_score(Prediction,y_train)data
Score

#### Dump the model file in pickel

import pickle
pickle.dump(regressor,open('Model.pkl','wb'))    


model = pickle.load(open('Model.pkl','rb'))
print(model.predict([[6,5,9,0,2,2,1,6.96602,6.96602,2,1,3,4,1,1]]))
print(regressor.predict([[5,5,43,0,2,1,1,1256,1256,1,0,1,4,3,1]]))
print(regressor.predict([[5,5,43,0,2,1,1,7.13,7.13,1,0,1,4,3,1]]))
print(regressor.predict([[0.227273,0.444444,0.721311,0.000000,0.500000,0.250000,1.000000,0.501253,0.468559,0.333333,0.000000,0.200000,0.800000,1.000000,0.250000]]))


X_train_features = [feature for feature in X_train.columns]
X_train_data = dataset[X_train_features]
scalar.fit(X_train_data)
X_train_data = scalar.transform(X_train_data)
pickle.dump(scalar, open('scalar.pkl', 'wb'))
