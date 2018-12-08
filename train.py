import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
print("Reading responses.csv into a dataframe")
import pandas as pd
df = pd.read_csv("responses.csv")
#print("Printing df.head()")
df.head()
print("********DATA PREPROCESSING********")
print("Finding the unique values for the catogorical attributes")
df['Smoking'].unique()
df['Alcohol'].unique()
df['Punctuality'].unique()
df['Lying'].unique()
df['Internet usage'].unique()
df["Gender"].unique()
df['Only child'].unique()
df['Village - town'].unique()
df['House - block of flats'].unique()
print("Converting catogorical values to integers between (1-5)")
for i in df["Smoking"]:
    if i == "never smoked":
        df.replace(i, 1.0, inplace=True)
    elif i == "tried smoking":
        df.replace(i, 2.0, inplace=True)
    elif i == "former smoker":
        df.replace(i, 3.0, inplace=True)
    elif i == "current smoker":
        df.replace(i, 4.0, inplace=True)
for i in df["Alcohol"]:
    if i == "never":
        df.replace(i, 1.0, inplace=True)
    elif i == "social drinker":
        df.replace(i, 2.0, inplace=True)
    elif i == "drink a lot":
        df.replace(i, 3.0, inplace=True)
for i in df["Punctuality"]:
    if i == "i am always on time":
        df.replace(i, 1.0, inplace=True)
    elif i == "i am often early":
        df.replace(i, 2.0, inplace=True)
    elif i == "i am often running late":
        df.replace(i, 3.0, inplace=True)
for i in df["Lying"]:
    if i == "never":
        df.replace(i, 1.0, inplace=True)
    elif i == "sometimes":
        df.replace(i, 2.0, inplace=True)
    elif i == "only to avoid hurting someone":
        df.replace(i, 3.0, inplace=True)
    elif i == "everytime it suits me":
        df.replace(i, 4.0, inplace=True)
for i in df["Internet usage"]:
    if i == "few hours a day":
        df.replace(i, 1.0, inplace=True)
    elif i == "most of the day":
        df.replace(i, 2.0, inplace=True)
    elif i == "less than an hour a day":
        df.replace(i, 3.0, inplace=True)
    elif i == "no time at all":
        df.replace(i, 4.0, inplace=True)
for i in df["Gender"]:
    if i == "female":
        df.replace(i, 1.0, inplace=True)
    elif i == "male":
        df.replace(i, 2.0, inplace=True)
for i in df["Left - right handed"]:
    if i == "right handed":
        df.replace(i, 1.0, inplace=True)
    elif i == "left handed":
        df.replace(i, 2.0, inplace=True)
for i in df["Education"]:
    if i == "college/bachelor degree":
        df.replace(i, 1.0, inplace=True)
    elif i == "secondary school":
        df.replace(i, 2.0, inplace=True)
    elif i == "primary school":
        df.replace(i, 3.0, inplace=True)
    elif i == "masters degree":
        df.replace(i, 4.0, inplace=True)
    elif i == "doctorate degree":
        df.replace(i, 5.0, inplace=True)
    elif i == "currently a primary school pupil":
        df.replace(i, 6.0, inplace=True)
for i in df["Only child"]:
    if i == "no":
        df.replace(i, 1.0, inplace=True)
    elif i == "yes":
        df.replace(i, 2.0, inplace=True)
for i in df["House - block of flats"]:
    if i == "block of flats":
        df.replace(i, 1.0, inplace=True)
    elif i == "house/bungalow":
        df.replace(i, 2.0, inplace=True)
for i in df["Village - town"]:
    if i == "village":
        df.replace(i, 1.0, inplace=True)
    elif i == "city":
        df.replace(i, 2.0, inplace=True)
print("Imputing the NaN values with the mode")
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imp.fit(df)
df_data = imp.transform(df)
df = pd.DataFrame(data=df_data[:,:],
                     index=[i for i in range(len(df_data))],
                     columns=df.columns.tolist())
from sklearn import model_selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

print("Selecting the best features")
y_all = [i for i in df.iloc[:, 95]]
responses_train = df.drop(["Empathy"], axis=1)
x_all = responses_train.iloc[:, :].values

test = SelectKBest(score_func=chi2, k=100)
bestfit = test.fit(x_all, y_all)
features = bestfit.transform(x_all)
#print(features.shape)
x_all=features
print("Partitioning the dataset into train, dev and test")
y = y_all[:int(len(y_all)*0.8)]
y_test = y_all[int(len(y_all)*0.8):]
x = x_all[:int(len(x_all)*0.8)]
x_test = x_all[int(len(x_all)*0.8):]
print("********SAVING x_test AND y_test USING PICKLE********")
x_train, x_dev, y_train, y_dev = model_selection.train_test_split(x, y, test_size=20, random_state=50)
import pickle
with open('x_test.pickle', 'wb') as output:
    pickle.dump(x_test, output)
with open('y_test.pickle', 'wb') as output:
      pickle.dump(y_test, output)
print("********SAVING x_train AND y_train USING PICKLE********")
with open('x_train.pickle', 'wb') as output:
    pickle.dump(x_train, output)
with open('y_train.pickle', 'wb') as output:
      pickle.dump(y_train, output)
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import warnings
print("********MODEL EVALUATION ON TRAINING DATA********")
models=[]
models.append(('Baseline Classifier',DummyClassifier(strategy='most_frequent')))
models.append(('SVC',SVC()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('Logistic Regression',LogisticRegression()))
models.append(('Random Forest',RandomForestClassifier(random_state=1)))
print("Evaluating classifiers on the training data using 10 fold cross validation")
for name,model in models:
    
            #print("Evaluating classifiers on the training data using 10 fold cross validation")
            kfold = model_selection.KFold(n_splits=10, random_state=50)
            pred = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
            print("Accuracy for", name,"is")
            print( pred.mean()*100, "%")

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
print("Random Forest has the highest accuracy in train set")

print("Tuning hyperparameters on the validation data for all the above models")

#logistic = LogisticRegression()
#C = np.logspace(0, 4, 10)
#penalty = ['l1', 'l2']

#multiclass=['ovr','multinomial']
#solver=['newton-cg', 'lbfgs', 'sag', 'saga']
#hyperparameters = dict(C=C, multi_class=multiclass, solver=solver)
#hyperparameters = dict(C=C, penalty=penalty)
#GS = GridSearchCV(logistic, hyperparameters, cv=10, verbose=0)
print("Tuning hyperparameters for RandomForest ")
rfc=RandomForestClassifier(random_state=42)
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
model=CV_rfc.fit(x_train, y_train)

pred_dev = model.predict(x_dev)
print("Accuracy on the development data: ",accuracy_score(y_dev, pred_dev)*100, "%")

params = model.best_estimator_.get_params()
print("best parameters found through GridSearch for Random Forest: ",params)


print("Choosing the best parameters for Random Forest")
model = RandomForestClassifier(bootstrap=params['bootstrap'], criterion=params['criterion'], max_depth=params['max_depth'], n_estimators=params['n_estimators'])
model.fit(x_train, y_train)

#filename = 'finalized_model.sav'
#pickle.dump(model, open(filename, 'wb'))
print("Saving the model")
with open('model.pickle', 'wb') as output:
    pickle.dump(model, output)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


print("Tuning hyperparameters  for Logistic Regression on the validation data")

logistic = LogisticRegression()
C = np.logspace(0, 4, 5)
multiclass=['ovr','multinomial']
solver=['newton-cg','lbfgs','saga']
hyperparameters = dict(C=C, multi_class=multiclass, solver=solver)
GS = GridSearchCV(logistic, hyperparameters, cv=10, verbose=0)
modellr = GS.fit(x_train, y_train)
pred_devlr = modellr.predict(x_dev)
print("Accuracy on the development data: ",accuracy_score(y_dev, pred_devlr)*100, "%")

paramslr = modellr.best_estimator_.get_params()
print("best parameters found through GridSearch for Logistic Regression: ",paramslr)

print("Building a Logistic Regression model with the best parameters")
modellr = LogisticRegression(multi_class=paramslr['multi_class'],solver=paramslr['solver'])
modellr.fit(x_train, y_train)

print("Saving the model")
with open('modellr.pickle', 'wb') as output:
    pickle.dump(modellr, output)

print("Tuning Hyperparametrs for KNN using GridSearchCV")
# define the parameter values that should be searched
# for python 2, k_range = range(1, 31)
knn=KNeighborsClassifier()
k_range = list(range(1, 31))
print(k_range)
# create a parameter grid: map the parameter names to the values that should be searched
# simply a python dictionary
# key: parameter name
# value: list of values that should be searched for that parameter
# single key-value pair for param_grid
param_grid = dict(n_neighbors=k_range)
print(param_grid)
# instantiate the grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(x_train,y_train)
predknn=grid.predict(x_dev)
print("Accuracy on the development data: ",accuracy_score(y_dev, predknn)*100, "%")

print("Getting the best parameters for KNN")
paramsknn = grid.best_estimator_.get_params()
print("best parameters found through GridSearch for KNN: ",paramsknn)

print("Building KNN classifier with the best parameters")
grid =KNeighborsClassifier(algorithm=paramsknn['algorithm'], metric=paramsknn['metric'], weights=paramsknn['weights'])
grid.fit(x_train, y_train)
print("Saving the model")
with open('modelknn.pickle', 'wb') as output:
    pickle.dump(grid, output)

print("tuning hyperparameters for SVM")
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : gammas}
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid)
grid_search.fit(x_train, y_train)
#grid_search.best_params_

print("Choosing the best parameters for SVM")
paramsvm = grid_search.best_estimator_.get_params()
print("best parameters found through GridSearch for SVM: ",paramsvm)


predsvm=grid_search.predict(x_dev)
print("Accuracy on the development data: ",accuracy_score(y_dev, predsvm)*100, "%")
print("Building SVM model with its best parameters")
grid_search = SVC(C=paramsvm['C'], cache_size=paramsvm['cache_size'], kernel=paramsvm['kernel'], gamma=paramsvm['gamma'])
grid_search.fit(x_train, y_train)

print("Saving the model")
with open('modelsvm.pickle', 'wb') as output:
    pickle.dump(grid_search, output)