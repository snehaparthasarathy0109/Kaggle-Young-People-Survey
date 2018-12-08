print("Predictions on the test data are made in this file,test.py")
import pickle
print("Loading x_train")
with open('x_train.pickle', 'rb') as x_train:
    x_train = pickle.load(x_train)
print("Loading y_train")
with open('y_train.pickle', 'rb') as y_train:
    y_train = pickle.load(y_train)

print("Loading x_test, the features for the test set")
with open('x_test.pickle', 'rb') as x_test:
    x_test = pickle.load(x_test)
print("Loading y_test")
with open('y_test.pickle', 'rb') as y_test:
    y_test = pickle.load(y_test)
print("Loading the baseline model")
with open('modeldummy.pickle', 'rb') as modeldummy:
    modeldummy = pickle.load(modeldummy)

print("Loading the Random Classifier model")
with open('model.pickle', 'rb') as model:
    model = pickle.load(model)
from sklearn.metrics import accuracy_score
pred_test = model.predict(x_test)
#print("Accuracy on the test data: ",accuracy_score(y_test, pred_test)*100, "%")

print("Loading Logistic Regression model")
with open('modellr.pickle', 'rb') as modellr:
    modellr = pickle.load(modellr)
print("Loading the knn model")
with open('modelknn.pickle', 'rb') as modelknn:
    modelknn = pickle.load(modelknn)
print("Loading the SVM model")
with open('modelsvm.pickle', 'rb') as modelsvm:
    modelsvm = pickle.load(modelsvm)
from sklearn.metrics import accuracy_score
pred_dummy = modeldummy.predict(x_test)
print("Accuracy on the test data for Baseline Classifier: ",accuracy_score(y_test, pred_dummy)*100, "%")
pred_test = model.predict(x_test)
print("Accuracy on the test data for Random Forest: ",accuracy_score(y_test, pred_test)*100, "%")
pred_testlr = modellr.predict(x_test)
print("Accuracy on the test data for Logistic Regression: ",accuracy_score(y_test, pred_testlr)*100, "%")
pred_testknn = modelknn.predict(x_test)
print("Accuracy on the test data for KNN: ",accuracy_score(y_test, pred_testknn)*100, "%")
pred_testsvm = modelsvm.predict(x_test)
print("Accuracy on the test data for SVM: ",accuracy_score(y_test, pred_testsvm)*100, "%")