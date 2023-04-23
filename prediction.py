import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt
from six import StringIO
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from scipy.stats import uniform

# loading data
telco = pd.read_csv('telco_2023.csv')

# keeping only the variables that have been considered important
# testing with all the variables make 2/3 algoriths have worse results

data1 = telco.iloc[:, 3:12] 
data2 = telco.iloc[:, 14:17]
data3 = telco.iloc[:,18]

data = pd.concat([data1,data2,data3],axis=1)

del data1, data2, data3


X = data.drop('churn', axis=1)
y = data['churn']

# checking to see if the categories are unbalanced

print(data['churn'].value_counts())  # 2.65 ratio - data is unbalanced 

# Split the data into training and test sets, stratifying the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2,stratify=y)  # stratify=y  the proportion of the target variable (y) is preserved in each split

# looking for the best performing Decision Tree

best_score = 0
best_n_splits = 0
best_model_dt = None

# best model_dt with recall scoring - I am using recall as the score because I want it to be "sensitive" to churn

for n_splits in range(5,12):
    model_dt = DecisionTreeClassifier()
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)
    scores = cross_val_score(model_dt, X_train, y_train, cv=kfold, scoring='recall')
    score = scores.mean()
    if score > best_score:
        best_score = score
        best_n_splits = n_splits
        best_model_dt = model_dt

best_model_dt.fit(X_train, y_train)
pred = best_model_dt.predict(X_test)
print("Best n_splits:", best_n_splits)
print("Best Recall:", best_score)
print("Accuracy:", metrics.accuracy_score(y_test, pred))
precision, recall, fscore, supp = metrics.precision_recall_fscore_support(y_test, pred, average=None)
print("Precision:", precision)
print("Recall:", recall)
print("F-score:", fscore)

confusion_matrix = metrics.confusion_matrix(y_test, pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['No churn','Churn'])
cm_display.plot()
plt.show()

# Visualizing the decision tree
dot_data = StringIO()
export_graphviz(best_model_dt, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

# knn model 

best_score = 0
best_n_neighbors = 0
best_model = None

for n_neighbors in range(2,12):
    for n_splits in range(5,12):
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)
        model_knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        scores = cross_val_score(model_knn, X_train, y_train, cv=kfold, scoring='recall')
        score = scores.mean()
        if score > best_score:
            best_score = score
            best_n_neighbors = n_neighbors
            best_model = model_knn

best_model.fit(X_train, y_train)
predictions = best_model.predict(X_test)
print("Best n_neighbors:", best_n_neighbors)
print("Best Recall:", best_score)
print("Accuracy:", metrics.accuracy_score(y_test, predictions))
precision, recall, fscore, supp = metrics.precision_recall_fscore_support(y_test, predictions,average=None)
print("Precision:", precision)
print("Recall:", recall)
print("F-score:", fscore)

confusion_matrix = metrics.confusion_matrix(y_test, predictions)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix,display_labels = ['No churn','Churn'])
cm_display.plot()
plt.show()


# naive Bayes
# Implementing Naive Bayes with cross validation
model_nb = GaussianNB()

# Define the cross-validation strategy
cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=2)

# Define the parameter grid for the hyperparameter tuning
param_grid = {'var_smoothing': uniform(1e-9, 1e-5)}

# Define the model for hyperparameter tuning
nb_tuned = RandomizedSearchCV(model_nb, param_grid, n_iter=100, cv=cv, scoring='recall')

# Fit the model using the training data
nb_tuned.fit(X_train, y_train)

# Make predictions using the test data
prob_predictions = nb_tuned.predict_proba(X_test)

# Set the threshold
threshold = 0.4  # tested 0.5, 0.4, 0.3, 0.2

# Convert probabilities to binary class predictions
predictions = [1 if prob[1] >= threshold else 0 for prob in prob_predictions]

# Get the accuracy of the model
print("Accuracy:", metrics.accuracy_score(y_test, predictions))

# Get precision, recall, f-score, and support values
precision, recall, fscore, supp = metrics.precision_recall_fscore_support(y_test, predictions, average=None)
print("Precision:", precision)
print("Recall:", recall)
print("F-score:", fscore)

# Plot the confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, predictions)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['No churn', 'Churn'])
cm_display.plot()
plt.show()

