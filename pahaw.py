# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# spliting libraries
from sklearn.model_selection import train_test_split
# feature scaling
from sklearn.preprocessing import StandardScaler
# shuffle
from sklearn.utils import shuffle
# model
from sklearn.linear_model import LogisticRegression
# confusion matrix
from sklearn import metrics
from sklearn.metrics import confusion_matrix
# accuracy score
from sklearn.metrics import accuracy_score
# classification report
from sklearn.metrics import classification_report


# import dataset
dataset = pd.read_csv('dataset/dataset.csv')

# str to categorical
dataset['Sex'] = pd.Series(dataset['Sex']).astype('category').cat.codes
dataset['Nationality'] = pd.Series(dataset['Nationality']).astype('category').cat.codes
dataset['Disease'] = pd.Series(dataset['Disease']).astype('category').cat.codes
dataset['Dominant hand'] = pd.Series(dataset['Dominant hand']).astype('category').cat.codes

# drop id, PD status, UPDRS V
dataset.drop(['id', 'PD status', 'UPDRS V'],inplace=True, axis=1)

# get X and y
y = dataset['Disease'].values.astype(np.int8)
X = dataset.drop(['Disease'], axis=1).values.astype(np.float32)



# NaN to 0
nan_cols = [i for i in range(X.shape[1]) if np.isnan(X[:, i]).any()]
X[:, nan_cols] = np.nan_to_num(X[:, nan_cols])


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# shuffle data
X_train, y_train = shuffle(X_train, y_train, random_state=0)


# Fitting Logistic Regression to the Training set
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

print(f"______________________Testing Set______________________")
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Accuracy Score
print(accuracy_score(y_test, y_pred))

# Classification Report
print(classification_report(y_test, y_pred))

# # ROC curve
# y_pred_proba = classifier.predict_proba(X_test)[::,1]
# fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
# auc = metrics.roc_auc_score(y_test, y_pred_proba)
# plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
# plt.legend(loc=4)
# plt.show()


print(f"______________________Full dataset Set______________________")
# scale full dataset
X_full = sc.transform(X)

# predict full dataset
y_pred_full = classifier.predict(X_full)

# Making the Confusion Matrix
cm = confusion_matrix(y, y_pred_full)
print(cm)

# Accuracy Score
print(accuracy_score(y, y_pred_full))

# Classification Report
print(classification_report(y, y_pred_full))

# ROC curve
y_pred_proba = classifier.predict_proba(X_full)[::,1]
fpr, tpr, _ = metrics.roc_curve(y,  y_pred_proba)
auc = metrics.roc_auc_score(y, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
