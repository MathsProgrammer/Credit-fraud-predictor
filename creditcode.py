import pandas as pd
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,plot_roc_curve,accuracy_score,recall_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.tree  import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

data = pd.read_csv('creditcard.csv')
data.head()

data.describe().round(2)

data.isnull().sum()  #Checking 0 vals
data.duplicated().sum()  #Checking duplicates

data = data.drop_duplicates()  #Removing duplicates

#Fraud (1) vs legitimate (0), 

fraudornot = data['Class'].value_counts().tolist()
labels = ["99.8% Not fraudulent", "0.02% fraudulent"]

plt.pie(fraudornot,labels = labels)
plt.show()

corr=data.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.5, center=0, linewidths=.5, cbar_kws={"shrink": .5})


sns.kdeplot(data=data[data['Class'] == 0]['V17'], label="Normal", shade=True)
sns.kdeplot(data=data[data['Class'] == 1]['V17'], label="Fraud", shade=True)
plt.legend()
plt.show()


sns.kdeplot(data=data[data['Class'] == 0]['V14'], label="Normal", shade=True)
sns.kdeplot(data=data[data['Class'] == 1]['V14'], label="Fraud", shade=True)
plt.legend()
plt.show()


sns.kdeplot(data=data[data['Class'] == 0]['V12'], label="Normal", shade=True)
sns.kdeplot(data=data[data['Class'] == 1]['V12'], label="Fraud", shade=True)
plt.legend()
plt.show()

#only non reduced and scaled data is time of transaction and amount so do EDA on these

amounts = data['Amount'].values
times = data['Time'].values

sns.distplot(amounts, color='cyan')
plt.title("Distribution of transaction amounts")
plt.xlabel("Transaction value (USD)")
plt.show

sns.distplot(times, color='g')
plt.title("Distribution of transaction times")
plt.xlabel("Time (seconds)")
plt.show()

fraud=data[data['Class']==1]
legit=data[data['Class']==0]

plt.scatter(fraud.Time,fraud.Amount,alpha = 0.5,c="cyan")
plt.title("Fraudulent transaction time vs amount")
plt.xlabel("Time (seconds)")
plt.ylabel("Transaction value (USD)")
plt.show()

plt.scatter(legit.Time,legit.Amount,alpha = 0.5,c = "magenta")
plt.title("Legitimate transaction time vs amount")
plt.xlabel("Time (seconds)")
plt.ylabel("Transaction value (USD)")
plt.show()

scaler = StandardScaler()
data['Scaled_Amount']=scaler.fit_transform(data['Amount'].values.reshape(-1,1))
data.drop(['Amount','Time'],axis=1,inplace=True)

data.head()


X = data.drop('Class', axis = 1)
y = data['Class']

#even out the data 50/50 with smote (Synthetic Minority Oversampling TEchnique)
smote=SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

print(len(X_smote))

plt.title("Balanced data with SMOTE")
labella = ["50% fradulent", "50% legitimate"]
plt.pie([len(X_smote),len(y_smote)],labels = labella )
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size = 0.25, random_state=(2))

classifier=GaussianNB()
classifier.fit(X_train , y_train)
classifier.score(X_test , y_test).round(5)

naive_preds=classifier.predict(X_test)
confusion_matrix(y_test,naive_preds)

plot_confusion_matrix(classifier, X_test, y_test,cmap='plasma',include_values=True)
plt.title('Naive: Confusion Matrix', fontsize=14)
plt.show()

dt =DecisionTreeClassifier(max_features=8 , max_depth=6)
dt.fit(X_train , y_train)
dt.score(X_test , y_test).round(5)

DT_preds=dt.predict(X_test)
confusion_matrix(y_test,DT_preds)

plot_confusion_matrix(dt, X_test, y_test,cmap='cividis')
plt.title('DT: Confusion Matrix', fontsize=14)
plt.show()

lr = LogisticRegression(C = 100)
lr.fit(X_train , y_train)

lr.score(X_test , y_test).round(5)
LR_preds=lr.predict(X_test)
confusion_matrix(y_test,LR_preds)
plot_confusion_matrix(lr, X_test, y_test,cmap='inferno')
plt.title('LogisticRegression: Confusion Matrix', fontsize=14)
plt.show()

Rclf = RandomForestClassifier(max_features=8 , max_depth=6)
Rclf.fit(X_train, y_train)
Rclf.score(X_test, y_test).round(5)

Rclf_preds=Rclf.predict(X_test)
confusion_matrix(y_test,Rclf_preds)

plot_confusion_matrix(Rclf, X_test, y_test,cmap='Blues')
plt.title('RF: Confusion Matrix', fontsize=14)
plt.show()


knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

knn.score(X_test, y_test).round(5)

Knn_preds=knn.predict(X_test)
confusion_matrix(y_test,Knn_preds)

plot_confusion_matrix(knn, X_test, y_test)
plt.title('KNN: Confusion Matrix', fontsize=14)
plt.show()

xgb = XGBClassifier()
xgb.fit(X_train , y_train)
xgb.score(X_test , y_test).round(5)

xgb_preds=xgb.predict(X_test)
confusion_matrix(y_test,xgb_preds)
plot_confusion_matrix(xgb, X_test, y_test,cmap='Purples')
plt.title('XGB: Confusion Matrix', fontsize=14)
plt.show()


####Create dictionary for results, then done