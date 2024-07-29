import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

def getData()->pd.DataFrame:
    return pd.read_csv(os.path.dirname(__file__) + "/files/peoples.csv")

def trainingData(data:pd.DataFrame, features:list, target:str, test_size:float = 0.2, random_state:int = 0)->tuple:
    return train_test_split(data[features], target, test_size=test_size, random_state=random_state)

def modelRandomForest(ext_estimators:int)->RandomForestClassifier:
    return RandomForestClassifier(n_estimators=ext_estimators # Número de 
                                ,criterion='gini' # Criterio de división
                                ,max_features='sqrt' # Número de características a considerar
                                ,bootstrap=True # Muestreo con reemplazo
                                ,max_samples=1/2 # Muestra de entrenamiento
                                ,oob_score=True # Error fuera de la bolsa
                                ,max_depth=2 # Profundidad máxima
                                ,random_state=42 # Semilla
                               )

def predictModel(model:RandomForestClassifier, X_test:pd.DataFrame)->pd.Series:
    return model.predict(X_test)

def precision(model:RandomForestClassifier, X_test:pd.DataFrame, y_test:pd.Series)->float:
    return model.score(X_test, y_test)

def OOB(model:RandomForestClassifier)->float:
    return model.oob_score_

def evaluation(y_test:pd.Series, y_pred:pd.Series)->tuple:
    return confusion_matrix(y_test, y_pred), classification_report(y_test, y_pred)

def ROC(y_test:pd.Series, y_pred:pd.Series)->tuple:
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def plotROC(fpr:list, tpr:list, roc_auc:float)->None:
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example (Curva ROC)')
    plt.legend(loc="lower right")
    plt.show()

def example_model_training():
    data = getData()
    features = data.columns[:-3]
    target = data["user_entry"]
    X_train, X_test, y_train, y_test = trainingData(data, features, target)
    model = modelRandomForest(len(data))
    model.fit(X_train, y_train)
    y_pred = predictModel(model, X_test)
    confusion, matrix = evaluation(y_test, y_pred)
    print("Precisión: ", precision(model, X_test, y_test))
    print("OOB:", OOB(model))
    print("Matriz de confusión: ")
    print(confusion)
    print("Reporte de clasificación: ")
    print(matrix)
    fpr, tpr, roc_auc = ROC(y_test, y_pred)
    plotROC(fpr, tpr, roc_auc)

example_model_training()