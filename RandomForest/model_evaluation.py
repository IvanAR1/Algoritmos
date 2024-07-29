from model_training import getData, trainingData, modelRandomForest, evaluation
from sklearn.model_selection import GridSearchCV
from sklearn import tree
import matplotlib.pyplot as plt

def gridSearchCV(model:tree.DecisionTreeClassifier, X_train, y_train, param_grid = {
    'n_estimators': [50, 100, 1000],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
})->GridSearchCV:
    return GridSearchCV(estimator=model
                        ,param_grid=param_grid
                        ,cv=5
                        ,n_jobs=-1
                        ,verbose=0
                        ).fit(X_train, y_train)

def plotGridSearchCV(grid_search:GridSearchCV, X_test, y_test)->None:
    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test)
    confusion, matrix = evaluation(y_test, y_pred_best)
    print("Mejores parámetros: ", grid_search.best_params_, "\n")
    print("Matriz de confusión: ", confusion, "\n")
    print("\nReporte de clasificación con mejores parámetros: ", "\n", matrix)

def bestPlotModels(model:modelRandomForest, features:list)->None:
    for i, tree_in_forest in enumerate(model.estimators_):
        plt.figure(figsize=(12, 12))
        tree.plot_tree(tree_in_forest, feature_names=features, filled=True)
        plt.title(f"Árbol {i+1}")
        plt.show()

def example_model_evaluation():
    peoples = getData()
    X = peoples.columns[:-3]
    Y = peoples["user_entry"]
    X_train, X_test, y_train, y_test = trainingData(peoples, X, Y, test_size=0.2, random_state=0)
    model = modelRandomForest(len(peoples))
    grid_search = gridSearchCV(model=model, X_train=X_train, y_train=y_train)
    grid_search.fit(X_train, y_train)
    bestModel = grid_search.best_estimator_
    plotGridSearchCV(grid_search, X_test, y_test)
    bestPlotModels(bestModel, X)
    

example_model_evaluation()