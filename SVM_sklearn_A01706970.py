#Implementación de un algoritmo de ML con framework
#José María Ibarra a01706970

#Librerías y métodos
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

#Máquina de soporte vectorial (SVM)

iris = load_iris()
cancer = load_breast_cancer()

#Datos de ejemplo (Iris)
def dataset_(dataset_num):
    """Prepara el dataeset para entrenamiento 

	Args:
		dataset (int) numero correspondiendo a uno de los datasets disponibles: iris (1), cancer (2)

	Returns:
		Dataset preparado
	"""
    datasets_dict = {1:load_iris(), 2:load_breast_cancer()}
    dataset = datasets_dict[dataset_num]
    X = dataset.data
    y = dataset.target

    if dataset_num == 1: #Seleccionar dos clases y dos atributos
        X = X[y != 2, :]  
        y = y[y != 2]
    
    y[y == 0] = -1  # Ajuste de clases (-1, 1)
    
    # Split de datos en test y train
    split_ratio = 0.7
    split_index = int(split_ratio * len(X))

    # Shuffle
    random_indices = np.random.permutation(len(X))
    X_shuffled = X[random_indices]
    y_shuffled = y[random_indices]

    X_train = X_shuffled[:split_index]
    y_train = y_shuffled[:split_index]
    X_test = X_shuffled[split_index:]
    y_test = y_shuffled[split_index:]

    #añadir bias a train
    samples_train = [] 
    for i in range(X_train.shape[0]):
        sample = np.concatenate(([1], X_train[i]))
        samples_train.append(sample)

    X_train = np.array(samples_train)

    #añadir bias a test
    samples_test = []
    for i in range(X_test.shape[0]):
        sample = np.concatenate(([1], X_test[i]))
        samples_test.append(sample)
    X_test = np.array(samples_test)

    return X_train, y_train, X_test, y_test

def run(data):
    """Genera una corrida de entrenamiento y evaluación de modelo con el dataset indicado

	Args:
		data (int) numero correspondiendo a uno de los datasets disponibles: iris (1), cancer (2)

	Yields:
		Evaluación del modelo
	"""

    #pipeline de modelo
    model = make_pipeline(
        StandardScaler(),
        LinearSVC(penalty='l2' #regularización L2, 
                  ,loss='hinge', dual=True,
                  C = 0.001, fit_intercept=True,
                  random_state=10, max_iter=2000)
    )

    #limpiar dataset
    X_train, y_train, X_test, y_test = dataset_(data)

    #fit (entrenamiento) del modelo
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #cross validation
    k=10 #folds
    cross_val = cross_val_score(model, X_train, y_train, cv=k) 
    print(f'{k}-fold cross-validation score: {cross_val}')
    print(f'Mean cv score: {cross_val.mean()}')

    #evaluación del modelo
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.show()

#corridas
run(1)
run(2)