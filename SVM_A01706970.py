#Implementación de un algoritmo de ML sin framework
#José María Ibarra a01706970

#Librerías y métodos
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

#Máquina de soporte vectorial (SVM)

glosses = [] #inicialización de pérdida global del modelo

def hyperplane_function(params, sample):
    """Calculca la función lineal para la clasificación de clases

	Args:
		params (list) parametros de la función
        sample (list) valores de las variables

	Returns:
		cálculo de la predicción de clase
	"""

    result = 0 #se inicializa el valor de salida
    for i in range(len(params)): 
        result += params[i] * sample[i] #obtiene el producto
    result = int(np.sign(result)) #predice la etiqueta con la función sign
    return result

def hloss(prediction, true_label):
    """Calculca la función de pérdida Hinge Loss

	Args:
		prediction (list) etiquetas de valores predichas
        true_label (list) etiquetas de clase verdaderas

	Returns:
		cálculo de Hinge Loss"""
    
    result = max(0, 1 - true_label * prediction) #fórmula de Hinge Loss
    return result

def calculate_loss(params, samples, y, C):
    """Calculca la pérdida por cada teración del promedio

        Args:
            params (list) parámetros actuales 
            samples (list) valores de las variables
            y (list) etiquetas de clase verdaderas
            C (float) parámetro de regularización

        Returns:
            cálculo de pérdida"""

    loss = 0 #se inicaliza la pérdida
    for i in range(len(samples)):
        f_ = hyperplane_function(params, samples[i]) #calcula la predicción para las variables correspondientes
        loss += hloss(f_, y[i]) #calcula la pérdida (acumulada)
    regularization_term = 0 #inicaliza un término de regularización
    for i in range(1, len(params)):
        regularization_term += params[i] * params[i] #actualiza el término de regularización
    loss = loss + C * regularization_term * 0.5 #calcula la pérdida total
    return loss

def gradient(params, sample, y, C):
    """Calculca la pérdida por cada teración del promedio

        Args:
            params (list) parámetros actuales 
            samples (list) valores de las variables
            y (list) etiquetas de clase verdaderas
            C (float) parámetro de regularización

        Returns:
            cálculo de pérdida"""

    f_ = hyperplane_function(params, sample) #obtiene la predicción
    grad = np.zeros_like(params) #genera un arreglo de zeros con forma de los parámetros
    if y * f_ < 1: #actualiza el gradiente solo si la predicición es incorrecta
        for i in range(len(params)):
            grad[i] = -y * sample[i]
    for i in range(1, len(params)): #ajusta el gradiente en general con el parámetro de regularización
        grad[i] += -C * params[i]
    return grad

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
        X = X[y != 2, :2]  
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

    samples_train = [] 
    for i in range(X_train.shape[0]):
        sample = np.concatenate(([1], X_train[i]))
        samples_train.append(sample)

    samples_train = np.array(samples_train) #Añadir bias

    return X_train, y_train, X_test, y_test, samples_train


def run(data, num):
    """Genera una corrida de entrenamiento y evaluación de modelo con el dataset indicado

	Args:
		data (int) numero correspondiendo a uno de los datasets disponibles: iris (1), cancer (2)

	Returns:
		Modelo y evaluación
	"""
    X_train, y_train, X_test, y_test, samples_train = dataset_(data) #genera los datos en formato correcto
    params = np.zeros(X_train.shape[1] + 1) #inizalicaión de arreglo de parámetros
    C = 0.00001 # Regularizacion
    learning_rate = 0.0001 #aprendizaje
    epochs = 0 #inizialicación de núm. d épocas
    max_epochs = 100 #épocas máximas

    print(f'******************** Modelo {data}.{num}: ********************')
    while epochs < max_epochs:
        total_loss = 0
        for i in range(len(samples_train)):
            grad = gradient(params, samples_train[i], y_train[i], C) #gradiente
            params -= learning_rate * grad #actualización de parámetros
            total_loss += calculate_loss(params, samples_train, y_train, C)  #pérdida total
        avg_loss = total_loss / len(samples_train) #promedio de pérdida
        glosses.append(avg_loss) #actualización de pérdida global
        print(f"Epoch {epochs} - Average Loss: {avg_loss}")
        epochs += 1

        #break si la pérdida empieza a aumentar
        if epochs > 1 and glosses[epochs - 1] > glosses[epochs - 2]:
            break

    # Plot
    plt.plot(glosses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epochs")
    plt.show()

    #añadir bias a datos de prueba
    samples_test = []
    for i in range(X_test.shape[0]):
        sample = np.concatenate(([1], X_test[i]))
        samples_test.append(sample)
    samples_test = np.array(samples_test)

    correct_predictions = 0 #inicalización de núm. de preds correctas
    
    #conteo de predicciones correctas
    for i in range(len(samples_test)):
        prediction = np.sign(hyperplane_function(params, samples_test[i]))
        if prediction == y_test[i]:
            correct_predictions += 1

    #cálculo de accuracy
    accuracy = correct_predictions / len(samples_test) * 100
    print(f"Accuracy: {accuracy:.2f}%")

    # Predicciones
    predictions = []

    for i in range(len(samples_test)):
        prediction = np.sign(hyperplane_function(params, samples_test[i]))
        predictions.append(prediction)

    #print("True Labels:", y_test)
    #print("Predictions:", predictions)
    #print("Final Parameters:", params)
    print(f'Métricas:\n {classification_report(y_test, predictions)}')
    cm = confusion_matrix(y_test, predictions)
    # disp = ConfusionMatrixDisplay(cm)
    # disp.plot()
    # plt.show()

    return params

for i in range(1, 4): 
    run(1, i) #Corrida con dataset Iris

for i in range(1,4):
    run(2, i) #Corrida con dataset Cancer