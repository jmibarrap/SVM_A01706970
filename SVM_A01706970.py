#Implementación de un algoritmo de ML sin framework
#José María Ibarra a01706970

#Librerías
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

#SVM

glosses = []

def hyperplane_function(params, sample):
    result = 0
    for i in range(len(params)):
        result += params[i] * sample[i]
    return result

def hloss(prediction, true_label):
    return max(0, 1 - true_label * prediction)

def calculate_loss(params, samples, y, C):
    loss = 0
    for i in range(len(samples)):
        f_ = hyperplane_function(params, samples[i])
        loss += hloss(f_, y[i])
    regularization_term = 0
    for i in range(1, len(params)):
        regularization_term += params[i] * params[i]
    loss = loss + C * regularization_term * 0.5
    return loss

def gradient(params, sample, y, C):
    f_ = hyperplane_function(params, sample)
    grad = np.zeros_like(params)
    if y * f_ < 1:
        for i in range(len(params)):
            grad[i] = -y * sample[i]
    for i in range(1, len(params)):
        grad[i] += -C * params[i]
    return grad

#Datos de ejemplo (Iris)
iris = load_iris()
X = iris.data
y = iris.target
#Seleccionar dos clases y dos atributos 
X = X[y != 2, :2]  
y = y[y != 2]
y[y == 0] = -1  # Ajuste de clases (-1, 1)

# Split de datos en test y train
split_ratio = 0.7
split_index = int(split_ratio * len(X))

# Suffle
random_indices = np.random.permutation(len(X))
X_shuffled = X[random_indices]
y_shuffled = y[random_indices]

X_train = X_shuffled[:split_index]
y_train = y_shuffled[:split_index]
X_test = X_shuffled[split_index:]
y_test = y_shuffled[split_index:]

params = np.zeros(X_train.shape[1] + 1)
C = 0.1  # Regularizacion
learning_rate = 0.0005

samples_train = []
for i in range(X_train.shape[0]):
    sample = np.concatenate(([1], X_train[i]))
    samples_train.append(sample)

samples_train = np.array(samples_train) #Añadir bias

epochs = 0
max_epochs = 1000

while epochs < max_epochs:
    total_loss = 0
    for i in range(len(samples_train)):
        grad = gradient(params, samples_train[i], y_train[i], C)
        params -= learning_rate * grad
        total_loss += calculate_loss(params, samples_train, y_train, C)  # Calculate total loss
    avg_loss = total_loss / len(samples_train)
    glosses.append(avg_loss)
    print(f"Epoch {epochs} - Average Loss: {avg_loss}")
    epochs += 1

    #Break si la pérdida empíeza a subir
    if epochs > 1 and glosses[epochs - 1] > glosses[epochs - 2]:
        break

# Plot
plt.plot(glosses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs. Epochs")
plt.show()

# Test
samples_test = []
for i in range(X_test.shape[0]):
    sample = np.concatenate(([1], X_test[i]))
    samples_test.append(sample)

samples_test = np.array(samples_test) #Añadir bias a datos de prueba

correct_predictions = 0

for i in range(len(samples_test)):
    prediction = np.sign(hyperplane_function(params, samples_test[i]))
    if prediction == y_test[i]:
        correct_predictions += 1

accuracy = correct_predictions / len(samples_test) * 100
print(f"Accuracy on test data: {accuracy:.2f}%")

# Predicciones
predictions = []

for i in range(len(samples_test)):
    prediction = np.sign(hyperplane_function(params, samples_test[i]))
    predictions.append(prediction)

print("True Labels:", y_test)
print("Predictions:", predictions)
print("Final Parameters:", params)
