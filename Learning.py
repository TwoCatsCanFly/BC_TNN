import numpy as np

weights = np.array([0.7,0.2,-0.5])
alfa = 0.1
streetlights = np.array([[0,0,1],
                         [0,1,1],
                         [0,0,1],
                         [1,1,1],
                         [0,1,1],
                         [1,0,1]])

walk_vs_stop = np.array([0,1,0,1,1,0])

for iteration in range(40):
    error_for_all_lights = 0
    for row_index in range(len(walk_vs_stop)):
        input = streetlights[row_index]
        goal_prediction = walk_vs_stop[row_index]
        prediction = input.dot(weights)
        error = (prediction - goal_prediction) ** 2# COST FUNCTION
        error_for_all_lights += error# FULL COST FUNCTION
        delta = prediction - goal_prediction
        weights = weights - (alfa * (input * delta))
        # (input * delta) - Влияние входа на результат
        # если ошибка большая то соответствующий вес понижается,
        # если маленькая - повышается
        print('Prediction: '+str(prediction))
    print('Weights: ' + str(weights))
    print('Error: ' + str(error))