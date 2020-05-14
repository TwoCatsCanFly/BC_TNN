import numpy as np
np.random.seed(1)

def relu(x):
    return (x>0)*x
    #прикольная запись, если х>0 то возвращает х если нет то 0

def relu2deriv(output):
    return output>0
    #прикольная запись, если output>0 то возвращает 1
    # если нет то 0





streetlights = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1],
                         [1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1],
                         [1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1],
                         [1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1]])

walk_vs_stop = np.array([[ 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]]).T

alpha = 0.2
hidden_size = 4 # количество скрытых нейронов
weights_0_1 = 2*np.random.random((3,hidden_size)) -1
# 3x4 matrix
weights_1_2 = 2*np.random.random((hidden_size,1)) -1
# 4x1 matrix



for iteration in range(60):
   layer_2_error = 0
   for i in range(len(streetlights)):

      layer_0 = streetlights[i:i+1]
      # same as input from before выдает[[x]]

      layer_1 = relu(np.dot(layer_0,weights_0_1))
      # same as input AND prediction from before
      # выдает значения 1го слоя, если значение уходит в минус
      # то туда суется 0

      layer_2 = np.dot(layer_1,weights_1_2)
      # same as prediction from before
      # последний слой сети, считается как обычно

      layer_2_error += np.sum((layer_2 - walk_vs_stop[i:i+1]) ** 2)
      # Полная COST FUNCTION

      layer_2_delta = (layer_2 - walk_vs_stop[i:i+1])
      # backpropegation - Layer_2_delta backpropagated to layer_1_delta.
      # Gives us a weighting of how much each weight contributed to that error.

      layer_1_delta=layer_2_delta.dot(weights_1_2.T)*relu2deriv(layer_1)
      # *relu2deriv(layer_1) - если нейрон нулевой то и изменения не помогут
      # значит в дельну суем 0

      weights_1_2 -= alpha * layer_1.T.dot(layer_2_delta)
      # умножаем 1х1 матрицу на 4х1 матрицу и получаем веса 4х1
      weights_0_1 -= alpha * layer_0.T.dot(layer_1_delta)
      # умножаем 1х3 матрицу на 4х1 матрицу и получаем веса 4х3

print("Weights 1 to 2:\n" + str(weights_1_2))
print("Weights 0 to 1:\n" + str(weights_0_1))


right = 0

for i in range(len(streetlights)):
    layer_0 = streetlights[i:i+1]
    layer_1 = relu(np.dot(layer_0,weights_0_1))
    layer_2 = np.dot(layer_1,weights_1_2)
    a=0
    if layer_2[0][0]<0.1: a=0
    elif layer_2[0][0]>0.98: a=1
    print('Facts: ',walk_vs_stop[i],'    output: ', round(layer_2[0][0]))
    if walk_vs_stop[i]==a:
        right+=1

print('Правильно: {}\nВсего попыток: {}\nТочность {}%'.format(str(right), len(streetlights),(right/len(streetlights))*100))

