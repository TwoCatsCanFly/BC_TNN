# 1) An Empty Network

weight = 0.1
alpha = 0.01

def neural_network(input, weight):
    prediction = input * weight
    return prediction

# 2) PREDICT: Making A Prediction And Evaluating Error

number_of_toes = [8.5] # входящие данные
win_or_lose_binary = [1] # (won!!!) это желаемый результат
input = number_of_toes[0]
goal_pred = win_or_lose_binary[0]

pred = neural_network(input,weight)
error = (pred - goal_pred) ** 2 # COST FUNCTION

# 3) COMPARE: Calculating "Node Delta" and Putting it on the Output Node

delta = pred - goal_pred #на сколько результат отличается от желаемого

# 4) LEARN: Calculating "Weight Delta" and Putting it on the Weight

weight_delta = input * delta

# 5) LEARN: Updating the Weight

weight -= weight_delta * alpha

print(weight)