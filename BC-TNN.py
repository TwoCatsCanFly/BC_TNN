weight =[[0.1,0.1,-0.3], #веса для каждой связи нейрона след. слоя с передыдущим
         [0.1,0.2,0.0],
         [0.0,1.3,0.1]]

alfa = 0.01 # костыль для градиентного спуска

def neural_network(inp,weight):
    prediction = vect_mat_mul(inp,weight)#умножаем матрицу весов
    # на матрицу входящих данных
    return prediction

toes = [8.5,9.5,9.9,9.0]
wlrec = [0.65,0.8,0.8,0.9]
nfans = [1.2,1.3,0.5,1.0]

inp = [toes[0],wlrec[0],nfans[0]]

def vect_mat_mul(vect,matrix):
    output = [0] * len(vect)
    for i in range(len(vect)):
        output[i] = w_sum(vect, matrix[i])# сует расчитанные значения
        # следующего нейронав выходной слой
    return output

def w_sum(a,b):
    assert (len(a) == len(b))#проверка на допустимость умножения матриц,
    # к-во строк должно быть одинаковое для умножения, иначе нахуй пойдешь
    output = 0
    for i in range(len(a)):#здесь вычисляем значение каждого нейрона следующего слоя,
        # пока что без сигмоида))
        output+=(a[i]*b[i])# перемножает входящие данные на каждый вес
    return output


pred = neural_network(inp,weight)
print(pred)