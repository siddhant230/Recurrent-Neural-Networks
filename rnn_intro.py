import numpy as np
import pandas as pd
from tqdm import tqdm
import random

class RNN:
    def __init__(self, inp_list, lr=0.072):
        self.err = []
        self.inp = inp_list

        ###weight
        self.weights = []
        for i in range(len(inp_list) - 1):
            w = np.random.normal(0.0, pow(inp_list[i + 1], -0.5), (inp_list[i + 1], inp_list[i]))
            self.weights.append(w)

        #####biases
        self.biases = []
        for j in range(len(inp_list) - 1):
            b = np.random.normal(0.0, pow(inp_list[j + 1], -0.5), (inp_list[j + 1], 1))
            self.biases.append(b)

        ###activation function###
        self.lr = lr

    def dsigmoid(self, x):
        return x * (1 - x)

    def sigmoid(self, z):
        return (1 / (1 + np.exp(-z)))

    def train(self, input_vector, expected_vector, previous_vector=[]):

        output_list = []
        input_vector = np.array(input_vector)
        input_vector = input_vector.reshape(len(input_vector), 1)

        previous_vector = np.array(previous_vector)
        previous_vector = previous_vector.reshape(len(previous_vector), 1)

        input_vector = input_vector + previous_vector

        expected_vector = np.array(expected_vector)
        expected_vector = expected_vector.reshape(len(expected_vector), 1)

        for layer_idx in range(len(self.weights)):
            output_list.append(input_vector)
            input_vector = self.sigmoid(np.dot(self.weights[layer_idx], input_vector) + self.biases[layer_idx])

        output_list.append(input_vector)
        predicted_vector = output_list[-1]
        error = expected_vector-predicted_vector


        for i in range(len(self.weights) - 1, -1, -1):
            grad_w = self.lr * np.dot((error * self.dsigmoid(output_list[i + 1])), np.transpose(output_list[i]))
            self.weights[i] = self.weights[i] + grad_w
            grad_b = self.lr * (error * self.dsigmoid(output_list[i + 1]))
            self.biases[i] = self.biases[i] + grad_b
            error = np.dot(np.transpose(self.weights[i]), error)

        return output_list[-1]

    def predict(self, input_vector):
        output_list = []
        input_vector = np.array(input_vector)
        input_vector = input_vector.reshape(len(input_vector), 1)
        for layer_idx in range(len(self.weights)):
            input_vector = self.sigmoid(np.dot(self.weights[layer_idx], input_vector) + self.biases[layer_idx])
            output_list.append(input_vector)

        return output_list[-1]


if __name__ == '__main__':

    foods = ['Apple Pie', 'Burger', 'Chicken']
    foods = pd.Series(foods)
    epochs = 1000

    one_hot_encoded_foods = list(pd.get_dummies(foods).values)
    obj = RNN([len(one_hot_encoded_foods[0]), 3, len(one_hot_encoded_foods[0])])
    previous_output = [0, 0, 0]

    print('BEFORE TRAINING : ')
    print(obj.predict(one_hot_encoded_foods[0]), one_hot_encoded_foods[1],'\n')
    print(obj.predict(one_hot_encoded_foods[1]), one_hot_encoded_foods[2],'\n')
    print(obj.predict(one_hot_encoded_foods[2]), one_hot_encoded_foods[0],'\n')

    for epoch in range(epochs):
        for each_food_index in range(len(one_hot_encoded_foods)):

            index=random.randint(0,2)
            input_vector = one_hot_encoded_foods[index]
            expected_vector = one_hot_encoded_foods[(index + 1) % len(one_hot_encoded_foods)]
            if len(previous_output) == 0:
                previous_output=obj.train(input_vector=input_vector, expected_vector=expected_vector,previous_vector=previous_output)
            else:
                previous_output=obj.train(input_vector=input_vector, expected_vector=expected_vector,previous_vector=previous_output)

    print('AFTER TRAINING : ')
    print(obj.predict(one_hot_encoded_foods[0]), one_hot_encoded_foods[1],'\n')
    print(obj.predict(one_hot_encoded_foods[1]), one_hot_encoded_foods[2],'\n')
    print(obj.predict(one_hot_encoded_foods[2]), one_hot_encoded_foods[0],'\n')
