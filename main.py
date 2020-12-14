import numpy as np
from NeuralNetwork import NeuralNetwork


if __name__ == '__main__':
    neural_network = NeuralNetwork()
    print("synaptic_weights before training :")
    print(neural_network.synaptic_weights)
    training_inputs = np.array([[1, 0, 1],
                                [0, 1, 1],
                                [1, 1, 1],
                                [0, 1, 1]])
    training_outputs = np.array([[1, 0, 1, 0]]).T
    neural_network.train(training_inputs, training_outputs, 20000)
    print("synaptic_weights after training :")
    print(neural_network.synaptic_weights)
    print("enter inputs :")
    print("input 1 :")
    input1 = input()
    print("input 2 :")
    input2 = input()
    print("input 3 :")
    input3 = input()
    print("neural network predicted output :")
    print(neural_network.predict(np.array([input1, input2, input3 ])))
