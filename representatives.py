# This collection of functions is meant to modify the dataframe containing information about the generated Neural Networks, save and load models (Normal Keras function had a bug in the new update), 
# and quantify distance between neural network models (by calculating the L2-distance between their weight matrices).

import numpy as np
from numpy import linalg as LA
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

# Add Names of the archtitectures to the original Dataframe
def add_names(main_directory, number_of_Nodes):
    # Write the name of the architecture from the number of nodes
    architecture = "{}x{}x{}".format(number_of_Nodes, 10* number_of_Nodes, number_of_Nodes)
    # Load in the Evaluation data
    eval_df = pd.read_csv("{}/evaluation/{}.csv".format(main_directory, architecture), delim_whitespace=True)
    names = []
    for index in range(eval_df.shape[0]):
        names.append("FF_{}_{}-{}".format(architecture, eval_df['Initializer_Bounds'][index], eval_df['Model_Index'][index]))
    eval_df['Name'] = names
    eval_df.rename(index=str, columns={'Time_Step' : 'Date-Time'}, inplace=True)
    eval_df = eval_df.reindex(columns = ['Name', 'Architecture', 'Initializer_Bounds', 'Model_Index', 'MSE', 'Validation_MSE', 'R^2', 'Explained_Variance', 'Predicted_Value', 'Date-Time'])
    print eval_df
    eval_df.to_csv("{}/evaluation/{}.csv".format(main_directory, architecture), sep='\t', index=False)

# Add Weights of the models to the original Dataframe
def add_weights(main_directory, number_of_Nodes, middle_layer):
    # Write the name of the architecture from the number of nodes
    architecture = "{}x{}x{}".format(number_of_Nodes, 10* number_of_Nodes, number_of_Nodes)
    # Load in the Evaluation data
    eval_df = pd.read_csv("{}/evaluation/{}.csv".format(main_directory, architecture), delim_whitespace=True)
    weights = []
    for index in range(eval_df.shape[0]):
        weights.append(load_model("{}/models/{}".format(main_directory, eval_df['Name'][index])).get_weights())
    eval_df['Weights'] = weights
    eval_df = eval_df.reindex(columns = ['Name', 'Architecture', 'Initializer_Bounds', 'Model_Index', 'MSE', 'Validation_MSE', 'R^2', 'Explained_Variance', 'Predicted_Value', 'Weights', 'Date-Time'])
    print eval_df
    eval_df.to_csv("{}/evaluation/{}.csv".format(main_directory, architecture), sep='\t', index=False)

# Select 20 Networks with the Least Mean Square Error
def select(main_directory, number_of_Nodes, middle_layer):
    # Write the name of the architecture from the number of nodes
    architecture = "{:02}x{:03}x{:02}".format(number_of_Nodes, 10 * middle_layer, number_of_Nodes)
    # Load in the Evaluation data
    eval_df = pd.read_csv("{}/evaluation/set2/{}.csv".format(main_directory, architecture), delim_whitespace=True)
    # Sort the Evaluation data by Mean Square Error from least to greatest. Pick the 20 best performers of the 100
    selected_df = eval_df.sort_values('MSE')[0:20]
    selected_df.reset_index(drop=True, inplace=True)
    # Return the names of the 20 best performers    
    return selected_df

# Load the Models with the Least Mean Square Error
def recover_models(main_directory, selected_df):
    networks = selected_df['Name']
    models = []
    # Load in the networks whose weights you want to average
    print "___________"
    print "Loading Selected Models... "
    for network in networks:
        models.append(load_model("/clusterhome/chaudhry/networks/models/{}".format(network)))
    
    return models
# Retrieve the Weights for the Models with the Least Mean Square Error
def recover_weights(main_directory, selected_df):
    networks = selected_df['Name'][0:5]
    weights = []
    # Load in the networks whose weights you want to average
    print "___________"
    for network in networks:
        print "Loading Selected Models... "
        weights.append(load_model("/clusterhome/chaudhry/networks/models/{}".format(network)).get_weights())
    return weights

# Find the Average of the Weight Matrices
def average(main_directory, models):
    # Sum all of the weight matrices of these networks
    sum_matrix = models[0].get_weights()
    print "___________"
    print "Summing Weight Matrices... "
    for model in models[1:20]:
        sum_matrix = sum_matrix + model.get_weights()

    # Divide every element in the weight matrix by 
    average_weights = [x / len(models) for x in sum_matrix]
    return average_weights

# Find the L2-Distance between the particular model's weights and the average weight matrix
def difference(model, average_weights):
    # Subtract my model's weights from the average weights
    diff_matrix = []
    for index in range(len(model.get_weights())):
        diff_matrix.append(np.subtract(model.get_weights()[index], average_weights[index]))
    diff_matrix = np.array(diff_matrix)
    
    # Calculate the Frobenius Norm (Mean Square Error) of this difference matrix
    square_error = 0
    for row in diff_matrix:
        square_error += np.sum((row)**2)
    distance = np.sqrt(square_error)
    
    return distance

# Find the L2-Distance between two model's weight matrices.
def difference_weights(weights1, weights2):
    # Subtract my model's weights from the average weights
    diff_matrix = []
    for index in range(len(weights1)):
        diff_matrix.append(np.subtract(weights1[index], weights2[index]))
    diff_matrix = np.array(diff_matrix)
    
    # Calculate the Frobenius Norm (Mean Square Error) of this difference matrix
    square_error = 0
    for row in diff_matrix:
        square_error += np.sum((row)**2)
    distance = np.sqrt(square_error)
    
    return distance

# Find the L-2 Distance between two rows of weight matrices.
def difference_layers(layers1, layers2):
    # Subtract my model's weights from the average weights
    diff_matrix = np.subtract(layers1, layers2)
    diff_matrix = np.array(diff_matrix)
    
    # Calculate the Frobenius Norm (Mean Square Error) of this difference matrix
    square_error = 0
    for row in diff_matrix:
        square_error += np.sum((row)**2)
    distance = np.sqrt(square_error)
    
    return distance    

# This was used to find the 5 most differentiated networks by taking their distance from the average weight matrix.
"""
# Find 5 Networks that are farthest away from one another
def varied_representatives(weights, selected_df, average_weights):
    # Compute the Distances from the Average Weight Set
    distance_Matrix = np.empty((20))
    for index, weight in enumerate(weights):
        distance_Matrix[index] = difference_weights(weight, average_weights)
    
    # Sort the Distances from Average
    selected_df['Distance from Average'] = distance_Matrix
    distance_df = selected_df.sort_values('Distance from Average', ascending=False)
        
    # Set the First Representative equal to the one farthest from the Average
    representative_df = distance_df[0]

    for index, weights in enumerate(weights):
        distance_Matrix[index] = difference(model, average_weights)
    

    representative_df.append
    networks = [None] * 5
    print distance_df
    
    return representative_df
"""