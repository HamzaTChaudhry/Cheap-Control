import numpy as np
from numpy import linalg as LA

main_directory = "/clusterhome/chaudhry/experiments/"
input_data = "trial_1_halfres_reduced"

input_path = "{}/data/{}".format(main_directory, input_data)
# Load in the Training Data
reduced_angles_train = np.load(input_path + '/angles_train.npy')
reduced_muscles_train = np.load(input_path + '/muscles_train.npy')
# Load in the Test Data
reduced_angles_test = np.load(input_path + '/angles_test.npy')
reduced_muscles_test = np.load(input_path + '/muscles_test.npy')

input_data = "trial_1_halfres_shifted"
input_path = "{}/data/{}".format(main_directory, input_data)
# Load in the Training Data
shifted_angles_train = np.load(input_path + '/angles_train.npy')
shifted_muscles_train = np.load(input_path + '/muscles_train.npy')
# Load in the Test Data
shifted_angles_test = np.load(input_path + '/angles_test.npy')
shifted_muscles_test = np.load(input_path + '/muscles_test.npy')

reduced_muscles_train[50][0]
shifted_muscles_train[0][0]