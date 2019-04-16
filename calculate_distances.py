import pandas as pd
import numpy as np
import tensorflow as tf
from numpy import linalg as LA
import matplotlib.pyplot as plt

# This function is meant to calculate and visualize various measures of performance between the produced muscle activation patterns and desired muscle activation patterns.
def calculate_distance(main_directory, input_data, network):
    input_path = "{}/data/{}".format(main_directory, input_data)
    # Load in the Training Data
    angles_train = np.load(input_path + '/angles_train.npy')
    muscles_train = np.load(input_path + '/muscles_train.npy')
    # Load in the Test Data
    angles_test = np.load(input_path + '/angles_test.npy')
    muscles_test = np.load(input_path + '/muscles_test.npy')

    network = tf.keras.models.load_model('{}/models/set8/{}'.format(main_directory,network), compile=False)
    network.compile(optimizer='adam', loss='mean_squared_error')

    muscles_network = network.predict(angles_train)
    muscles_train.shape
    muscles_network.shape

    L2_TF         = network.evaluate(angles_train, muscles_train)
    print 'TensorFlow L2 Distance = ' + str(L2_TF)

    L1     = np.average(LA.norm(muscles_train - muscles_network, axis = 1, ord = 1))
    print 'Average L1 Distance = ' + str(L1)

    L2     = np.average(LA.norm(muscles_train - muscles_network, axis = 1, ord = 2))
    print 'Average L2 Distance = ' + str(L2)

    L3            =  np.average(LA.norm(muscles_train - muscles_network, axis = 1, ord = 3))
    print 'Average L3 Distance = ' + str(L3)
   
    L4            =  np.average(LA.norm(muscles_train - muscles_network, axis = 1, ord = 4))
    print 'Average L4 Distance = ' + str(L4)
   
    L5            =  np.average(LA.norm(muscles_train - muscles_network, axis = 1, ord = 5))
    print 'Average L5 Distance = ' + str(L5)

    L20            =  np.average(LA.norm(muscles_train - muscles_network, axis = 1, ord = 20))
    print 'Average L20 Distance = ' + str(L20)

    Linf          =  np.average(LA.norm(muscles_train - muscles_network, axis = 1, ord = np.inf))
    print 'Average Linf Distance = ' + str(Linf)

    L2_max          = np.amax(LA.norm(muscles_train - muscles_network, axis = 1, ord = 2))
    print 'Maximum L2 Distance = ' + str(L2_max)
    L2_min          = np.amin(LA.norm(muscles_train - muscles_network, axis = 1, ord = 2))
    print 'Minimum L2 Distance = ' + str(L2_min)

    return [L1,L2,L3,L4,L5,L20,Linf,L2_min,L2_max]
    
main_directory = "/clusterhome/chaudhry/experiments/"
input_data = "trial_1_halfres_reduced"
distance_vector=np.zeros((5,9))

for idx,network in enumerate(['FF_01x020x01-87', 'FF_02x030x02-57', 'FF_03x020x03-47','FF_04x050x04-55','FF_05x010x05-86']):
    distance_vector[idx] = calculate_distance(main_directory, input_data, network)

plt.figure(1)
for idx,network in enumerate(['FF_01x020x01-87', 'FF_02x030x02-57', 'FF_03x020x03-47','FF_04x050x04-55','FF_05x010x05-86']):
    plt.plot(['L1','L2','L3','L4','L5','L20','Linf'], distance_vector[idx][:7])
plt.xlabel('Distance Function')
plt.ylabel('Distance')
plt.legend(['FF_01x020x01-87', 'FF_02x030x02-57', 'FF_03x020x03-47','FF_04x050x04-55','FF_05x010x05-86'], title = 'Network', loc='upper right')

plt.figure(2)
plt.plot(['FF_01x020x01-87', 'FF_02x030x02-57', 'FF_03x020x03-47','FF_04x050x04-55','FF_05x010x05-86'], [distance_vector[idx][-1] for idx in range(5)])
plt.plot(['FF_01x020x01-87', 'FF_02x030x02-57', 'FF_03x020x03-47','FF_04x050x04-55','FF_05x010x05-86'], [distance_vector[idx][-2] for idx in range(5)])
plt.plot(['FF_01x020x01-87', 'FF_02x030x02-57', 'FF_03x020x03-47','FF_04x050x04-55','FF_05x010x05-86'], [distance_vector[idx][1] for idx in range(5)])
plt.xlabel('Network')
plt.ylabel('L2 Distance')
plt.ylabel('L2 Distance')
plt.legend(['Maximum', 'Minimum', 'Average'], title = 'Network', loc='upper right')
plt.show()

