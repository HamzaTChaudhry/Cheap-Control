import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow import keras


def preprocess(directory):
    # Import and Label Motion Data. Drop Z Coordinates.
    motion_df = pd.read_csv(directory + "/worm_motion_log.txt", delim_whitespace=True, header=None)
    motion_df.rename(columns={motion_df.columns[0] : 'Time_Step', motion_df.columns[1] : 'X', motion_df.columns[101] : 'Y', motion_df.columns[201] : 'Z'}, inplace=True)
    for pos in range(99):
        motion_df.rename(columns={motion_df.columns[pos+2] : 'X_' + str(pos)}, inplace=True)
        motion_df.rename(columns={motion_df.columns[pos+102] : 'Y_' + str(pos)}, inplace=True)
        motion_df.rename(columns={motion_df.columns[pos+202] : 'Z_' + str(pos)}, inplace=True)
    motion_df.drop(motion_df.columns[201:301], axis=1, inplace=True)
    motion = motion_df.values


    # Create Array with Normalized Relative Vectors for every other point along the midline.
    Time, Positions = motion.shape[0], 95
    L = np.zeros((Time,Positions,2))
    R = np.zeros((Time,Positions,2))
    for t in range(Time):
        for pos in range(Positions):
            L[t,pos,0] = (motion[t,  2+(pos)]   - motion[t,  2+(pos+2)]) / LA.norm([motion[t,2+(pos)]   - motion[t,2+(pos+2)], motion[t,102+(pos)]   - motion[t,102+(pos+2)]])
            L[t,pos,1] = (motion[t,102+(pos)]   - motion[t,102+(pos+2)]) / LA.norm([motion[t,2+(pos)]   - motion[t,2+(pos+2)], motion[t,102+(pos)]   - motion[t,102+(pos+2)]])
            R[t,pos,0] = (motion[t,  2+(pos+4)] - motion[t,  2+(pos+2)]) / LA.norm([motion[t,2+(pos+4)] - motion[t,2+(pos+2)], motion[t,102+(pos+4)] - motion[t,102+(pos+2)]])
            R[t,pos,1] = (motion[t,102+(pos+4)] - motion[t,102+(pos+2)]) / LA.norm([motion[t,2+(pos+4)] - motion[t,2+(pos+2)], motion[t,102+(pos+4)] - motion[t,102+(pos+2)]])

    # Inverse Kinematics to Derive Angles Between Positions
    angles = np.zeros((Time,Positions))
    for t in range(Time):
        for pos in range(Positions):
            c_Ang = abs(L[t,pos,0:2].dot(R[t,pos,0:2]))
            s_Ang = np.cross(L[t,pos,0:2],R[t,pos,0:2])
            if s_Ang > 0:
                angles[t,pos] =   (1 - c_Ang)
            else:
                angles[t,pos] = - (1 - c_Ang)

    # Create More Training Data by Adding Gaussian Noise
    mu, sigma = 0, 0.001

    noise1 = np.random.normal(mu, sigma, [Time,Positions])
    noise2 = np.random.normal(mu, sigma, [Time,Positions])
    noise3 = np.random.normal(mu, sigma, [Time,Positions])
    noise4 = np.random.normal(mu, sigma, [Time,Positions])

    angles_noise1 = angles + noise1
    angles_noise2 = angles + noise2
    angles_noise3 = angles + noise3
    angles_noise4 = angles + noise4

    # Import and Label Muscle Data
    muscles_df = pd.read_table(directory + "/muscles_activity_buffer.txt", delim_whitespace=True, header=None)
    muscles = muscles_df.values

    # Create Training and Test Sets.

    angles_train        = np.zeros((Time/2,Positions))
    angles_test         = np.zeros((Time/2,Positions))
    muscles_train       = np.zeros((Time/2,96))
    muscles_test        = np.zeros((Time/2,96))
    
    angles_noise1_train = np.zeros((Time/2,Positions))
    angles_noise2_train = np.zeros((Time/2,Positions))
    angles_noise3_train = np.zeros((Time/2,Positions))
    angles_noise4_train = np.zeros((Time/2,Positions))
    
    sample_interval = 20

    for i in range((Time/2) / sample_interval):
        angles_train[sample_interval * (i) : sample_interval * (i+1), : ]       = angles[sample_interval  * (2*i)  : sample_interval * (2*i+1), : ]
        angles_test[sample_interval * (i) : sample_interval * (i+1),  : ]       = angles[sample_interval  * (2*i+1): sample_interval * (2*i+2), : ]
        muscles_train[sample_interval * (i) : sample_interval * (i+1),: ]       = muscles[sample_interval * (2*i)  : sample_interval * (2*i+1), : ]
        muscles_test[sample_interval * (i) : sample_interval * (i+1), : ]       = muscles[sample_interval * (2*i+1): sample_interval * (2*i+2), : ]
        
        angles_noise1_train[sample_interval * (i) : sample_interval * (i+1),:]  = angles_noise1[sample_interval  * (2*i)  : sample_interval * (2*i+1), : ]
        angles_noise2_train[sample_interval * (i) : sample_interval * (i+1),:]  = angles_noise2[sample_interval  * (2*i)  : sample_interval * (2*i+1), : ]
        angles_noise3_train[sample_interval * (i) : sample_interval * (i+1),:]  = angles_noise3[sample_interval  * (2*i)  : sample_interval * (2*i+1), : ]
        angles_noise4_train[sample_interval * (i) : sample_interval * (i+1),:]  = angles_noise4[sample_interval  * (2*i)  : sample_interval * (2*i+1), : ]
        
    angles_train_large  = np.vstack((angles_train,angles_noise1_train,angles_noise2_train,angles_noise3_train,angles_noise4_train))
    muscles_train_large = np.vstack((muscles_train,muscles_train,muscles_train,muscles_train,muscles_train))
    
    training_set = [angles_train_large,muscles_train_large]
    return list(training_set)

def construct_feedforward_network(training_set, number_of_InterNeurons, name):
    # 99 input nodes for postural information of the worm's midline.
    network = keras.Sequential()
    # 4 interneurons reducing the dimensionality of the movement.
    network.add(keras.layers.Dense(4, activation='tanh', input_dim = 95))
    # 50 interneurons allowing for universal approximation.
    network.add(keras.layers.Dense(number_of_InterNeurons, activation='tanh'))
    # 4 interneurons responsible for motor control.
    network.add(keras.layers.Dense(4, activation='tanh'))
    # 96 output nodes for muscle activation values.
    network.add(keras.layers.Dense(96, activation='sigmoid'))

    # Configuring the Learning Process
    network.compile(optimizer='Adam',
                loss='mse')

    # Train the network, iterating on the data in batches of 500 samples
    network.fit(training_set[0], training_set[1], epochs=250, batch_size = 500)

    # Save the Network
    network.save("/clusterhome/chaudhry/openworm/networks/" + name)

