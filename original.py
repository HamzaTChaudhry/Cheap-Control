import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow import keras
from keras import initializers

# Import and Label Motion Data. Drop Z Coordinates.
motion_df = pd.read_csv('/clusterhome/chaudhry/openworm/simulations/trial_2_crawling/worm_motion_log.txt', delim_whitespace=True, header=None)

motion_df.rename(columns={motion_df.columns[0] : 'Time_Step', motion_df.columns[1] : 'X', motion_df.columns[101] : 'Y', motion_df.columns[201] : 'Z'}, inplace=True)
for pos in range(99):
    motion_df.rename(columns={motion_df.columns[pos+2] : 'X_' + str(pos)}, inplace=True)
    motion_df.rename(columns={motion_df.columns[pos+102] : 'Y_' + str(pos)}, inplace=True)
    motion_df.rename(columns={motion_df.columns[pos+202] : 'Z_' + str(pos)}, inplace=True)

motion_df.drop(motion_df.columns[201:301], axis=1, inplace=True)

Time = len(motion_df)
Positions = 95

# Create Array with Normalized Relative Vectors for every other point along the midline.
L = np.zeros((Time,Positions,2))
R = np.zeros((Time,Positions,2))
motion = motion_df.values

for t in range(Time):
    for pos in range(Positions):
        L[t,pos,0] = (motion[t,  2+(pos)]   - motion[t,  2+(pos+2)]) / LA.norm([motion[t,2+(pos)]   - motion[t,2+(pos+2)], motion[t,102+(pos)]   - motion[t,102+(pos+2)]])
        L[t,pos,1] = (motion[t,102+(pos)]   - motion[t,102+(pos+2)]) / LA.norm([motion[t,2+(pos)]   - motion[t,2+(pos+2)], motion[t,102+(pos)]   - motion[t,102+(pos+2)]])
        R[t,pos,0] = (motion[t,  2+(pos+4)] - motion[t,  2+(pos+2)]) / LA.norm([motion[t,2+(pos+4)] - motion[t,2+(pos+2)], motion[t,102+(pos+4)] - motion[t,102+(pos+2)]])
        R[t,pos,1] = (motion[t,102+(pos+4)] - motion[t,102+(pos+2)]) / LA.norm([motion[t,2+(pos+4)] - motion[t,2+(pos+2)], motion[t,102+(pos+4)] - motion[t,102+(pos+2)]])


# Inverse Kinematics to Derive Angles Between Positions
angles = np.zeros((Time,Positions))
c_Ang = np.zeros((Time,Positions))
s_Ang = np.zeros((Time,Positions))

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

muscles_df = pd.read_table("/clusterhome/chaudhry/openworm/simulations/full_resolution_crawling_5sec/muscles_activity_buffer.txt", delim_whitespace=True, header=None)
muscles = muscles_df.values

#muscles_df = pd.concat([motion_df['Time_Step'],muscles_df], axis=1)

# Create Training and Test Sets.
# Still need to sample randomly.

angles_train        = np.zeros((Time/2,Positions))
angles_noise1_train = np.zeros((Time/2,Positions))
angles_noise2_train = np.zeros((Time/2,Positions))
angles_noise3_train = np.zeros((Time/2,Positions))
angles_noise4_train = np.zeros((Time/2,Positions))
angles_test         = np.zeros((Time/2,Positions))
muscles_train       = np.zeros((Time/2,96))
muscles_test        = np.zeros((Time/2,96))

for i in range(250):
    angles_noise1_train[20 * (i) : 20 * (i+1),:]  = angles_noise1[20  * (2*i)  : 20 * (2*i+1), : ]
    angles_noise2_train[20 * (i) : 20 * (i+1),:]  = angles_noise2[20  * (2*i)  : 20 * (2*i+1), : ]
    angles_noise3_train[20 * (i) : 20 * (i+1),:]  = angles_noise3[20  * (2*i)  : 20 * (2*i+1), : ]
    angles_noise4_train[20 * (i) : 20 * (i+1),:]  = angles_noise4[20  * (2*i)  : 20 * (2*i+1), : ]
    angles_train[20 * (i) : 20 * (i+1), : ]       = angles[20  * (2*i)  : 20 * (2*i+1), : ]
    angles_test[20 * (i) : 20 * (i+1),  : ]       = angles[20  * (2*i+1): 20 * (2*i+2), : ]
    muscles_train[20 * (i) : 20 * (i+1),: ]       = muscles[20 * (2*i)  : 20 * (2*i+1), : ]
    muscles_test[20 * (i) : 20 * (i+1), : ]       = muscles[20 * (2*i+1): 20 * (2*i+2), : ]

angles_train_large  = np.vstack((angles_train,angles_noise1_train,angles_noise2_train,angles_noise3_train,angles_noise4_train))
muscles_train_large = np.vstack((muscles_train,muscles_train,muscles_train,muscles_train,muscles_train))


# Constructing Feedforward Neural Network Model
# 99 input nodes for postural information of the worm's midline.
num = 20
for index in range(5):
    network = keras.Sequential()
    # 4 interneurons reducing the dimensionality of the movement.
    network.add(keras.layers.Dense(num   , activation='tanh'   , kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.5),  input_dim = Positions))
    # 50 interneurons allowing for universal approximation.
    network.add(keras.layers.Dense(num*10, activation='tanh'   , kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.5)))
    # 4 interneurons responsible for motor control.
    network.add(keras.layers.Dense(num   , activation='tanh'   , kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.5)))
    # 96 output nodes for muscle activation values.
    network.add(keras.layers.Dense(96    , activation='sigmoid', kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.5)))

    # Configuring the Learning Process
    network.compile(optimizer='Adam',
                loss='mse')

    # Train the network, iterating on the data in batches of 500 samples
    network.fit(angles_train_large, muscles_train_large, epochs=250, batch_size = 500)
    network.save('/clusterhome/chaudhry/openworm/networks/SampleNetwork_' + str(index))

# Evaluate the network
SampleNetwork_0   = tf.keras.models.load_model("/clusterhome/chaudhry/openworm/networks/SampleNetwork_0")
SampleNetwork_1   = tf.keras.models.load_model("/clusterhome/chaudhry/openworm/networks/SampleNetwork_1")
SampleNetwork_2   = tf.keras.models.load_model("/clusterhome/chaudhry/openworm/networks/SampleNetwork_2")
SampleNetwork_3   = tf.keras.models.load_model("/clusterhome/chaudhry/openworm/networks/SampleNetwork_3")
SampleNetwork_4   = tf.keras.models.load_model("/clusterhome/chaudhry/openworm/networks/SampleNetwork_4")

NetworkList = [SampleNetwork_0, SampleNetwork_1, SampleNetwork_2, SampleNetwork_3, SampleNetwork_4]

error = np.zeros(5)

for index, network in enumerate(NetworkList):
    error[index] = network.evaluate(angles_test, muscles_test, batch_size = 500)

plt.plot(error)
plt.show()

norm_list = np.zeros((10))

norm_sq = 0.0
for i in range(8):
    norm_sq += LA.norm(list(np.subtract(SampleNetwork_2.get_weights(),SampleNetwork_3.get_weights()))[i]) **2
np.sqrt(norm_sq)



plt.figure(1)
plt.plot(muscles_test)
plt.figure(2)
plt.plot(SampleNetwork_0.predict(angles_test))
plt.figure(3)
plt.plot(SampleNetwork_1.predict(angles_test))
plt.figure(4)
plt.plot(SampleNetwork_2.predict(angles_test))
plt.figure(5)
plt.plot(SampleNetwork_3.predict(angles_test))
plt.figure(6)
plt.plot(SampleNetwork_4.predict(angles_test))
plt.show()


