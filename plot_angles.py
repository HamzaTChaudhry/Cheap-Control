import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

# Load in Worm Motion Data
simulation = 'trial_1_halfres'

angles_train  = np.load('/clusterhome/chaudhry/experiments/data/' + simulation + '/angles_train.npy')
muscles_train  = np.load('/clusterhome/chaudhry/experiments/data/' + simulation + '/muscles_train.npy')
angles_test  = np.load('/clusterhome/chaudhry/experiments/data/' + simulation + '/angles_test.npy')
muscles_test  = np.load('/clusterhome/chaudhry/experiments/data/' + simulation + '/muscles_test.npy')

angles_train = angles_train * 10
muscles_train = muscles_train
angles_test = angles_test * 10
muscles_test = muscles_test

np.save('/clusterhome/chaudhry/experiments/data/' + 'trial_1_halfres_scaled' + '/angles_train', angles_train)
np.save('/clusterhome/chaudhry/experiments/data/' + 'trial_1_halfres_scaled' + '/muscles_train', muscles_train)
np.save('/clusterhome/chaudhry/experiments/data/' + 'trial_1_halfres_scaled' + '/angles_test', angles_test)
np.save('/clusterhome/chaudhry/experiments/data/' + 'trial_1_halfres_scaled' + '/muscles_test', muscles_test)

# motion_df = pd.read_csv('/clusterhome/chaudhry/openworm/simulations/{}/worm_motion_log.txt'.format(simulation), delim_whitespace=True, header=None)

# # Reorganize Dataframe
# motion_df.rename(columns={motion_df.columns[0] : 'Time_Step', motion_df.columns[1] : 'X', motion_df.columns[101] : 'Y', motion_df.columns[201] : 'Z'}, inplace=True)
# for pos in range(99):
#     motion_df.rename(columns={motion_df.columns[pos+2] : 'X_' + str(pos)}, inplace=True)
#     motion_df.rename(columns={motion_df.columns[pos+102] : 'Y_' + str(pos)}, inplace=True)
#     motion_df.rename(columns={motion_df.columns[pos+202] : 'Z_' + str(pos)}, inplace=True)

# # Drop Z Coordinates
# motion_df.drop(motion_df.columns[201:301], axis=1, inplace=True)

# # Drop First 1000 data points since these only represent the worm's initialization
# motion_df.drop(motion_df.index[0:1000], inplace=True)

# Time = len(motion_df)
# Positions = 95

# # Create Array with Normalized Relative Vectors for every other point along the midline.
# L = np.zeros((Time,Positions,2))
# R = np.zeros((Time,Positions,2))
# motion = motion_df.values

# for t in range(Time):
#     for pos in range(Positions):
#         L[t,pos,0] = (motion[t,  2+(pos)]   - motion[t,  2+(pos+2)]) / LA.norm([motion[t,2+(pos)]   - motion[t,2+(pos+2)], motion[t,102+(pos)]   - motion[t,102+(pos+2)]])
#         L[t,pos,1] = (motion[t,102+(pos)]   - motion[t,102+(pos+2)]) / LA.norm([motion[t,2+(pos)]   - motion[t,2+(pos+2)], motion[t,102+(pos)]   - motion[t,102+(pos+2)]])
#         R[t,pos,0] = (motion[t,  2+(pos+4)] - motion[t,  2+(pos+2)]) / LA.norm([motion[t,2+(pos+4)] - motion[t,2+(pos+2)], motion[t,102+(pos+4)] - motion[t,102+(pos+2)]])
#         R[t,pos,1] = (motion[t,102+(pos+4)] - motion[t,102+(pos+2)]) / LA.norm([motion[t,2+(pos+4)] - motion[t,2+(pos+2)], motion[t,102+(pos+4)] - motion[t,102+(pos+2)]])


# # Inverse Kinematics to Derive Angles Between Positions
# angles = np.zeros((Time,Positions))
# c_Ang = np.zeros((Time,Positions))
# s_Ang = np.zeros((Time,Positions))

# for t in range(Time):
#     for pos in range(Positions):
#         c_Ang = abs(L[t,pos,0:2].dot(R[t,pos,0:2]))
#         s_Ang = np.cross(L[t,pos,0:2],R[t,pos,0:2])
#         if s_Ang > 0:
#             angles[t,pos] =   (1 - c_Ang)
#         else:
#             angles[t,pos] = - (1 - c_Ang)

# Create More Training Data by Adding Gaussian Noise
# mu, sigma = 0, 0.025
# noise1 = np.random.normal(mu, sigma, [Time,Positions])
# noise2 = np.random.normal(mu, sigma, [Time,Positions])
# noise3 = np.random.normal(mu, sigma, [Time,Positions])
# noise4 = np.random.normal(mu, sigma, [Time,Positions])
# angles_noise1 = angles + noise1
# angles_noise2 = angles + noise2
# angles_noise3 = angles + noise3
# angles_noise4 = angles + noise4

np.max(angles) - np.min(angles)
np.argwhere(angles == np.max(angles))
np.argwhere(angles == np.min(angles))


plt.figure(1)
plt.plot(angles)

# plt.figure(2)
# plt.plot(angles_noise1)

plt.show()
