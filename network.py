import numpy as np
from tensorflow import keras
from keras import layers, initializers, optimizers, losses, metrics, models
from keras import backend as K
from keras.models import Sequential, load_model
from keras.callbacks import TensorBoard
from sklearn.metrics import r2_score, explained_variance_score

# Construct FeedForward Neural Network that will reproduce the natural behavior of the worm.
def construct(main_directory, input_data, set_number, number_of_Nodes, middle_layer, model_index):
    number_of_Positions = 95
    architecture = "FF_{:02}x{:03}x{:02}".format(number_of_Nodes, 10 * middle_layer, number_of_Nodes)
    
    input_path = "{}/data/{}".format(main_directory, input_data)
    model_name = "{}-{:02}".format(architecture, model_index)
    output_path = "{}/models/set{}/{}".format(main_directory, set_number, model_name)

    # Log the Training process of the data
    tensorboard = TensorBoard(log_dir="{}/logs/set{}/{}".format(main_directory, set_number, model_name))

    # 99 input nodes for postural information of the worm's midline (a line tracing the middle of the worm that tells us its posture at every time-step).
    network = Sequential()
    # N interneurons reducing the dimensionality of the worm's posture/shape. This corresponds to mapping the worm's shape being represented by up to 4 eigenworms.
    network.add(layers.Dense(number_of_Nodes     , 
                            kernel_initializer = initializers.RandomUniform(minval=-1, maxval=1),
                            bias_initializer = initializers.RandomUniform(minval=-1, maxval = 1),
                            activation='tanh',
                            input_dim = number_of_Positions))
    # Varying Number of interneurons allowing for recombination of the worm's posture. I intuitively understand this layer as permitting universal approximation while the N-node layers correspond to an information bottleneck.
    network.add(layers.Dense(10 * middle_layer, 
                            kernel_initializer = initializers.RandomUniform(minval=-1, maxval=1),
                            bias_initializer = initializers.RandomUniform(minval=-1, maxval = 1),
                            activation='tanh'))
    # N interneurons corresponding to the future behavior. In behavior space, an entity's momentum vector will have the same dimensions as its position vector.
    network.add(layers.Dense(number_of_Nodes     , 
                            kernel_initializer = initializers.RandomUniform(minval=-1, maxval=1),
                            bias_initializer = initializers.RandomUniform(minval=-1, maxval = 1),
                            activation='tanh'))
    # 96 output nodes for muscle activation values. We are training model on expected muscle activation values since the physics simulation takes incredibly long, so we cannot train it directly on the behavior as we'd like.
    network.add(layers.Dense(96                  , 
                            kernel_initializer = initializers.RandomUniform(minval=-1, maxval=1),
                            bias_initializer = initializers.RandomUniform(minval=-1, maxval = 1),
                            activation='sigmoid'))

    # Configuring the Learning Process. The Adam operator is used 
    network.compile(optimizer='adam', loss='mean_squared_error')

    # Load in the Training Data
    angles_train = np.load(input_path + '/angles_train.npy')
    muscles_train = np.load(input_path + '/muscles_train.npy')

    # Load in the Test Data
    angles_test = np.load(input_path + '/angles_test.npy')
    muscles_test = np.load(input_path + '/muscles_test.npy')
    
    # Train the network with 1000 Epochs, iterating on the data in batches of 1000 samples
    network.fit(angles_train, muscles_train, epochs=1000, batch_size = 1000, validation_data=(angles_test, muscles_test), callbacks=[tensorboard], verbose=1)

    # Save Network Using TensorFlow. 
    network.save(output_path)

    # Save Network Weights and Architectures independently. This is because the new Keras update has made saving and loading models funky and desyncs the simulation. It took me so long to debug this. 
    network.save_weights('{}/models/set{}/weights/{}.h5'.format(main_directory, set_number, model_name))
    with open('{}/models/set{}/architectures/{}.json'.format(main_directory, set_number, model_name), 'w') as f:
        f.write(network.to_json())

    return network

# Evaluate the Neural Network's performance in reproducing the desired muscle activation patterns with various statistical measures.
def evaluate(main_directory, input_data, network):
    input_path = "{}/data/{}".format(main_directory, input_data)
    # Load in the Training Data
    angles_train = np.load(input_path + '/angles_train.npy')
    muscles_train = np.load(input_path + '/muscles_train.npy')
    # Load in the Test Data
    angles_test = np.load(input_path + '/angles_test.npy')
    muscles_test = np.load(input_path + '/muscles_test.npy')

    MSE            = network.evaluate(angles_train, muscles_train, batch_size = 500)
    validation_MSE = network.evaluate(angles_test , muscles_test , batch_size = 500) 
    prediction     = network.predict(angles_test)
    explained_var  = explained_variance_score(muscles_test, prediction)
    r2             = r2_score(muscles_test, prediction)
    weights        = network.get_weights()

    return [MSE, validation_MSE, r2, explained_var, prediction, weights]