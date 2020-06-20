import tensorflow as tf
from tensorflow.keras import layers

def DenseModel(model_config):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(model_config['neurons'][0], activation='relu',input_dim = model_config['input_dim']))
    for i in range(1,len(model_config['neurons'])):
        model.add(tf.keras.layers.Dense(model_config['neurons'][i], activation='relu'))
    model.add(tf.keras.layers.Dense(model_config['output_dim'], activation='linear'))
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate = model_config['learning_rate'], beta_1 = model_config['beta_1'], beta_2 = model_config['beta_2'], epsilon = model_config['epsilon']))
    return model