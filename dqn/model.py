import tensorflow as tf

from tensorflow import keras

class Agent(keras.Model):
    def __init__(self, action_shape, number_topic):
        super().__init__()
        init = tf.keras.initializers.HeUniform()

        self.dense1 = keras.layers.Dense(1024, activation=tf.nn.tanh, kernel_initializer=init)
        self.dense2 = keras.layers.Dense(512, activation=tf.nn.tanh, kernel_initializer=init)
        # self.dense3 = keras.layers.Dense(256, activation=tf.nn.tanh, kernel_initializer=init)
        self.dense4 = keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init)
        self.topic_embedding = keras.layers.Embedding(number_topic, 8, input_length=1, trainable=False)


    def call(self, inputs):
        (observation, topic_ids) = inputs
        topic_embedding = self.topic_embedding(topic_ids)

        x = self.dense1(observation)
        x = keras.layers.Concatenate(axis=1)([x, topic_embedding])
        x = self.dense2(x)
        # x = self.dense3(x)
        output = self.dense4(x)
        return output