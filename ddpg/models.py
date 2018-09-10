import tensorflow as tf
import tensorflow.contrib as tc


class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            
            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x


class Critic(Model):
    def __init__(self, name='critic', layer_norm=True):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)
            

            # if(len(x.shape) >= 3):
            #     print(x.shape) # 1st run: (?, 220, 64); 2nd & 3rd run: (?, 220, 64)
            #     print(action.shape) # (?, 6); (?, 220, 6)
            #     rowpad = int((x.shape[1] - action.shape[1]) / 2) # (220-6)/2
            #     if(len(action.shape) >= 3):
            #         colpad = int((x.shape[2] - action.shape[2]) / 2) # (64-6)/2
            #         paddings = tf.constant([[0, 0], [rowpad, rowpad], [colpad, colpad]])
            #     else:
            #         colpad = int((x.shape[2] - 1) / 2) # (64-1)/2
            #         action = tf.expand_dims(action, axis=2)
            #         print(action.shape)
            #         paddings = tf.constant([[0, 0], [rowpad, rowpad], [colpad, colpad+1]])
            #     action = tf.pad(action, paddings, "CONSTANT")
            #     print(action.shape)
            # else:
            #     print(x.shape) # (?, 64)
            #     print(action.shape) # (?, 6)
            #     rowpad = int((x.shape[1] - action.shape[1]) / 2) # (64-6)/2
            #     paddings = tf.constant([[0, 0], [rowpad, rowpad]])
            #     action = tf.pad(action, paddings, "CONSTANT")
            #     print(action.shape)
            # print("concatenating")
            x = tf.concat([x, action], axis=-1)
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
