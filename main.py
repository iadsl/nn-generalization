import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.datasets import mnist
from tensorflow.keras.applications import VGG19
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, add, GlobalAveragePooling2D, Dense, Flatten

import os

def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

def load_cifar10_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

def load_cifar100_data():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = to_categorical(y_train, 100)
    y_test = to_categorical(y_test, 100)
    return (x_train, y_train), (x_test, y_test)

def vgg19_model(num_classes):
    base_model = VGG19(include_top=False, input_shape=(32, 32, 3), weights='imagenet')
    base_model.trainable = True  # Fine-tune the model
    
    model = tf.keras.Sequential([
        base_model,
        Flatten(),
        Dense(4096, activation='relu'),
        Dense(4096, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def wide_residual_network(num_classes, depth=28, width=10):
    def residual_block(x, filters, increase=False):
        stride = (1, 1)
        if increase:
            stride = (2, 2)
        o1 = Activation('relu')(BatchNormalization()(x))
        conv_1 = Conv2D(filters, (3, 3), strides=stride, padding='same')(o1)
        o2 = Activation('relu')(BatchNormalization()(conv_1))
        conv_2 = Conv2D(filters, (3, 3), padding='same')(o2)
        if increase:
            projection = Conv2D(filters, (1, 1), strides=stride, padding='same')(o1)
            block = add([conv_2, projection])
        else:
            block = add([conv_2, x])
        return block

    n = (depth - 4) // 6
    inputs = Input(shape=(32, 32, 3))
    x = Conv2D(16, (3, 3), padding='same')(inputs)

    for i in range(n):
        x = residual_block(x, 16 * width)
    for i in range(n):
        x = residual_block(x, 32 * width, increase=(i == 0))
    for i in range(n):
        x = residual_block(x, 64 * width, increase=(i == 0))

    x = Activation('relu')(BatchNormalization()(x))
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def lenet5_model():
    model = models.Sequential()
    model.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.AveragePooling2D())
    model.add(layers.Conv2D(16, (5, 5), activation='relu'))
    model.add(layers.AveragePooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dense(84, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def lenet1000_model():
    model = models.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dense(300, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X, y):
    model.fit(X, y, epochs=1, verbose=0)
    return model

def count_activation_frequency(model, X):
    activations = {}
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            intermediate_model = models.Model(inputs=model.input, outputs=layer.output)
            intermediate_output = intermediate_model.predict(X)
            activations[layer.name] = np.mean(intermediate_output > 0, axis=0)
    return activations

def calculate_threshold(frequencies, alpha):
    return np.percentile(frequencies, alpha * 100)

def mark_nodes_for_reinitialization(activations, alpha):
    mem = {}
    for layer_name, freqs in activations.items():
        threshold = calculate_threshold(freqs, alpha)
#         print(freqs[0])
        mem[layer_name] = [i for i, freq in enumerate(freqs) if freq < threshold]
    return mem

def reinitialize_weights(model, mem):
    for layer in model.layers:
        if layer.name in mem and len(mem[layer.name]) > 0:
            weights, biases = layer.get_weights()
            for node_index in mem[layer.name]:
                weights[:, node_index] = np.random.normal(size=weights[:, node_index].shape)
            if biases is not None:
                for node_index in mem[layer.name]:
                    biases[node_index] = np.random.normal(size=biases[node_index].shape)
            layer.set_weights([weights, biases])
    return model

def reduce_memorization_skills(model, X, y, alpha, k, save_dir, model_name):
    iteration_count = 0
    change_count = 1

    while iteration_count < k and change_count > 0:
        change_count = 0
        iteration_count += 1
        
        model = train_model(model, X, y)
        activations = count_activation_frequency(model, X)
        
        mem = mark_nodes_for_reinitialization(activations, alpha)
        
        #Reinitialize all outgoing weights of nodes in Mem
        if any(len(mem[layer_name]) > 0 for layer_name in mem):
            change_count += 1
            model = reinitialize_weights(model, mem)
            
        if iteration_count%10 == 0:
            print(f"epoch:{iteration_count}")

    # Save the final model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path+'.modelsave')
#     print(f"Final model saved at {model_path}")

    return model


def main():
    # Load data and train the model with reduced memorization skills
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    alpha = 0.5
    k = 100
    save_dir = './models'
    model_name = 'lenet5_'
    
    for i in range(1):
        model = lenet5_model()
        model = reduce_memorization_skills(model, x_train, y_train, alpha, k, save_dir, model_name+str(i))
    
        # Evaluate the model
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
        print(f'{test_loss},{test_acc}')


if __name__ == '__main__':
    main()