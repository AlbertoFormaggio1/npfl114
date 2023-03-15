#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import Dict
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
import re

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--cnn", default='CB-6-3-5-valid,F,H-32', type=str, help="CNN architecture.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        # TODO: Create the model. The template uses the functional API, but
        # feel free to use subclassing if you want.
        inputs = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])


        # TODO: Add CNN layers specified by `args.cnn`, which contains
        # a comma-separated list of the following layers:
        # - `C-filters-kernel_size-stride-padding`: Add a convolutional layer with ReLU
        #   activation and specified number of filters, kernel size, stride and padding.
        # - `CB-filters-kernel_size-stride-padding`: Same as `C`, but use batch normalization.
        #   In detail, start with a convolutional layer without bias and activation,
        #   then add a batch normalization layer, and finally the ReLU activation.
        # - `M-pool_size-stride`: Add max pooling with specified size and stride, using
        #   the default "valid" padding.
        # - `R-[layers]`: Add a residual connection. The `layers` contain a specification
        #   of at least one convolutional layer (but not a recursive residual connection `R`).
        #   The input to the `R` layer should be processed sequentially by `layers`, and the
        #   produced output (after the ReLU nonlinearity of the last layer) should be added
        #   to the input (of this `R` layer).
        # - `F`: Flatten inputs. Must appear exactly once in the architecture.
        # - `H-hidden_layer_size`: Add a dense layer with ReLU activation and the specified size.
        # - `D-dropout_rate`: Apply dropout with the given dropout rate.
        # You can assume the resulting network is valid; it is fine to crash if it is not.
        #
        # Produce the results in the variable `hidden`.
        hidden_layers = re.split(r',(?![^\[]*])', args.cnn)
        layer = inputs

        for hidden_layer in hidden_layers:
            layer = self.obtain_layer(hidden_layer, layer)
        hidden = layer

        # Add the final output layer
        outputs = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)(hidden)

        super().__init__(inputs=inputs, outputs=outputs)
        self.compile(
            optimizer=tf.optimizers.Adam(jit_compile=False),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    def obtain_layer(self, parameters, input_layer):
        if parameters[0] != 'R':
            parameters = parameters.split('-')

        if parameters[0] == 'C':
            # Remove all the parameters after the first one
            # Converting all arguments to integer
            [filters, kernel_size, stride] = [eval(i) for i in parameters[1:-1]]
            padding = parameters[-1]
            out = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding=padding,
                                         activation=tf.nn.relu)(input_layer)
            return out
        elif parameters[0] == 'CB':
            # Converting all arguments to integer
            [filters, kernel_size, stride] = [eval(i) for i in parameters[1:-1]]
            padding = parameters[-1]
            # You don't use the bias in a BN layer
            conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding=padding,
                                          use_bias=False)(input_layer)
            # apply normalization between weight application and activation function
            batch_norm = tf.keras.layers.BatchNormalization()(conv)
            # Apply the activation
            out = tf.keras.layers.Activation('relu')(batch_norm)
            return out
        elif parameters[0] == 'M':
            [pool_size, stride] = [eval(i) for i in parameters[1:]]
            return tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=stride, padding='valid')(input_layer)
        elif parameters[0] == 'R':
            layers_str = parameters[3:-1].split(',')
            # Creating the 1st layer (the one to return that will be attached to the network)
            layer = input_layer
            for layer_str in layers_str:
                layer = self.obtain_layer(layer_str, layer)
            output = tf.keras.layers.add([layer, input_layer])
            return output
        elif parameters[0] == 'F':
            out = tf.keras.layers.Flatten()(input_layer)
            return out
        elif parameters[0] == 'H':
            hidden_layer_size = (parameters[1])
            out = tf.keras.layers.Dense(units=hidden_layer_size, activation=tf.nn.relu)(input_layer)
            return out
        elif parameters[0] == 'D':
            rate = eval(parameters[1])
            return tf.keras.layers.Dropout(rate=rate)(input_layer)

def main(args: argparse.Namespace) -> Dict[str, float]:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data
    mnist = MNIST()

    # Create the model and train it
    model = Model(args)
    model.summary()

    logs = model.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[model.tb_callback],
    )

    # Return development metrics for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
