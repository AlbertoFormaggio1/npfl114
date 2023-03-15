#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
from typing import Dict, Tuple

from cifar10 import CIFAR10

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=256, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=200, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--augment", default='tf_image', type=str)
parser.add_argument("--learning-rate", default=0.05, type=float)
parser.add_argument("--initial_filters", default=16, type=int)


class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        self.kernel_size = 3

        inputs = tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])
        layer = inputs

        #VGG layers
        starting_filters = args.initial_filters
        for i in range(4):
            filter_num = starting_filters * 2**i
            for j in range(2):
                layer = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3), padding='same',
                                               kernel_initializer='he_uniform')(layer)
                layer = tf.keras.layers.BatchNormalization()(layer)
                layer = tf.keras.layers.Activation(tf.nn.relu)(layer)

            layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(layer)

        layer = tf.keras.layers.Flatten()(layer)
        layer = tf.keras.layers.Dense(units=128, activation='relu', kernel_initializer='he_uniform')(layer)

        # Add the final output layer
        outputs = tf.keras.layers.Dense(len(CIFAR10.LABELS), activation=tf.nn.softmax)(layer)

        super().__init__(inputs=inputs, outputs=outputs)

        steps = 45000 // args.batch_size * args.epochs
        la = args.learning_rate
        ela = 0.0005
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(la, steps, alpha=ela/la, name=None)

        self.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=args.momentum),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)



def main(args: argparse.Namespace) -> None:
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

    # Load data
    cifar = CIFAR10()

    train = tf.data.Dataset.from_tensor_slices((cifar.train.data['images'], cifar.train.data['labels']))
    dev = tf.data.Dataset.from_tensor_slices((cifar.dev.data['images'], cifar.dev.data['labels']))

    def image_to_float(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return tf.image.convert_image_dtype(image, tf.float32), label

    # Simple data augmentation using `tf.image`.
    generator = tf.random.Generator.from_seed(args.seed)

    def train_augment_tf_image(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if generator.uniform([]) >= 0.5:
            image = tf.image.flip_left_right(image)
        image = tf.image.resize_with_crop_or_pad(image, CIFAR10.H + 6, CIFAR10.W + 6)
        image = tf.image.resize(image, [generator.uniform([], CIFAR10.H, CIFAR10.H + 12 + 1, dtype=tf.int32),
                                        generator.uniform([], CIFAR10.W, CIFAR10.W + 12 + 1, dtype=tf.int32)])
        image = tf.image.crop_to_bounding_box(
            image, target_height=CIFAR10.H, target_width=CIFAR10.W,
            offset_height=generator.uniform([], maxval=tf.shape(image)[0] - CIFAR10.H + 1, dtype=tf.int32),
            offset_width=generator.uniform([], maxval=tf.shape(image)[1] - CIFAR10.W + 1, dtype=tf.int32),
        )
        return image, label

    # Simple data augmentation using layers.
    def train_augment_layers(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.keras.layers.RandomFlip("horizontal", seed=args.seed)(
            image)  # Bug, flip always; fixed in TF 2.12.
        image = tf.keras.layers.RandomZoom(0.2, seed=args.seed)(image)
        image = tf.keras.layers.RandomTranslation(0.15, 0.15, seed=args.seed)(image)
        image = tf.keras.layers.RandomRotation(0.1, seed=args.seed)(image)  # Does not always help (too blurry?).
        return image, label

    train = train.map(image_to_float)

    train = train.shuffle(len(train))
    train = train.map(train_augment_tf_image)
    #train = train.map(train_augment_layers)
    train = train.batch(args.batch_size)
    train.prefetch(tf.data.AUTOTUNE)

    dev = dev.map(image_to_float)
    dev = dev.batch(args.batch_size)
    dev = dev.prefetch(tf.data.AUTOTUNE)

    model = Model(args)
    model.summary()

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=6, restore_best_weights=True)

    lr_drop = 10
    def lr_scheduler(epoch):
        return args.learning_rate * (10 ** (-(epoch // lr_drop)))
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    logs = model.fit(
        train,
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=dev,
        callbacks=[model.tb_callback]
    )

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    model.save(os.path.join(args.logdir, 'VGG_tfimg_bs256'))
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as predictions_file:
        test = np.array([tf.image.convert_image_dtype(x, tf.float32) for x in cifar.test.data['images']])
        for probs in model.predict(test, batch_size=args.batch_size):
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
