import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import pathlib
import tensorflow as tf

tf.get_logger().setLevel('ERROR')


class Info(tf.Module):
    def __init__(
        self, 
        number: float, 
        filepath: str = None,
    ) -> None:
        super(Info, self).__init__()

        self._number = number
        self.get_number.get_concrete_function()

        if filepath:
            self._labelmap_asset = tf.saved_model.Asset(filepath)
            self._labelmap = tf.Variable(
                pathlib.Path(filepath).read_text().splitlines())

            self.get_label.get_concrete_function(
                tf.TensorSpec(shape=[], dtype=tf.int64))
            self.get_labelmap.get_concrete_function()
            self.get_labelmap_asset.get_concrete_function()

    @tf.function
    def get_label(self, index):
        return self._labelmap[index]

    @tf.function
    def get_labelmap(self):
        return self._labelmap

    @tf.function
    def get_labelmap_asset(self):
        return self._labelmap_asset

    @tf.function
    def get_number(self):
        return tf.constant(self._number, dtype=tf.float32)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(8))
model.add(tf.keras.layers.Dense(4))
model.add(tf.keras.layers.Dense(2))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

parser = argparse.ArgumentParser(
    description='TensorFlow 2 SavedModel Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--no_labelmap', action='store_true', 
    help='Do not add labelmap')
args = parser.parse_args()

if args.no_labelmap:
    model.info = Info(number=0.12345)
else:
    model.info = Info(number=0.12345, filepath='./labelmap.txt')

model_dir = './saved_model'
if os.path.isdir(model_dir):
    from shutil import rmtree
    rmtree(model_dir)
tf.saved_model.save(model, model_dir)
