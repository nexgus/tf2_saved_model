from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf


model_dir = './saved_model'
model = tf.saved_model.load(model_dir)

number = model.info.get_number().numpy()
print(f'number: {number}')

asset = model.info.get_labelmap_asset().numpy()
print(f'asset: {asset.decode("utf-8")}')

labelmap = [label.decode('utf-8') for label in model.info.get_labelmap().numpy()]
print(f'labelmap: {labelmap})')

for i in range(3):
    label = model.info.get_label(i).numpy().decode('utf-8')
    print(f'label {i}: {label}')
