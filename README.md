# Model Zoo: MNIST example

This repo serves as an example of how to download, run, and modify machine
learning models that are part of the Simula Model Zoo. The accompanying
[notebook](run_model.ipynb) goes through the steps in detail, but the
following snippets are enough to get going:

### 1) Install requirements

```shell
wget https://raw.githubusercontent.com/smaeland/model-zoo-mnist-example/main/requirements.txt
pip install -r requirements.txt
```

### 2) Download and load the model 

```shell
wget https://github.com/smaeland/model-zoo-mnist-example/raw/main/cnn-model-v1.zip
unzip cnn-model-v1.zip
python
```

```python
>>> import tensorflow as tf
>>> model = tf.keras.models.load_model('cnn-model-v1')
```

### 3) Download example data, and preprocess it

```python
>>> (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
>>> 
>>> x_train = x_train.astype("float32") / 255
>>> x_test = x_test.astype("float32") / 255
>>> 
>>> x_train = tf.expand_dims(x_train, axis=-1)
>>> x_test = tf.expand_dims(x_test, axis=-1)
>>> 
>>> y_train = tf.keras.utils.to_categorical(y_train)
>>> y_test = tf.keras.utils.to_categorical(y_test)
```

### 4) Make predictions

```python
>>> preds = model.predict(x_test)
>>> acc = tf.keras.metrics.CategoricalAccuracy()
>>> acc.update_state(y_test, preds)
>>> print('Accuracy:', acc.result().numpy())
Accuracy: 0.991
```

