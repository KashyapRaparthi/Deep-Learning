# %% [markdown]
# **Architecture's Used** \
# -3 Hidden Layers with different no of neurons in each layer \
# -Format of no of neurons in each layer: H1-H2-H3 
# 
# Architecture 1 : 64-128-256\
# Architecture 2 : 128-256-512\
# Architecture 3 : 256-128-64\
# Architecture 4 : 512-256-128

# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf

# %%
#Reading the input files
train=[]
train_label=[]
path=r"C:\Users\Asus\Documents\Python Codes\Assignments\Deep Learning\Assignment-3\Data\\"
for i in os.listdir(path+"train") :
  for j in os.listdir(path+"train\\"+i):
    train.append(cv2.imread(path+"train\\"+i+"\\"+j, cv2.IMREAD_GRAYSCALE))
    train_label.append(int(i))
train=np.array(train)
train_label=np.array(train_label)

test=[]
test_label=[]
for i in os.listdir(path+"test") :
  for j in os.listdir(path+"test\\"+i):
    test.append(cv2.imread(path+"test\\"+i+"\\"+j, cv2.IMREAD_GRAYSCALE))
    test_label.append(int(i))
test=np.array(test)
test_label=np.array(test_label)

validation=[]
validation_label=[]
for i in os.listdir(path+"val") :
  for j in os.listdir(path+"val\\"+i):
    validation.append(cv2.imread(path+"val\\"+i+"\\"+j, cv2.IMREAD_GRAYSCALE))
    validation_label.append(int(i))
validation=np.array(validation)
validation_label=np.array(validation_label)

# %% [markdown]
# # Intializer

# %%
initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=68)

# %% [markdown]
# # **Stochastic Gradient Descent**

# %%
model1 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28),name='Input_layer'),
    tf.keras.layers.Dense(512, activation='sigmoid',kernel_initializer=initializer,name='Hidden_layer-1'),
    tf.keras.layers.Dense(256, activation='sigmoid',kernel_initializer=initializer,name='Hidden_Layer-2'),
    tf.keras.layers.Dense(128, activation='sigmoid',kernel_initializer=initializer,name='Hidden_layer-3'),
    tf.keras.layers.Dense(10, activation='softmax',kernel_initializer=initializer,name='Output_layer')
])

Optimizer1 = tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0,name='SGD')
model1.compile(optimizer=Optimizer1,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
Callback = tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.0001,patience=1)
Predictions_1 = model1.fit(train, train_label, epochs=10000, batch_size=1, callbacks=Callback)

plt.plot(Predictions_1.history['loss'])
plt.title("Stochastic Gradient Descent")
plt.xlabel("Epochs")
plt.ylabel("Average error")
plt.show()

_ , _ = model1.evaluate(train, train_label)
_ , _ = model1.evaluate(validation, validation_label)
_ , _ = model1.evaluate(test, test_label)

Predict = model1.predict(test, verbose=1)
predictions = np.argmax(Predict, axis=1)
tf.math.confusion_matrix(test_label, predictions)

# %% [markdown]
# 

# %% [markdown]
# # **Batch Gradient Descent**

# %%
model2 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28),name='Input_layer'),
    tf.keras.layers.Dense(512, activation='sigmoid',kernel_initializer=initializer,name='Hidden_layer-1'),
    tf.keras.layers.Dense(256, activation='sigmoid',kernel_initializer=initializer,name='Hidden_Layer-2'),
    tf.keras.layers.Dense(128, activation='sigmoid',kernel_initializer=initializer,name='Hidden_layer-3'),
    tf.keras.layers.Dense(10, activation='softmax',kernel_initializer=initializer,name='Output_layer')
])

Optimizer2 = tf.keras.optimizers.SGD(learning_rate=0.001,name='SGD')
model2.compile(optimizer=Optimizer2,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
Callback = tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.0001,patience=1)
Predictions_2 = model2.fit(train, train_label, epochs=100000, batch_size=11385, callbacks=Callback)


plt.plot(Predictions_2.history['loss'])
plt.title("Batch Gradient Descent")
plt.xlabel("Epochs")
plt.ylabel("Average error")
plt.show()


_ , _ = model2.evaluate(train, train_label)
_ , _ = model2.evaluate(validation, validation_label)
_ , _ = model2.evaluate(test, test_label)

Predict = model2.predict(test, verbose=1)
predictions = np.argmax(Predict, axis=1)
tf.math.confusion_matrix(test_label, predictions)


# %% [markdown]
# # **Generalized Delta Rule**

# %%
model3 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28),name='Input_layer'),
    tf.keras.layers.Dense(512, activation='sigmoid',kernel_initializer=initializer,name='Hidden_layer-1'),
    tf.keras.layers.Dense(256, activation='sigmoid',kernel_initializer=initializer,name='Hidden_Layer-2'),
    tf.keras.layers.Dense(128, activation='sigmoid',kernel_initializer=initializer,name='Hidden_layer-3'),
    tf.keras.layers.Dense(10, activation='softmax',kernel_initializer=initializer,name='Output_layer')
])


Optimizer3 = tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.9,name='SGD')
model3.compile(optimizer=Optimizer3,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
Callback = tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.0001,patience=1)
Predictions_3 = model3.fit(train, train_label, epochs=10000, batch_size=11385, callbacks=Callback)


plt.plot(Predictions_3.history['loss'])
plt.title("Generalized Delta Rule")
plt.xlabel("Epochs")
plt.ylabel("Average error")
plt.show()

_ , _ = model3.evaluate(train, train_label)
_ , _ = model3.evaluate(validation, validation_label)
_ , _ = model3.evaluate(test, test_label)

Predict = model3.predict(test, verbose=1)
predictions = np.argmax(Predict, axis=1)
tf.math.confusion_matrix(test_label, predictions)



# %% [markdown]
# # **NAG**

# %%
model4 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28),name='Input_layer'),
    tf.keras.layers.Dense(512, activation='sigmoid',kernel_initializer=initializer,name='Hidden_layer-1'),
    tf.keras.layers.Dense(256, activation='sigmoid',kernel_initializer=initializer,name='Hidden_Layer-2'),
    tf.keras.layers.Dense(128, activation='sigmoid',kernel_initializer=initializer,name='Hidden_layer-3'),
    tf.keras.layers.Dense(10, activation='softmax',kernel_initializer=initializer,name='Output_layer')
])


Optimizer4 = tf.keras.optimizers.SGD(learning_rate=0.001, nesterov=True,momentum=0.7,name='SGD')
model4.compile(optimizer=Optimizer4,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
Callback = tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.0001,patience=1)
Predictions_4 = model4.fit(train, train_label, epochs=10000, batch_size=1, callbacks=Callback)



plt.plot(Predictions_4.history['loss'])
plt.title("SGD_NAG")
plt.xlabel("Epochs")
plt.ylabel("Average error")
plt.show()



_ , _ = model4.evaluate(train, train_label)
_ , _ = model4.evaluate(validation, validation_label)
_ , _ = model4.evaluate(test, test_label)

Predict = model4.predict(test, verbose=1)
predictions = np.argmax(Predict, axis=1)
tf.math.confusion_matrix(test_label, predictions)


# model.summary()

# %% [markdown]
# # **AdaGrad**

# %%
model5 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28),name='Input_layer'),
    tf.keras.layers.Dense(512, activation='sigmoid',kernel_initializer=initializer,name='Hidden_layer-1'),
    tf.keras.layers.Dense(256, activation='sigmoid',kernel_initializer=initializer,name='Hidden_Layer-2'),
    tf.keras.layers.Dense(128, activation='sigmoid',kernel_initializer=initializer,name='Hidden_layer-3'),
    tf.keras.layers.Dense(10, activation='softmax',kernel_initializer=initializer,name='Output_layer')
])


Optimizer5 = tf.keras.optimizers.Adagrad(learning_rate=0.001,name='Adagrad')
model5.compile(optimizer=Optimizer5,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
Callback = tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.0001,patience=1)
Predictions_5 = model5.fit(train, train_label, epochs=10000, batch_size=1, callbacks=Callback)

plt.plot(Predictions_5.history['loss'])
plt.title("AdaGrad")
plt.xlabel("Epochs")
plt.ylabel("Average error")
plt.show()

_ , _ = model5.evaluate(train, train_label)
_ , _ = model5.evaluate(validation, validation_label)
_ , _ = model5.evaluate(test, test_label)

Predict = model5.predict(test, verbose=1)
predictions = np.argmax(Predict, axis=1)
tf.math.confusion_matrix(test_label, predictions)

# %% [markdown]
# # **RMSProp**

# %%
model6 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28),name='Input_layer'),
    tf.keras.layers.Dense(512, activation='sigmoid',kernel_initializer=initializer,name='Hidden_layer-1'),
    tf.keras.layers.Dense(256, activation='sigmoid',kernel_initializer=initializer,name='Hidden_Layer-2'),
    tf.keras.layers.Dense(128, activation='sigmoid',kernel_initializer=initializer,name='Hidden_layer-3'),
    tf.keras.layers.Dense(10, activation='softmax',kernel_initializer=initializer,name='Output_layer')
])


Optimizer6 = tf.keras.optimizers.RMSprop(learning_rate=0.001,rho=0.9,epsilon=1e-08,name='RMSprop')
model6.compile(optimizer=Optimizer6,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
Callback = tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.0001,patience=1)
Predictions_6 = model6.fit(train, train_label, epochs=10000, batch_size=1, callbacks=Callback)

plt.plot(Predictions_6.history['loss'])
plt.title("RMSProp")
plt.xlabel("Epochs")
plt.ylabel("Average error")
plt.show()

_ , _ = model6.evaluate(train, train_label)
_ , _ = model6.evaluate(validation, validation_label)
_ , _ = model6.evaluate(test, test_label)

Predict = model6.predict(test, verbose=1)
predictions = np.argmax(Predict, axis=1)
tf.math.confusion_matrix(test_label, predictions)

# %% [markdown]
# # **Adam**

# %%
model7 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28),name='Input_layer'),
    tf.keras.layers.Dense(512, activation='sigmoid',kernel_initializer=initializer,name='Hidden_layer-1'),
    tf.keras.layers.Dense(256, activation='sigmoid',kernel_initializer=initializer,name='Hidden_Layer-2'),
    tf.keras.layers.Dense(128, activation='sigmoid',kernel_initializer=initializer,name='Hidden_layer-3'),
    tf.keras.layers.Dense(10, activation='softmax',kernel_initializer=initializer,name='Output_layer')
])


Optimizer7 = tf.keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08,name='Adam')
model7.compile(optimizer=Optimizer7,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
Callback = tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.0001,patience=1)
Predictions_7 = model7.fit(train, train_label, epochs=10000, batch_size=1, callbacks=Callback)



plt.plot(Predictions_7.history['loss'])
plt.title("Adam")
plt.xlabel("Epochs")
plt.ylabel("Average error")
plt.show()

_ , _ = model7.evaluate(train, train_label)
_ , _ = model7.evaluate(validation, validation_label)
_ , _ = model7.evaluate(test, test_label)

Predict = model7.predict(test, verbose=1)
predictions = np.argmax(Predict, axis=1)
tf.math.confusion_matrix(test_label, predictions)


# %% [markdown]
# # Plots

# %%
plt.plot(Predictions_1.history["loss"],label="SGD")
plt.plot(Predictions_2.history["loss"],label="SGD-Batch")
plt.plot(Predictions_3.history["loss"],label="SGD-Delta")
plt.plot(Predictions_4.history["loss"],label="SGD-NAG")
plt.plot(Predictions_5.history["loss"],label="AdaGrad")
plt.plot(Predictions_6.history["loss"],label="RMSProp")
plt.plot(Predictions_7.history["loss"],label="Adam")
plt.legend()

# %%
model1.summary()


