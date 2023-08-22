# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tensorflow import initializers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling


# %% [markdown]
# # Resizing the images to 224 by 224

# %%
# path = r"C:\Users\Asus\Documents\Python Codes\Assignments\Deep Learning\Assignment -5\Data\\"
# output_path = r"C:\Users\Asus\Documents\Python Codes\Assignments\Deep Learning\Assignment -5\Resized_Data\\"

# for i in os.listdir(path):
#     for j in os.listdir(path+i):
#         os.makedirs(output_path+i+'\\'+j)
#         for k in os.listdir(path+i+"\\"+j):
#             image_path = os.path.join(path, i, j, k)  # Get the full image path
#             # print(image_path)
#             image = plt.imread(image_path)
#             resized_image = cv2.resize(image, (224, 224))
#             output_path_1 = os.path.join(output_path,i,j,k)
#             plt.imsave(output_path_1, resized_image)

# %%
#Reading the input files
train=[]
train_label=[]
tr_img=[]
tr_img_labels=[]
path = r"C:\Users\Asus\Documents\Python Codes\Assignments\Deep Learning\Assignment-5\Resized_Data\\"
dict={
  "butterfly" : 0,
  "chandelier" : 1,
  "grand_piano" : 2,
  "Leopards" : 3,
  "revolver" :4
}
for i in os.listdir(path+"train"):
  Flag=0
  for j in os.listdir(path+"train\\"+i):
    img = cv2.imread(path+"train\\"+i+"\\"+j)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    train.append(RGB_img)
    train_label.append(dict[str(i)])
    if(Flag==0):
      img = cv2.imread(path+"train\\"+i+"\\"+j)
      RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      tr_img.append(RGB_img)
      tr_img_labels.append(dict[str(i)])
      Flag=1

train=np.array(train)
train_label=np.array(train_label)
tr_img=np.array(tr_img)
tr_img_labels=np.array(tr_img_labels)

test=[]
test_label=[]
test_img=[]
test_img_labels=[]

for i in os.listdir(path+"test"):
  Flag=0
  for j in os.listdir(path+"test\\"+i):
    img = cv2.imread(path+"test\\"+i+"\\"+j)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    test.append(RGB_img)
    test_label.append(dict[str(i)])
    if(Flag==0):
      img = cv2.imread(path+"test\\"+i+"\\"+j)
      RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      test_img.append(RGB_img)
      test_img_labels.append(dict[str(i)])
      Flag=1

test=np.array(test)
test_label=np.array(test_label)
test_img = np.array(test_img)
test_img_labels = np.array(test_img_labels)

validation=[]
validation_label=[]
val_img=[]
val_img_labels=[]

for i in os.listdir(path+"val"):
  Flag=0
  for j in os.listdir(path+"val\\"+i):
    img = cv2.imread(path+"val\\"+i+"\\"+j)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    validation.append(RGB_img)
    validation_label.append(dict[str(i)])
    if(Flag==0):
      img = cv2.imread(path+"val\\"+i+"\\"+j)
      RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      val_img.append(RGB_img)
      val_img_labels.append(dict[str(i)])
      Flag=1

validation=np.array(validation)
validation_label=np.array(validation_label)
val_img = np.array(val_img)
val_img_labels = np.array(val_img_labels)

train_label = tf.keras.utils.to_categorical(train_label, num_classes=5)
test_label_1 = tf.keras.utils.to_categorical(test_label, num_classes=5)
validation_label = tf.keras.utils.to_categorical(validation_label, num_classes=5)


# %% [markdown]
# # Architecture 1

# %%
# input_shape = (224, 224, 3)
# tf.random.set_seed(64)
# initializer = initializers.RandomNormal(mean=0.0, stddev=0.05)
# model = tf.keras.models.Sequential([

#     Rescaling(1./255, input_shape=input_shape, name='rescaling'),

#     tf.keras.layers.Conv2D(filters=8, kernel_size=11, strides=4, padding='valid', activation='relu',name="1st_Conv_Layer"),
#     tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),

#     tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='valid', activation='relu',name="2nd_Conv_Layer"),
#     tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),

#     tf.keras.layers.Flatten(),

#     tf.keras.layers.Dense(units=128, activation='relu'),
#     tf.keras.layers.Dense(units=5, activation='softmax')
# ])
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Callback = tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.0001,patience=5)
# history = model.fit(train,train_label,verbose=1,callbacks=Callback,epochs=50)

# val_acc = model.evaluate(validation, validation_label)[1]
# test_acc = model.evaluate(test,test_label_1)[1]
# Predict = model.predict(test, verbose=0)
# predictions = np.argmax(Predict, axis=1)
# cm = tf.math.confusion_matrix(test_label, predictions)
# print("/n",cm)
# print("Validation accuracy : ", val_acc)
# print("Test Accuracy : ",test_acc)

# model.save("Architecture-1.h5")

# %% [markdown]
# # Architecture 2

# %%
# input_shape = (224, 224, 3)
# tf.random.set_seed(64)
# initializer = initializers.RandomNormal(mean=0.0, stddev=0.05)
# model = tf.keras.models.Sequential([

#     Rescaling(1./255, input_shape=input_shape, name='rescaling'),

#     tf.keras.layers.Conv2D(filters=8, kernel_size=11, strides=4, padding='valid', activation='relu',name="1st_Conv_Layer"),
#     tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),

#     tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='valid', activation='relu',name="2nd_Conv_Layer"),
#     tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),

#     tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='valid', activation='relu',name="2nd_Conv_Layer"),
#     tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),

#     tf.keras.layers.Flatten(),

#     tf.keras.layers.Dense(units=128, activation='relu'),
#     tf.keras.layers.Dense(units=5, activation='softmax')
# ])
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Callback = tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.0001,patience=5)
# history = model.fit(train,train_label,verbose=1,callbacks=Callback,epochs=50)

# val_acc = model.evaluate(validation, validation_label)[1]
# test_acc = model.evaluate(test,test_label_1)[1]
# Predict = model.predict(test, verbose=0)
# predictions = np.argmax(Predict, axis=1)
# cm = tf.math.confusion_matrix(test_label, predictions)
# print("/n",cm)
# print("Validation accuracy : ", val_acc)
# print("Test Accuracy : ",test_acc)

# model.save("Architecture-2.h5")

# %% [markdown]
# # Architecture 3
# ## Best architecture based on Validation Accuracy

# %%
input_shape = (224, 224, 3)
tf.random.set_seed(64)
initializer = initializers.RandomNormal(mean=0.0, stddev=0.05)
model = tf.keras.models.Sequential([

    Rescaling(1./255, input_shape=input_shape, name='rescaling'),

    tf.keras.layers.Conv2D(filters=8, kernel_size=11, strides=4, padding='valid', activation='relu',name="1st_Conv_Layer"),
    tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),

    tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='valid', activation='relu',name="2nd_Conv_Layer"),
    tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),

    tf.keras.layers.Conv2D(filters=32,kernel_size=3,strides=1, padding='valid', activation='relu',name="3rd_Conv_Layer"),
    
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu',name="4th_Conv_Layer"),
    tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=5, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
Callback = tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.0001,patience=5)
history = model.fit(train,train_label,verbose=1,callbacks=Callback,epochs=50)

val_acc = model.evaluate(validation, validation_label)[1]
test_acc = model.evaluate(test,test_label_1)[1]
Predict = model.predict(test, verbose=0)
predictions = np.argmax(Predict, axis=1)
cm = tf.math.confusion_matrix(test_label, predictions)
print("/n",cm)
print("Validation accuracy : ", val_acc)
print("Test Accuracy : ",test_acc)

model.save("Architecture-3.h5")

# %%
model.summary()

# %%
model.layers

# %% [markdown]
# # Feature Maps of Convolutional Layer

# %%
img = train[8]

plt.imshow(img)
plt.axis("off")
plt.show()

layer_outputs = [i.output for i in model.layers]
Intermediate_model = tf.keras.models.Model(inputs=model.input,outputs=layer_outputs)
feature_maps = Intermediate_model.predict(np.array([img]))

for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(feature_maps[1][0,:, :, i])
    plt.axis("off")
plt.tight_layout()
plt.show()

# layer_outputs = [layer.output for layer in model.layers[5:]] 
# activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
# activations = activation_model.predict(np.array([img]))


for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(feature_maps[3][0][:, :, i])
    plt.axis("off")
plt.tight_layout()
plt.show()


for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(feature_maps[6][0][:, :, i])
    plt.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# # VGG19

# %%
# from tensorflow.keras.applications import VGG19
# from tensorflow.keras.layers import Flatten, Dense
# from tensorflow.keras.models import Model

# # Loading the VGG19 model
# base_model = VGG19(input_shape=(224,224,3),weights="imagenet", include_top=False)

# # Adding classification layer at the end
# x = Flatten()(base_model.output)
# x = Dense(5, activation="softmax")(x)

# # Combining the model
# model = Model(inputs=base_model.input, outputs=x, name='Modified_VGG19')

# # not training the base model weights
# for layer in base_model.layers:
#     layer.trainable = False


# %%
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Callback = tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.0001,patience=1)
# history = model.fit(train,train_label,verbose=1,callbacks=Callback,epochs=50)

# val_acc = model.evaluate(validation, validation_label)[1]
# test_acc = model.evaluate(test,test_label_1)[1]
# Predict = model.predict(test, verbose=0)
# predictions = np.argmax(Predict, axis=1)
# cm = tf.math.confusion_matrix(test_label, predictions)
# print("/n",cm)
# print("Validation accuracy : ", val_acc)
# print("Test Accuracy : ",test_acc)

# model.save("Modified_VGG19.h5")


