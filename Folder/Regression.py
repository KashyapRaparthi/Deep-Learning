# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# %% [markdown]
# # Univariate Data

# %%
data = np.loadtxt("C:/#MTECH COURSE/Semester 2/DL/Data for A1/Group12/Regression/UnivariateData/12.csv" , delimiter = ",")
# print(data)
print(data.shape)

# %%
x = data[: , 0].reshape(-1,1)
y = data[: , 1].reshape(-1,1)
# print(x)
# print(y)
df_x = pd.DataFrame(x)
# df_x
df_y = pd.DataFrame(y)
df = pd.DataFrame(data)
df = df.sample(frac = 1)
df

# %%
def splitting(df , factor = 0.7):
    no_train_data = 1001*0.7
    training = df[:int(no_train_data)]
    testing = df[700:]
    x_train = training[:][0]
    y_train = training[:][1]
    x_test = testing[:][0]
    y_test = testing[:][1]
    return x_train , y_train , x_test , y_test

# %%
x_train , y_train , x_test , y_test = splitting(df , factor = 0.7)
plt.title('Training and Testing data')
plt.scatter(x_train , y_train, color = 'blue' , label = 'Training data')
plt.scatter(x_test , y_test , color = 'r' , label = 'Testing data')
plt.xlabel('Input values')
plt.ylabel('Output values')
plt.grid()
plt.legend()
# plt.axis([0,1.1,0,4.2])
plt.show()

# %%
x_train_arr = x_train.to_numpy()
y_train_arr = y_train.to_numpy()
x_test_arr = x_test.to_numpy()
y_test_arr = y_test.to_numpy()

# %%
def grad_desc(x , w , eta , yn , sn):
    return w + eta*(yn - sn)*x

# %%
def error(yn , sn):
    err = 0.5*(yn - sn)**2
    return err

# %%
def average_error(error_list):
    return sum(error_list)/len(error_list)

# %%
def mean_square_error(y , y1):
    mse = 0
    for i in range(len(y)):
        mse += (1/len(y))*((y[i] - y1[i])**2)
    return mse

# %%
def run_train(x , y):
    w = np.random.randn(2)
    max_epoches = 100
    eta = 0.0001
    average_error_list = []
    while(max_epoches):
        error_list = []
        for i in range(len(x)):
            xn = x[i]
            sn = np.dot(w.T , [1 , xn])
            yn = y[i]
            err = error(yn , sn)
            error_list.append(err)
            w = grad_desc(xn , w , eta , yn , sn)
        max_epoches = max_epoches - 1
        avg_error = average_error(error_list)
        average_error_list.append(avg_error)
    # print(error_list) 
    return w , average_error_list

# %%
w_train , average_error_list_train = run_train(x_train_arr , y_train_arr)

# %%
plt.plot(average_error_list_train)
# plt.plot(mean_square_error_list_train)
plt.title('Average error vs epoches')
plt.xlabel('Number of epoches')
plt.ylabel('Average error')
plt.grid()
plt.show()

# %%
def run_test(x , w):
    y_test = []
    for i in range(len(x)):
        xn = x[i]
        sn = np.dot(w.T , [1 , xn])
        y_test.append(sn)
    return y_test

# %%
y_test_upd = run_test(x_test_arr , w_train)
# print(len(y_test_upd))
y_train_upd = run_test(x_train_arr , w_train)
# print(len(y_train_upd))

# %%
plt.scatter(x_train , y_train , color = 'blue' , label = 'Targeted')
plt.scatter(x_train , y_train_upd , color = 'r' , label = 'Modeled')
plt.title('Modelled vs Targeted for Training')
plt.xlabel('Input values')
plt.ylabel('Output values')
plt.grid()
plt.legend()
plt.show()

# %%
plt.scatter(x_test , y_test , color = 'blue' , label = 'Targeted')
plt.scatter(x_test , y_test_upd , color = 'r' , label = 'Modeled')
plt.title('Modelled vs Targeted for Testing')
plt.xlabel('Input values')
plt.ylabel('Output values')
plt.grid()
plt.legend()
plt.show()

# %%
#Plot of Mean square error
mse_train = mean_square_error(y_train_arr , y_train_upd)
print("MSE for training:" , mse_train)
mse_test = mean_square_error(y_test_arr , y_test_upd)
print("MSE for testing:" , mse_test)
plt.bar(['Training', 'Testing'], [mse_train , mse_test], color = 'blue' , linewidth = 10)
# plt.hist(mse_test , color = 'r' , linewidth = 10)
plt.grid()
plt.show()

# %%
#Scatter plot with target output on x-axis and model output on y-axis, for training data and test data
plt.scatter(y_train , y_train_upd , label = 'Target vs Modeled')
plt.plot([0.5,4.5] , [0.5,4.5] , color = 'r' , label = 'y = x line') #for checking where are target and modeled output same
plt.title('Target vs Modeled for Training')
plt.xlabel('Target')
plt.ylabel('Modeled')
plt.grid()
plt.legend()
plt.show()

# %%
plt.scatter(y_test , y_test_upd , label = 'Target vs Modeled')
plt.plot([0.5,4.5] , [0.5,4.5] , color = 'r' , label = 'y = x line')
plt.title('Target vs Modeled for Testing')
plt.xlabel('Target')
plt.ylabel('Modeled')
plt.grid()
plt.legend()
plt.show()

# %% [markdown]
# # Bivariate Data

# %%
data_b = np.loadtxt("C:/#MTECH COURSE/Semester 2/DL/Data for A1/Group12/Regression/BivariateData/12.csv" , delimiter = ",")
print(data_b.shape)

# %%
# x_b = data[: , 0].reshape(-1,1)
# y_b = data[: , 1].reshape(-1,1)
# print(x_b)
# print(y_b)
# df_x = pd.DataFrame(x)
# df_x
# df_y = pd.DataFrame(y)
df_b = pd.DataFrame(data_b)
df_b = df_b.sample(frac = 1)
df_b

# %%
fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.view_init(elev=10, azim=120)
ax.scatter3D(df_b[0], df_b[1], df_b[2] , color = 'green')
plt.title("Given Data")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("Output")
plt.show()

# %%
def splitting_bivariate(df , factor = 0.7):
    no_train_data = 10201*0.7
    training = df[:int(no_train_data)]
    testing = df[7140:]
    # x_train = training[:][0]
    return training , testing

# %%
training , testing = splitting_bivariate(df_b , factor = 0.7)

# %%
training

# %%
training_arr = training.to_numpy()
testing_arr = testing.to_numpy()

# %%
training_y = []
testing_y = []
for i in range(len(training_arr)):
    training_y.append(training_arr[i][2])
for i in range(len(testing_arr)):
    testing_y.append(testing_arr[i][2])
# print(training_y) 
#print(testing_y)

# %%
def run_train_bivariate(x):
    w = np.random.randn(3)
    max_epoches = 100
    eta = 0.001
    average_error_list = []
    while(max_epoches):
        error_list = []
        for i in range(len(x)):
            sn = np.dot(w.T , [1 , x[i][0] , x[i][1]])
            yn = x[i][2]
            err = error(yn , sn)
            error_list.append(err)
            w = grad_desc(np.array([1 , x[i][0] , x[i][1]]) , w , eta , yn , sn)
        max_epoches = max_epoches - 1
        avg_error = average_error(error_list)
        average_error_list.append(avg_error)
    # print(error_list) 
    return w , average_error_list

# %%
w_train_biv , average_error_list_train_biv = run_train_bivariate(training_arr)

# %%
def run_test_biv(x , w):
    y_test = []
    for i in range(len(x)):
        sn = np.dot(w.T , [1 , x[i][0] , x[i][1]])
        y_test.append(sn)
    return y_test

# %%
y_test_upd_biv = run_test_biv(testing_arr , w_train_biv)

y_train_upd_biv = run_test_biv(training_arr , w_train_biv)

# %%
fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.view_init(elev=10, azim=50)
ax.scatter3D(training[0] , training[1] , y_train_upd_biv , color = 'red')
plt.title("Linear Surface")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("Output")
plt.show()

# %%
plt.plot(average_error_list_train_biv)
plt.title('Average error vs number of epochs')
plt.xlabel('Epoches')
plt.ylabel('Average error')
plt.grid()
plt.show()

# %%
#Plot of Mean square error

mse_train_biv = mean_square_error(training_y , y_train_upd_biv)
print("MSE for training:" , mse_train_biv)
mse_test_biv = mean_square_error(testing_y , y_test_upd_biv)
print("MSE for testing:" , mse_test_biv)
plt.bar(['Training', 'Testing'], [mse_train_biv , mse_test_biv], color = 'blue' , linewidth = 10)
# plt.hist(mse_train_biv , color = 'blue' , linewidth = 10)
# plt.hist(mse_test_biv , color = 'r' , linewidth = 10)
plt.grid()
plt.show()

# %%
fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.view_init(elev=10, azim=50)
ax.scatter3D(training[0], training[1], training[2] , color = 'green')
ax.scatter3D(training[0] , training[1] , y_train_upd_biv , color = 'red')
# ax.scatter3D(testing[0] , testing[1] , testing[2] , color = 'blue')
plt.title("Modelled vs Targeted for Training")
# ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("Output")
plt.show()

# %%
fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.view_init(elev=10, azim=50)
ax.scatter3D(testing[0], testing[1], testing[2] , color = 'green')
ax.scatter3D(testing[0] , testing[1] , y_test_upd_biv , color = 'red')
# ax.scatter3D(testing[0] , testing[1] , testing[2] , color = 'blue')
plt.title("Modelled vs Targeted for Testing")
# ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("Output")
plt.show()

# %%
#Scatter plot with target output on x-axis and model output on y-axis, for training data and test data
plt.scatter(training_y , y_train_upd_biv , label = 'Target vs Modeled')
plt.plot([0,150] , [0,150] , color = 'r' , label = 'y = x line') #For checking where are target and modeled are same
plt.title('Target vs Modeled for Training')
plt.xlabel('Target')
plt.ylabel('Modeled')
plt.grid()
plt.legend()
plt.show()

# %%
plt.scatter(testing_y , y_test_upd_biv)
# plt.plot([0.5,4.5] , [0.5,4.5] , color = 'r')
plt.plot([0,150] , [0,150] , color = 'r' , label = 'y = x line')
plt.title('Target vs Modeled for Testing')
plt.xlabel('Target')
plt.ylabel('Modeled')
plt.grid()
plt.legend()
plt.show()


