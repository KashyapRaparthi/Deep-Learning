# %% [markdown]
# # Classification
# ## Linearly Seperable Data 

# %%
import numpy as np
from matplotlib import pyplot as plt

# Load the data from the text file
data1 = np.loadtxt(r'C:\Users\Asus\Documents\Python Codes\Assignments\Deep Learning\Assignment-1\Data\Classification\LS_Group12\Class1.txt')
data2 = np.loadtxt(r'C:\Users\Asus\Documents\Python Codes\Assignments\Deep Learning\Assignment-1\Data\Classification\LS_Group12\Class2.txt')
data3 = np.loadtxt(r'C:\Users\Asus\Documents\Python Codes\Assignments\Deep Learning\Assignment-1\Data\Classification\LS_Group12\Class3.txt')

# Split the data into training and testing sets
np.random.seed(42)
np.random.shuffle(data1)
np.random.shuffle(data2)
np.random.shuffle(data3)

train = np.concatenate([data1[:350],data2[:350],data3[:350]])
test = np.concatenate([data1[350:],data2[350:],data3[350:]])


train1 = data1[:350]
train2 = data2[:350]
train3 = data3[:350]

test1 = data1[350:]
test2 = data2[350:]
test3 = data3[350:]

# %%
plt.scatter(test1[:, 0], test1[:, 1],label='Class-1',edgecolors="blue")
plt.scatter(test2[:, 0], test2[:, 1],label='Class-2',edgecolors="orange")
plt.scatter(test3[:, 0], test3[:, 1],label='Class-3',edgecolors="green")
plt.xlabel("Abscissa")
plt.ylabel("Ordinate")
plt.title("Linearly Seperable Data \n Testing Data")
plt.legend()
plt.show()

# %%
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Perceptron(Data):
    W=np.array([1,1,1], dtype=float)
    epoch=100
    Err=np.array([])
    Error=np.array([])
    # while(1):
    #     if epoch==0:
    #         break
    #     else:
    #         for i in range(len(Data)):
    #             Xi = [1,Data[i,0],Data[i,1]] # augmenting sample
    #             Activation_Output = np.dot(W.T,Xi) # activation value
    #             S=Sigmoid(Activation_Output) # signal
    #             if i<350:
    #                 Yn=0
    #             else:
    #                 Yn=1
    #             En=((Yn-S)**2)/2 #instantaneous error
    #             LRP=0.5
    #             Err=np.append(Err,En) 
    #             Delta=LRP*(Yn-S)*S*(1-S) 
    #             W = np.add(W,np.dot(Delta,Xi),out=W,casting="unsafe") # correction
            
    #         AvgError=np.mean(Err)
    #         Error = np.append(Error,AvgError)
    #         epoch-=1

    for j in range(epoch):
        for i in range(len(Data)):
            Xi = np.concatenate(([1],Data[i,:]), axis=0) # augmenting sample
            Activation_Output = np.dot(W,Xi) # activation value
            S =Sigmoid(Activation_Output)
            if i<350:
                Yn=0
            else:
                Yn=1
            En = 0.5*(Yn-S)**2
            Err=np.append(Err,En) 
            eta = 0.25
            Delta = (Yn-S)*S*(1-S)
            del_w = eta*Delta*Xi
            # W = W + del_w # update
            W = np.add(W, del_w, out=W, casting='unsafe')
        AvgError=np.mean(Err)
        Error = np.append(Error,AvgError)
    return W,Error


# %%
W12,Err12=Perceptron(np.concatenate([train1,train2]))
W13,Err13=Perceptron(np.concatenate([train1,train3]))
W23,Err23=Perceptron(np.concatenate([train2,train3]))

# %%
# plt.plot(Err13)
# plt.plot(Err12)
# plt.plot(Err23)
plt.plot(Err23)
plt.title("Error vs Epoch \n Class 2 vs Class 3")
plt.xlabel("Epoch")
plt.ylabel("Average Error")

# %%
def classifier(Data,Weights,Threshold,classes):
    test = np.concatenate(([1],Data), axis=0) # augmenting sample
    Activation_Potential = np.dot(Weights,test) # activation value
    Signal = Sigmoid(Activation_Potential) # Made changes here
        
    if Signal < Threshold: # Made changes here
        label=classes[0]
    else:
        label=classes[1]

    return label

# %%
print(train1[0,:])
print(W12)

# %%
# generating points in the region
x_arr = np.linspace(-10, 29, 1000)
y_arr = np.linspace(-15, 17, 1000)
xx, yy = np.meshgrid(x_arr, y_arr)
region = np.concatenate((xx.reshape((-1,1)), yy.reshape((-1,1))), axis=1)
pred_region1v2 = []
for point in region:
    pred_region1v2.append(classifier(point,W12,0.5,[1,2]))
pred_region1v2 = np.reshape(pred_region1v2, xx.shape)
plt.figure()
plt.contourf(xx, yy, pred_region1v2, alpha = 0.5, cmap='Set1')
plt.scatter(train1[:, 0],train1[:, 1], label='Class 1', edgecolors='black')
plt.scatter(train2[:, 0],train2[:, 1], label='Class 2', edgecolors='black')
plt.xlabel("Abscissa")
plt.ylabel("Ordinate")
plt.title("Class-1 vs Class-2")
plt.legend()


# %%
x_arr = np.linspace(-10, 29, 1000)
y_arr = np.linspace(-15, 17, 1000)
xx, yy = np.meshgrid(x_arr, y_arr)
region = np.concatenate((xx.reshape((-1,1)), yy.reshape((-1,1))), axis=1)
pred_region1v2 = []
for point in region:
    pred_region1v2.append(classifier(point,W13,0.5,[1,3]))
pred_region1v2 = np.reshape(pred_region1v2, xx.shape)

plt.figure()
plt.contourf(xx, yy, pred_region1v2, alpha = 0.5, cmap='Set1')
plt.scatter(train1[:, 0],train1[:, 1], label='Class 1', edgecolors='black')
plt.scatter(train3[:, 0],train3[:, 1], label='Class 3', edgecolors='black')
plt.xlabel("Abscissa")
plt.ylabel("Ordinate")
plt.legend()
plt.title("Class-1 vs Class-3")
plt.legend()

# %%
x_arr = np.linspace(-10, 29, 1000)
y_arr = np.linspace(-15, 17, 1000)
xx, yy = np.meshgrid(x_arr, y_arr)
region = np.concatenate((xx.reshape((-1,1)), yy.reshape((-1,1))), axis=1)
pred_region1v2 = []
for point in region:
    pred_region1v2.append(classifier(point,W23,0.5,[2,3]))
pred_region1v2 = np.reshape(pred_region1v2, xx.shape)
plt.figure()
plt.contourf(xx, yy, pred_region1v2, alpha = 0.5, cmap='Set1')
plt.scatter(train2[:, 0],train2[:, 1], label='Class 2', edgecolors='black')
plt.scatter(train3[:, 0],train3[:, 1], label='Class 3', edgecolors='black')
plt.xlabel("Abscissa")
plt.ylabel("Ordinate")
plt.legend()
plt.title("Class-3 vs Class-2")
plt.legend()

# %%
def classifier3(Data,Weights):
    label=[]
    label.append(classifier(Data,Weights[0],0.5,[1,2]))
    label.append(classifier(Data,Weights[1],0.5,[1,3]))
    label.append(classifier(Data,Weights[2],0.5,[2,3]))
    return max(label,key=label.count)

# %%
# print(classifier3(train3[0],[W12,W13,W23]))
predicted=np.array([])
GT=np.array([])
for i in test1:
    predicted=np.append(predicted,classifier3(i,[W12,W13,W23]))
    GT=np.append(GT,1)
for i in test2:
    predicted=np.append(predicted,classifier3(i,[W12,W13,W23]))
    GT=np.append(GT,2)
for i in test3:
    predicted=np.append(predicted,classifier3(i,[W12,W13,W23]))
    GT=np.append(GT,3)

print(len(predicted))
print(len(GT))

# %%
x_arr = np.linspace(-10, 29, 1000)
y_arr = np.linspace(-15, 17, 1000)
xx, yy = np.meshgrid(x_arr, y_arr)
region = np.concatenate((xx.reshape((-1,1)), yy.reshape((-1,1))), axis=1)
pred_region1v2 = []
for point in region:
    pred_region1v2.append(classifier3(point,[W12,W13,W23]))
pred_region1v2 = np.reshape(pred_region1v2, xx.shape)
plt.figure()
plt.contourf(xx, yy, pred_region1v2, alpha = 0.5, cmap='Set1')
plt.scatter(train1[:, 0],train1[:, 1], label='Class 1', edgecolors='black')
plt.scatter(train2[:, 0],train2[:, 1], label='Class 2', edgecolors='black')
plt.scatter(train3[:, 0],train3[:, 1], label='Class 3', edgecolors='black')
plt.xlabel("Abscissa")
plt.ylabel("Ordinate")
plt.title("Class-1 vs Class-2 vs Class-3")
plt.legend()

# %%
from sklearn.metrics import confusion_matrix,recall_score,accuracy_score,f1_score

CM = confusion_matrix(GT,predicted)
Accuracy= accuracy_score(GT,predicted)
print("Confusion Matrix \n",CM)
print(f'\nAccuracy : {Accuracy*100}%')


