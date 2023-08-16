# %% [markdown]
# # Classification
# ## Non-Linearly Seperable Data 

# %%
import numpy as np
from matplotlib import pyplot as plt

# %%
data=np.loadtxt(r"C:\Users\Asus\Documents\Python Codes\Assignments\Deep Learning\Assignment-1\Data\Classification\NLS_Group12.txt", dtype=float)
data.shape

# %%

data1= data[:500, :]
data2= data[500:1000, :]
data3= data[1000:, :]

train1 = data1[:350]
train2 = data2[:350]
train3 = data3[:350]

test1 = data1[350:]
test2 = data2[350:]
test3 = data3[350:]

# %%
plt.scatter(train1[:, 0], train1[:, 1],label='Class-1',edgecolors="blue")
plt.scatter(train2[:, 0], train2[:, 1],label='Class-2',edgecolors="orange")
plt.scatter(train3[:, 0], train3[:, 1],label='Class-3',edgecolors="green")
plt.xlabel("Abscissa")
plt.ylabel("Ordinate")
plt.title("Non-Linearly Seperable Data \n Training Data")
plt.legend()
plt.show()

# %%
plt.scatter(test1[:, 0], test1[:, 1],label='Class-1',edgecolors="blue")
plt.scatter(test2[:, 0], test2[:, 1],label='Class-2',edgecolors="orange")
plt.scatter(test3[:, 0], test3[:, 1],label='Class-3',edgecolors="green")
plt.xlabel("Abscissa")
plt.ylabel("Ordinate")
plt.title("Non-Linearly Seperable Data \n Testing Data")
plt.legend()
plt.show()

# %%
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Perceptron(Data):
    W=np.array([1,1,1], dtype=float)
    epoch=200
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
    
    for _ in range(epoch):
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
            eta = 0.01
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
# plt.plot(Err12)
# plt.plot(Err13)
plt.plot(Err23)
plt.title("Error vs Epoch \n Class 2 vs Class 3")
plt.xlabel("Epoch")
plt.ylabel("Average Error")

# %%
def classifier(Data,Weights,Threshold,classes):
    test = np.concatenate(([1],Data), axis=0) 
    Activation_Potential = np.dot(Weights,test) 
    Signal = Sigmoid(Activation_Potential) 
    if Signal < Threshold: 
        label=classes[0]
    else:
        label=classes[1]

    return label

# %%
print(train1[0,:])
print(W12)

# %%
print(np.min(data[:,0]))

# %%
# generating points in the Region
x_arr = np.linspace(-4, 4, 1000)
y_arr = np.linspace(-2, 2, 1000)
xx, yy = np.meshgrid(x_arr, y_arr)
Region = np.concatenate((xx.reshape((-1,1)), yy.reshape((-1,1))), axis=1)
Region.shape

# %%
predicted_region = []
for point in Region:
    predicted_region.append(classifier(point,W12,0.5,[1,2]))
predicted_region = np.reshape(predicted_region, xx.shape)
plt.figure()
plt.contourf(xx, yy, predicted_region, alpha = 0.5, cmap='Set1')
plt.scatter(train1[:, 0],train1[:, 1], label='Class 1', edgecolors='white')
plt.scatter(train2[:, 0],train2[:, 1], label='Class 2', edgecolors='white')
plt.title("Decision plot for Class 1 & Class 2")
plt.xlabel("Abcissa")
plt.ylabel('Ordinate')
plt.legend()

# %%
predicted_region = []
for point in Region:
    predicted_region.append(classifier(point,W13,0.5,[1,3]))
predicted_region = np.reshape(predicted_region, xx.shape)
plt.figure()
plt.contourf(xx, yy, predicted_region, alpha = 0.5, cmap='Set1')
plt.scatter(train1[:, 0],train1[:, 1], label='Class 1', edgecolors='white')
plt.scatter(train3[:, 0],train3[:, 1], label='Class 3', edgecolors='white')
plt.title("Decision plots for Class 1 & Class 3")
plt.xlabel("Abcissa")
plt.ylabel('Ordinate')
plt.legend()

# %%
predicted_region = []
for point in Region:
    predicted_region.append(classifier(point,W23,0.5,[2,3]))
predicted_region = np.reshape(predicted_region, xx.shape)
plt.figure()
plt.contourf(xx, yy, predicted_region, alpha = 0.5, cmap='Set1')
plt.scatter(train2[:, 0],train2[:, 1], label='Class 2', edgecolors='white')
plt.scatter(train3[:, 0],train3[:, 1], label='Class 3', edgecolors='white')
plt.title("Decision plot for Class 2 & Class 3")
plt.xlabel("Abcissa")
plt.ylabel('Ordinate')
plt.legend()

# %%
def classifier3(Data,Weights):
    label=[]
    label.append(classifier(Data,Weights[0],0.5,[1,2]))
    label.append(classifier(Data,Weights[1],0.5,[1,3]))
    label.append(classifier(Data,Weights[2],0.5,[2,3]))
    return max(label,key=label.count)

# %%
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
predicted_region = []
for point in Region:
    predicted_region.append(classifier3(point,[W12,W13,W23]))
predicted_region = np.reshape(predicted_region, xx.shape)
plt.figure()
plt.contourf(xx, yy, predicted_region, alpha = 0.5, cmap='Set1')
plt.scatter(test1[:, 0],test1[:, 1], label='Class 1', edgecolors='white')
plt.scatter(test2[:, 0],test2[:, 1], label='Class 2', edgecolors='white')
plt.scatter(test3[:, 0],test3[:, 1], label='Class 3', edgecolors='white')
plt.title("Decision plots for 3 Class")
plt.xlabel("Abcissa")
plt.ylabel('Ordinate')
plt.legend()

# %%
from sklearn.metrics import confusion_matrix,recall_score,accuracy_score,f1_score

CM = confusion_matrix(GT,predicted)
Accuracy= accuracy_score(GT,predicted)
print("Confusion Matrix \n",CM)
print(f'\nAccuracy : {round(Accuracy,2)*100}%')


