import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import cross_validate,cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix


## Dataset Layout
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

## Set the list of label names and read the 1st batch and the test set.
Labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
dict_batch1 = unpickle('data_batch_1')
dict_test = unpickle('test_batch')

TRAIN_NUM = -1
TEST_NUM = -1
## Split the data into 4 sets.
X_train = dict_batch1.get(b'data')[0:TRAIN_NUM]
y_train = dict_batch1.get(b'labels')[0:TRAIN_NUM]
X_test = dict_test.get(b'data')[0:TEST_NUM]
y_test = dict_test.get(b'labels')[0:TEST_NUM]

## Reshape the input dataset.
X_train = np.reshape(X_train, (-1,3072))/255
X_test = np.reshape(X_test, (-1,3072))/255

## 1. RBF SVM with multi-class classification
def SVM_RBF(C,cv):
    print('The Hyperparameter C here is:   ', C)
    RBF = svm.SVC(kernel = 'rbf', C = C, degree = 3, probability = True)
    RBF.fit(X_train, y_train)
    scores = cross_val_score(RBF, X_train, y_train, cv=cv)
    print('Accuracy Score of the Train part:   ', scores.mean())

        ## Do the prediction and then test the model using the test set.
    y_pred = RBF.predict(X_test)

        ## Output all the inspection index.
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy Score of the Test part:   ', accuracy)

    return y_pred, accuracy

## Initialize the lists.
cv = 5
Acc,Recall_tot, Precision_tot, F_1_tot = [],[],[],[]
C_list = np.linspace(1,6,11)


for i in C_list:

    ## Do the SVM test.
    y_pred, accuracy = SVM_RBF(i,cv)
    y_pred = y_pred.tolist()
    Acc.append(accuracy * 100)

    ## Initialize the TP, FP, FN sets.
    num_TP = np.zeros(10)
    num_FP = np.zeros(10)
    num_FN = np.zeros(10)

    ## Calculate TP, FP, FN.
    for j in range(len(y_pred)):
        if y_pred[j] == y_test[j]:
            num_TP[y_test[j]] += 1
        elif y_pred[j] != y_test[j]:
            num_FP[y_test[j]] += 1
            num_FN[y_pred[j]] += 1

    ## Calculate Recall, Precision, F_1.
    Recall = num_TP/(num_TP + num_FN)
    Precision = num_TP/(num_TP + num_FP)
    F_1 = 2 * (Recall * Precision)/(Recall + Precision)

    ## Make such results into dictionary form.
    output_Recall = dict(zip(Labels, Recall))
    output_Precision = dict(zip(Labels, Precision))
    output_F_1 = dict(zip(Labels, F_1))

    ## Show the results.
    print(
        '\n The Hyperparameter C here is:  {}, \n The Test Recalls are : {}, \n The Test Precisions are: {}, \n The Test F1 Values are: {} \n The total Accuracy is: {} \n'.format(
            i,
            output_Recall,
            output_Precision,
            output_F_1,
            accuracy
        ))

## Make the figure.
    Recall_tot.append(sum(Recall)*10)
    Precision_tot.append(sum(Precision)*10)
    F_1_tot.append(sum(F_1)*10)

l1 = plt.plot(C_list, Acc, 'r--', label = 'Total Accuracy')
l2 = plt.plot(C_list, Recall_tot, 'g--', label = 'Total Recall')
l3 = plt.plot(C_list, Precision_tot, 'b--', label = 'Total Precision')
l4 = plt.plot(C_list, F_1_tot, 'm--', label = 'Total F1 Values')
plt.plot(C_list, Acc, 'ro-', C_list, Recall_tot, 'g+-', C_list, Precision_tot, 'b^-', C_list, F_1_tot, 'mx-')
plt.title('Result vs. C')
plt.xlabel('C')
plt.ylabel('Result (%)')
plt.legend()
plt.show()



