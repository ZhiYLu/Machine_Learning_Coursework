import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import cross_validate,cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

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

## 1. linear SVM with multi-class classification
res_list = []
# for i in np.linspace(0.2,1,5):

def SVM_Linear(percent,cv):
    print('The PCA precentage here is', percent * 100, '%')
    pca = PCA(n_components = round(min(X_train.shape[0],X_train.shape[1] * percent)))
    pca.fit(X_train)
    X_train_fit = pca.transform(X_train)

    ## Do the linear SVM.
    linear = svm.SVC(kernel = 'linear', C = 1.0, probability=True)
    linear.fit(X_train_fit, y_train)
    scores = cross_val_score(linear, X_train_fit, y_train, cv = cv)
    print('Accuracy Score:   ',scores.mean())

    ## Do the prediction and then test the model using the test set.
    X_test_fit = pca.transform(X_test)
    y_pred = linear.predict(X_test_fit)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy Score of the Test part:   ', accuracy)

    return y_pred,accuracy

## Initialize the lists.
per_list = []
Acc,Recall_tot, Precision_tot, F_1_tot = [],[],[],[]
cv = 5

for i in [0.05, 0.10, 0.20, 0.50, 0.80, 1.00]:

    ## Collect the fearture scalers.
    per_list.append(round(min(X_train.shape[0],X_train.shape[1] * i)))

    ## Do the SVM test.
    y_pred,accuracy = SVM_Linear(i, cv)
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
    Recall = num_TP / (num_TP + num_FN)
    Precision = num_TP / (num_TP + num_FP)
    F_1 = 2 * (Recall * Precision) / (Recall + Precision)

    ## Make such results into dictionary form.
    output_Recall = dict(zip(Labels, Recall))
    output_Precision = dict(zip(Labels, Precision))
    output_F_1 = dict(zip(Labels, F_1))

    ## Show the results.
    print('\n The PCA percent here is:  {} %, \n The Test Recalls are : {}, \n The Test Precisions are: {}, \n The Test F1 Values are: {} \n The total accuray is: {}\n'.format(
        i * 100,
        output_Recall,
        output_Precision,
        output_F_1,
        accuracy
    ))

## Make the figure.
    Recall_tot.append(sum(Recall)*10)
    Precision_tot.append(sum(Precision)*10)
    F_1_tot.append(sum(F_1)*10)

l1 = plt.plot(per_list, Acc, 'r--', label = 'Total Accuracy')
l2 = plt.plot(per_list, Recall_tot, 'g--', label = 'Total Recall')
l3 = plt.plot(per_list, Precision_tot, 'b--', label = 'Total Precision')
l4 = plt.plot(per_list, F_1_tot, 'm--', label = 'Total F1 Values')
plt.plot(per_list, Acc, 'ro-', per_list, Recall_tot, 'g+-', per_list, Precision_tot, 'b^-', per_list, F_1_tot, 'mx-')
plt.title('Result vs. Feature Dimension')
plt.xlabel('Feature Dimension')
plt.ylabel('Result (%)')
plt.legend()
plt.show()



