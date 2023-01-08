import numpy as np
from sklearn.decomposition import PCA

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

## Reshape and normalize the input dataset.
X_train = np.reshape(X_train, (-1,3072))/255
X_test = np.reshape(X_test, (-1,3072))/255

## Applying PCA to different scales.
res_list = []
# for i in np.linspace(0.2,1,5):
for i in [0.05, 0.10, 0.20, 0.50, 0.80]:
    print('PCA Percent:   ',int(i * 100), '%')
    pca = PCA(n_components = round(min(X_train.shape[0],X_train.shape[1] * i)))
    pca.fit(X_train)
    X_train_fit = pca.transform(X_train)

    print('Information kept: ', sum(pca.explained_variance_ratio_) * 100, '%')
    print('Noise variance: ', pca.noise_variance_)

