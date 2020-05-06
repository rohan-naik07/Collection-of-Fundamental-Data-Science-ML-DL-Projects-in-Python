from tensorflow.keras.datasets import fashion_mnist
from scipy.misc import imsave

(X_train,Y_train),(X_test,Y_test)=fashion_mnist.load_data()

for i in range(5):
    imsave(name="uploads/{}.png".format(i),arr=X_test[i])