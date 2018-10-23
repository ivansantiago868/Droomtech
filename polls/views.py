from django.http import HttpResponse
from io import StringIO
#from zeep import Client
import numpy as np
import seaborn as sns
sns.set(style='whitegrid')
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import PIL, PIL.Image
import zeep
def index(request):
 urlservice="http://localhost:52079/WSTensorData.svc?wsdl"
 client = zeep.Client(urlservice)
 respuesta = client.service.obtenerModelos()
 print(respuesta)
 iris = pd.read_csv('.\input\Iris.csv')
 iris.shape
 iris.head()
 iris = iris[:100]
 iris.shape
 iris.Species = iris.Species.replace(to_replace=['Iris-setosa', 'Iris-versicolor'], value=[0, 1])
 plt.scatter(iris[:50].SepalLengthCm, iris[:50].SepalWidthCm, label='Iris-setosa')
 plt.scatter(iris[51:].SepalLengthCm, iris[51:].SepalWidthCm, label='Iris-versicolo')
 plt.xlabel('SepalLength')
 plt.ylabel('SepalWidth')
 plt.legend(loc='best')
 X = iris.drop(labels=['Id', 'Species'], axis=1).values
 y = iris.Species.values
 seed = 5
 np.random.seed(seed)
 tf.set_random_seed(seed)
 train_index = np.random.choice(len(X), round(len(X) * 0.8), replace=False)
 test_index = np.array(list(set(range(len(X))) - set(train_index)))
 train_X = X[train_index]
 train_y = y[train_index]
 test_X = X[test_index]
 test_y = y[test_index]
 train_X = min_max_normalized(train_X)
 test_X = min_max_normalized(test_X)
 A = tf.Variable(tf.random_normal(shape=[4, 1]))
 b = tf.Variable(tf.random_normal(shape=[1, 1]))
 init = tf.global_variables_initializer()
 sess = tf.Session()
 sess.run(init)
 data = tf.placeholder(dtype=tf.float32, shape=[None, 4])
 target = tf.placeholder(dtype=tf.float32, shape=[None, 1])
 mod = tf.matmul(data, A) + b
 loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mod, labels=target))
 learning_rate = 0.003
 batch_size = 30
 iter_num = 1500
 opt = tf.train.GradientDescentOptimizer(learning_rate)
 goal = opt.minimize(loss)
 # Define the accuracy
 # The default threshold is 0.5, rounded off directly
 prediction = tf.round(tf.sigmoid(mod))
 # Bool into float32 type
 correct = tf.cast(tf.equal(prediction, target), dtype=tf.float32)
 # Average
 accuracy = tf.reduce_mean(correct)
 # End of the definition of the model framework
 loss_trace = []
 train_acc = []
 test_acc = []
 for epoch in range(iter_num):
  # Generate random batch index
  batch_index = np.random.choice(len(train_X), size=batch_size)
  batch_train_X = train_X[batch_index]
  batch_train_y = np.matrix(train_y[batch_index]).T
  sess.run(goal, feed_dict={data: batch_train_X, target: batch_train_y})
  temp_loss = sess.run(loss, feed_dict={data: batch_train_X, target: batch_train_y})
  # convert into a matrix, and the shape of the placeholder to correspond
  temp_train_acc = sess.run(accuracy, feed_dict={data: train_X, target: np.matrix(train_y).T})
  temp_test_acc = sess.run(accuracy, feed_dict={data: test_X, target: np.matrix(test_y).T})
  # recode the result
  loss_trace.append(temp_loss)
  train_acc.append(temp_train_acc)
  test_acc.append(temp_test_acc)
  # output
  if (epoch + 1) % 300 == 0:
   print('epoch: {:4d} loss: {:5f} train_acc: {:5f} test_acc: {:5f}'.format(epoch + 1, temp_loss,temp_train_acc, temp_test_acc))
 # Visualization of the results
 # loss function
 print(loss_trace)
 plt.plot(loss_trace)
 plt.title('Cross Entropy Loss')
 plt.xlabel('epoch')
 plt.ylabel('loss')
 plt.grid(True)
 buffer = StringIO()
 canvas = plt.get_current_fig_manager().canvas
 canvas.draw()
 #graphIMG = PIL.Image.fromstring('RGB', canvas.get_width_height(), canvas.tostring_rgb())
 #graphIMG.save(buffer, "PNG")
 #pylab.close()
 #plt.show()
 # In[28]:
 # accuracy
 #plt.plot(train_acc, 'b-', label='train accuracy')
 #plt.plot(test_acc, 'k-', label='test accuracy')
 #plt.xlabel('epoch')
 #plt.ylabel('accuracy')
 #plt.title('Train and Test Accuracy')
 #plt.legend(loc='best')
 #plt.show()
 data='Transaccion Completa'
 return HttpResponse("Mensaje: %s %s" %(data,client))

def min_max_normalized(data):
 col_max = np.max(data, axis=0)
 col_min = np.min(data, axis=0)
 return np.divide(data - col_min, col_max - col_min)