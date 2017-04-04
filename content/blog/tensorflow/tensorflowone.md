+++
showonlyimage = false
draft = false
image = 'img/portfolio/tensorflow2.jpeg'
title = "TensorFlow Self-Learning Note 1"
date = "2017-03-31"
weight = 0
type = "post"
author = "Kris Yu"
tags = ["tensorflow","python","machine-learning"]
+++
What is the best deep learning framework? There is no answer to this question but I am sure that TensorFlow is one of the most competitive candidates. Come and learn tensorflow from the very begining with me!<!--more-->

-   [Introduction](#introduction)
    -   [What is a Data Flow Graph](#what-is-a-data-flow-graph)
    -   [Features of TensorFlow](#features-of-tensorFlow)
-   [Basis](#basis)
    -   [Graph Computing](#graph-computing)  
        -   [Create a Graph](#create-a-graph)
        -   [Run a Session](#run-a-session)
    -   [Working in Interactive Environment](#working-in-interactive-environment)
    -   [Variables](#variables)
    -   [Fetches](#fetches)
    -   [Feeds](#feeds)
-   [Example of Linear Regression](#example-of-linear-regression)
-   [MNIST Handwritten Digits Recognition](#mnist-handwritten-digits-recognition)
    -   [Model Training](#model-training)
    -   [Model Evaluation](#model-evaluation)
             
## Introduction

[TensorFlow](https://github.com/tensorflow/tensorflow) is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) that flow between them. This flexible architecture lets you deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device without rewriting code. TensorFlow also includes TensorBoard, a data visualization toolkit.

TensorFlow was originally developed by researchers and engineers working on the Google Brain team within Google's Machine Intelligence research organization for the purposes of conducting machine learning and deep neural networks research. The system is general enough to be applicable in a wide variety of other domains, as well.(cited from github)

### What is a Data Flow Graph

A data-flow graph (DFG) is a graph which represents a data dependancies between a number of operations. The nodes in the graph represent operations and edges connect nodes together by passing tensors. To conclude, tensors flowing in the DFG is TensorFlow.

### Features of TensorFlow

1. Flexibility

    TensorFlow is not only a neural network framwork. You can do whatever you want with the power of TensorFlow if you are able to make the calculations in a data flow graph

2. Portability
    
    TensorFlow is able to work on any devices with CPU or GPU. You can focus on your idea without concerning about hardware environment

3. Program Language Support
 
    TensorFlow currently support Python and C++

4. Efficiency

    TensorFlow makes full use of hardware resources and maximizes computing performance

## Basis
* Make all calculations as data flow graph
* Work with the graph by running a **Session**
* Present data as **tensors**
* Use **Variables** to keep information of status
* Fill in data and obtain the result of any operations with **feeds** and **fetches**

Each node in the graph is called an **op**(short for operation). An ops uses 0 or more **tensors** and generates 0 or more **tensors** with some calculations. A **tensor** is a multi-dimensional array. You can take a graph as a 4 dimensional array [batch, height, witdth, channels] where all elements in the array are floats.

Session puts all ops in the graph to CPUs or GPUs, then execute them with methods and return tensors finally. Tensor in Python is a numpy ndarray objects while in C++ is tensorflow::Tensor

### Graph Computing
You can take creating a graph as construct a neural network while executing the session as training the network.
#### Create a Graph

In TensorFlow, **Constant** is a kind of **op** without any input but you can make it a as a input of other ops. TensorFlow module in Python has a default graph so that you can directly generate ops on it. More details about [Graph Class](https://www.tensorflow.org/api_guides/python/framework#Graph)


```python
import tensorflow as tf
import numpy as np
from PIL import Image
```


```python
# Create a Constant op that produces a 1x2 matrix.  The op is
# added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
matrix1 = tf.constant([[3., 3.]])

# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.],[2.]])

# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', represents the result of the matrix
# multiplication.
product = tf.matmul(matrix1, matrix2)
```

Now there are 3 ops in the default graph: 2 **Constant( ) ops** and 1 **matmul( ) op**. In order to get the result of the matrices multiplication, we need a session the run the computings.

#### Run a Session
Find more details at [Session Class](https://www.tensorflow.org/api_guides/python/client#session-management)


```python
# Launch the default graph.
sess = tf.Session()

# To run the matmul op we call the session 'run()' method, passing 'product'
# which represents the output of the matmul op.  This indicates to the call
# that we want to get the output of the matmul op back.
#
# All inputs needed by the op are run automatically by the session.  They
# typically are run in parallel.
#
# The call 'run(product)' thus causes the execution of threes ops in the
# graph: the two constants and matmul.
#
# The output of the op is returned in 'result' as a numpy `ndarray` object.
result = sess.run(product)
print(result)

# Close the Session when we're done.
sess.close()
```

    [[ 12.]]


Do not forget to close the session in the end. However, you may want to use **with** to avoid the close


```python
with tf.Session() as sess:
  result = sess.run([product])
  print(result)
```

    [array([[ 12.]], dtype=float32)]


The computings are all on CPUs and GPUs. TensorFlow will use the first GPU by default if you are using GPU for executing. If you would like to work on other GPUs:
with tf.Session() as sess:
  with tf.device("/gpu:1"):
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.],[2.]])
    product = tf.matmul(matrix1, matrix2)
    ...
Names for Devices:

* "/cpu:0"： Your CPU
* "/gpu:0"： Your first GPU
* "/gpu:1"： Your second GPU

Documentation of [GPU](https://www.tensorflow.org/tutorials/using_gpu) using

### Working in Interactive Environment

We were using Session and Session.run() to execute the graph computing. However, you can use InteractiveSession class and methods like Tensor.eval(), Operation.run() in interactive environments such IPython and Jupyter Notebook.


```python
# Enter an interactive TensorFlow Session.
sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# Initialize 'x' using the run() method of its initializer op.
x.initializer.run()

# Add an op to subtract 'a' from 'x'.  Run it and print the result
sub = tf.sub(x, a)
print(sub.eval())

# Close the Session when we're done.
sess.close()
```

    [-2. -1.]


### Variables

Variables will keep their information of status while running a session. They are usually used to represent parameters which are need to be trained with iterations in an algorithm since they can memorize the status of each iteration. The variable in the following example take the role as a simple counter.


```python
# Create a Variable, that will be initialized to the scalar value 0.
state = tf.Variable(0, name="counter")

# Create an Op to add one to `state`.

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# Variables must be initialized by running an `init` Op after having 
# launched the graph.  We first have to add the `init` Op to the graph.
init_op = tf.global_variables_initializer()

# Launch the graph and run the ops.
with tf.Session() as sess:
    # run the init op
    sess.run(init_op)
    # Print the initial value of 'state'
    print(sess.run(state))
    # Run the op that updates 'state' and print 'state'.
    for i in range(3):
        sess.run(update)
        print(sess.run(state))
    
```

    0
    1
    2
    3


### Fetches

To fetch the result of an op, you can **print** the status after executing **sess.run()** 


```python
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermd = tf.add(input3, input2)
mul = tf.mul(input1, intermd)

with tf.Session() as sess:
    result = sess.run([mul, intermd])
    print(result)
```

    [21.0, 7.0]


### Feeds

First create a placeholder, then feed it with data


```python
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
  print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))
```

    [array([ 14.], dtype=float32)]


Not feeding a placehold will throw out errors.

## Example of Linear Regression


```python
# Randomly generate 100 pairs of (x, y) where y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype("float32")
y_data = x_data * 0.1 + 0.3

# Initialize w and b with variables
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b    # here + and * has been overloaded for tf.add() and tf.mul()

# Least Square Error
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Initialize the variables in tensorflow
init = tf.global_variables_initializer()

# Run the session
with tf.Session() as sess:
    sess.run(init)
    # monitor the changes of coefficients every 20 iterations
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(W), sess.run(b))
```

    0 [-0.21695098] [ 0.66673416]
    20 [-0.00646714] [ 0.35819602]
    40 [ 0.07154677] [ 0.31555283]
    60 [ 0.09239592] [ 0.30415648]
    80 [ 0.09796781] [ 0.30111083]
    100 [ 0.09945691] [ 0.30029687]
    120 [ 0.09985484] [ 0.30007935]
    140 [ 0.09996121] [ 0.3000212]
    160 [ 0.09998964] [ 0.30000567]
    180 [ 0.09999724] [ 0.30000153]
    200 [ 0.09999929] [ 0.3000004]


## MNIST Handwritten Digits Recognition

MNIST is a dataset of simple handwritten digits. You can get access to the data [here](http://yann.lecun.com/exdb/mnist/). There are 55,000 observations in training set and 10,000 in test set. Each 28x28 picture is flatten to an array of length 784 and the response is from 0 to 9. However, MNIST is the sample data set of tensorFlow in Python. It is quite easy to get the data by two lines of code:


```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz



```python
try_x,try_y =mnist.train.next_batch(1)
Image.fromarray(np.uint8(try_x.reshape(28,28)*255)).resize((300,300),Image.ANTIALIAS)
```




![png](/blog/tensorflow/output_30_0.png)



Since this is only an entry level multiclass classification, I will not discuss the details about the algorithm here but focus on the implementation with TensorFlow. 


```python
x = tf.placeholder(tf.float32, [None, 784])
```

Here x is about a specific number but a placeholder. We use [None, 784] to represent the whole MNIST data set where None could be anything.

We use variables to represent weights and bias which are changeable


```python
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
```

W and b are initialized with 0. W is 784x10 since the response is one hot vector and the bias has length 10.

Implement **softmax regression** only needs one line of code


```python
y = tf.nn.softmax(tf.matmul(x, W) + b)
```

### Model Training

Loss function is one of the essential parts of a machine learning algorithm and it indicates the goodness of training. Here we use cross-entropy as our loss function. Cross-entropy comes from information theory but not is applied in many other fields. Its defination is:
$$H_y^\prime(y)=-\sum_iy_i^\prime\log(y_i)$$ where y is the predicted probability and \\(y^\prime\\) is the true values. You can find more detials about cross-entropy [here](http://colah.github.io/posts/2015-09-Visual-Information/). 



In order to implement cross-entropy, we set a placeholder to store the true responses.


```python
y_ = tf.placeholder(tf.float32, [None, 10])
```

And then define the cross-entropy. This is not restricted to only one picture. It can be the whole data set.


```python
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
```

TensorFlow automatically uses back propagation to modify the parameters for the purpose of minizing the loss. The last thing you need to set is choosing a algorithm for minizing the loss. Here we select gradient descent. TensorFlow also offers other [optimization methods](https://www.tensorflow.org/api_guides/python/train#Optimizers)


```python
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
```


```python
# Initialize the variables before training
init = tf.global_variables_initializer()
```

#### Model Evaluation

We use **tf.argmax()** to predict the label and **tf.equal()** to find whether the predictions matche the true labels. correct_prediction is a list of boolean values. For example [True, False, True, True]. One can use **tf.cast()** to convert it into [1, 0, 1, 1] and the accuracy is 0.75


```python
# Start a session
sess = tf.Session()
# Initialize the variables
sess.run(init) 
# Train the model with 1000 as the maximum iterations
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(10)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Model Evaluation.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("Accuarcy on Test-dataset: ", sess.run(accuracy, \
                    feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

sess.close()
```

    Accuarcy on Test-dataset:  0.8688


Softmax regression is too simple to get an ideal result. Here our result is around 0.88 while it can reach 0.97 by some simple optimization. Nowadays, the best accuracy is 99.7%.
