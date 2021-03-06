

```python
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
%matplotlib inline
```

Generate pseudo-data


```python
n = 150
np.random.seed(110104)
x = 4.0 * (np.random.rand(n) - 0.5)
xfeat = np.column_stack((x**i for i in range(1, 9))) # polynomials up to degree 8
y = 2.0 - 0.2*x + 0.2*(2*x - 0.5)**2 + 0.2 * np.random.randn(n)
y[2] = 4.0 # Artificial outliers
y[3] = 7.0 # Artificial outliers
y[10] = 5.0 # Artificial outliers
y = y.reshape((n, 1))

plt.scatter(x, y)
```




    <matplotlib.collections.PathCollection at 0x1f219ce6e10>




![png](output_2_1.png)


Let's split data in test and train (we'll use only train for Bayesian evidence and test for cross-validation)


```python
n_train = 75
xfeat_train, y_train = xfeat[:n_train, :], y[:n_train, :]
xfeat_test, y_test = xfeat[n_train:, :], y[n_train:, :]
```

This is a function to generate training batches


```python
def get_batches(xfeat, y, batch_size):
    n_batches = xfeat.shape[0] // batch_size
    for i in range(n_batches):
        yield xfeat[i*batch_size:(i + 1)*batch_size,:], y[i*batch_size:(i + 1)*batch_size,:]
```

Create a very simple architecture with one hidden and a l2 regulariser


```python
def nn_input(dim_features, dim_output):
    features = tf.placeholder(tf.float32, (None, dim_features))
    target = tf.placeholder(tf.float32, (None, dim_output))
    lr = tf.placeholder(tf.float32)
    penalty = tf.placeholder(tf.float32)
    precision = tf.placeholder(tf.float32)
    return features, target, lr, penalty, precision
    
def hidden_layer(features, dim_features, dim_hidden):
    # The weights are created as a vector so that taking hessian is easier
    W = tf.Variable(tf.truncated_normal((dim_features*dim_hidden, ), stddev=1/np.sqrt(dim_features*dim_hidden)))
    Wmat = tf.reshape(W, (dim_features, dim_hidden))
    b = tf.Variable(tf.zeros(dim_hidden))
    hidden = tf.nn.relu(tf.matmul(features, Wmat) + b)
    return hidden, W, b

def output_layer(hidden, dim_hidden, dim_output):
    # The weights are created as a vector so that taking hessian is easier
    W2 = tf.Variable(tf.truncated_normal((dim_hidden*dim_output, ), stddev=1/np.sqrt(dim_hidden*dim_output)))
    W2mat = tf.reshape(W2, (dim_hidden, dim_output))
    b2 = tf.Variable(tf.zeros(1))
    output = tf.matmul(hidden, W2mat) + b2
    return output, W2, b2

def nn_loss(output, target, penalty, precision, W, W2):
    errors = output - target
    weights = tf.concat([W, W2], axis = 0)
    loss = precision * tf.reduce_sum(tf.square(errors)) + penalty * tf.reduce_sum(tf.square(weights))
    return loss, errors, weights

def nn_optimization(loss, lr):
    optim = tf.train.AdamOptimizer(lr).minimize(loss)
    return optim
    
class neural_network():
    def __init__(self, dim_features, dim_output, dim_hidden):
        self.features, self.target, self.lr, self.penalty, self.precision = nn_input(dim_features, dim_output)
        self.hidden, self.W, self.b = hidden_layer(self.features, dim_features, dim_hidden)
        self.output, self.W2, self.b2 = output_layer(self.hidden, dim_hidden, dim_output)
        self.loss, self.errors, self.weights = nn_loss(
            self.output, self.target, self.penalty, self.precision, self.W, self.W2)
        self.optim = nn_optimization(self.loss, self.lr)
        self.hessian = tf.hessians(self.loss, [self.W, self.W2])[0] # Jesus! Hessians!
```

Now create the neural network


```python
dim_features = xfeat_train.shape[1]
dim_output = y_train.shape[1]
dim_hidden = 50

tf.reset_default_graph()
nn = neural_network(dim_features, dim_output, dim_hidden) # Takes some time to create just because there is a hessian!
nn.__dict__
```




    {'W': <tf.Variable 'Variable:0' shape=(400,) dtype=float32_ref>,
     'W2': <tf.Variable 'Variable_2:0' shape=(50,) dtype=float32_ref>,
     'b': <tf.Variable 'Variable_1:0' shape=(50,) dtype=float32_ref>,
     'b2': <tf.Variable 'Variable_3:0' shape=(1,) dtype=float32_ref>,
     'errors': <tf.Tensor 'sub:0' shape=(?, 1) dtype=float32>,
     'features': <tf.Tensor 'Placeholder:0' shape=(?, 8) dtype=float32>,
     'hessian': <tf.Tensor 'TensorArrayStack/TensorArrayGatherV3:0' shape=(?, 400) dtype=float32>,
     'hidden': <tf.Tensor 'Relu:0' shape=(?, 50) dtype=float32>,
     'loss': <tf.Tensor 'add_2:0' shape=<unknown> dtype=float32>,
     'lr': <tf.Tensor 'Placeholder_2:0' shape=<unknown> dtype=float32>,
     'optim': <tf.Operation 'Adam' type=NoOp>,
     'output': <tf.Tensor 'add_1:0' shape=(?, 1) dtype=float32>,
     'penalty': <tf.Tensor 'Placeholder_3:0' shape=<unknown> dtype=float32>,
     'precision': <tf.Tensor 'Placeholder_4:0' shape=<unknown> dtype=float32>,
     'target': <tf.Tensor 'Placeholder_1:0' shape=(?, 1) dtype=float32>,
     'weights': <tf.Tensor 'concat:0' shape=(450,) dtype=float32>}



We now train and evaluate the network


```python
epochs = 500
print_every_epochs = 50
batch_size = 10
lr = 0.01
penalty = 0.00000001
precision = 25.0 # because we use noise with sigma = 0.2

hyperparams= {nn.penalty: penalty, nn.precision: precision, nn.lr: lr}
full_training_feed = {nn.features: xfeat_train, nn.target: y_train, **hyperparams}

with tf.Session() as sess:
    # Training
    sess.run(tf.global_variables_initializer())
 
    for epoch in range(epochs):
        for batch_x, batch_y in get_batches(xfeat_train, y_train, batch_size):
            batch_feed = {nn.features: batch_x, nn.target: batch_y, **hyperparams}
            sess.run(nn.optim, batch_feed)
        if epoch == 0 or (epoch  + 1) % print_every_epochs == 0:
            loss = sess.run(nn.loss, full_training_feed)
            print("epoch %2d training loss %4.4f" % (epoch + 1, loss))

    # Visualize fit, errors, and weights
    errors, weights, output = sess.run([nn.errors, nn.weights, nn.output], full_training_feed)
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize = (12, 3))
    ax0.scatter(xfeat_train[:,0], y_train)
    ax0.scatter(xfeat_train[:,0], output)
    ax0.set_title('Fitted vs Actual')
    ax1.hist(errors)
    ax1.set_title('distribution of errors')
    ax2.hist(weights)
    ax2.set_title("distribution of weights")
```

    epoch  1 training loss 17858.8691
    epoch 50 training loss 682.9286
    epoch 100 training loss 634.6017
    epoch 150 training loss 885.7295
    epoch 200 training loss 643.4275
    epoch 250 training loss 663.7785
    epoch 300 training loss 667.2808
    epoch 350 training loss 659.0621
    epoch 400 training loss 635.3800
    epoch 450 training loss 663.4702
    epoch 500 training loss 649.6409
    


![png](output_12_1.png)


Let's repeat this but with Bayesian evidence!


```python
def log_evidence(hessian, loss, penalty, precision, dim_hidden, n_obs, n_weights):
    s, logdet = np.linalg.slogdet(hessian)
    N, k = n_obs, n_weights
    posterior_energy = -0.5 * loss + 0.5 * k * np.log(2*np.pi) - 0.5 * logdet
    weights_energy = 0.5 * k * np.log(2 * np.pi / penalty) 
    model_energy = 0.5 * N * np.log(2 * np.pi / precision)
    symmetry_factor = dim_hidden * np.log(2) + np.sum(np.log(np.arange(1, dim_hidden + 1)))
    return symmetry_factor + posterior_energy - weights_energy - model_energy
```


```python
def test_nn(
        xfeat_train, y_train,
        xfeat_test, y_test,
        dim_hidden, 
        epochs,
        batch_size,
        lr,
        penalty,
        precision): 
    # Create network
    dim_features = xfeat.shape[1]
    dim_output = y.shape[1]
    tf.reset_default_graph()
    nn = neural_network(dim_features, dim_output, dim_hidden) # Takes some time to create just because there is a hessian!
    # Feed dictionaries
    hyperparams= {nn.penalty: penalty, nn.precision: precision, nn.lr: lr}
    with tf.Session() as sess:
        # Train
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            for batch_x, batch_y in get_batches(xfeat_train, y_train, batch_size):
                batch_feed = {nn.features: batch_x, nn.target: batch_y, nn.lr: lr, **hyperparams}
                sess.run(nn.optim, batch_feed)
        # Find Bayesian (Log)evidence
        full_train_feed = {nn.features: xfeat_train, nn.target: y_train, **hyperparams}
        full_test_feed = {nn.features: xfeat_test, nn.target: y_test, **hyperparams}
        hessian, loss_train, output, weights = sess.run([nn.hessian, nn.loss, nn.output, nn.weights], full_train_feed)
        evidence = log_evidence(hessian, loss_train, penalty, precision, dim_hidden, len(output), len(weights)) 
        loss_test = sess.run(nn.loss, full_test_feed)
        return evidence, loss_train, loss_test
```

Test and compare evidence for multiple precisions and noises


```python
# Optimization parameters
epochs = 500
batch_size = 10
lr = 0.01

# Network parameters
dim_hidden_list = [2, 5, 10, 25, 50, 100]
penalty_list = [0.00000001, .00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
precision = 25.0

evidence_trace = []
generror_trace = []
for dim_hidden in dim_hidden_list:   
    dim_av_evidence = 0
    for penalty in penalty_list:     
        bayes_train, loss_train, loss_test = test_nn(
            xfeat_train, y_train, xfeat_test, y_test, dim_hidden, epochs, batch_size, lr, penalty, precision)
        evidence_trace.append(bayes_train)
        generror_trace.append(loss_test)
        print_str = 'dim_hid: %4d  penalty: %14.8f evidence: %9.2f' + \
            ' loss_train: %9.2f loss_test: %9.2f '
        dim_av_evidence += bayes_train / len(penalty_list)
        print(print_str %(dim_hidden, penalty, bayes_train, loss_train, loss_test))
    print('Average evidence for %d hidden variables is %0.2f' % (dim_hidden, dim_av_evidence))
```

    dim_hid:    2  penalty:     0.00000001 evidence:   -416.04 loss_train:    693.27 loss_test:    138.20 
    dim_hid:    2  penalty:     0.00001000 evidence:   -414.04 loss_train:    666.99 loss_test:    172.23 
    dim_hid:    2  penalty:     0.00010000 evidence:   -386.62 loss_train:    684.87 loss_test:    179.39 
    dim_hid:    2  penalty:     0.00100000 evidence:   -356.24 loss_train:    689.19 loss_test:    137.48 
    dim_hid:    2  penalty:     0.01000000 evidence:   -353.30 loss_train:    673.73 loss_test:    150.74 
    dim_hid:    2  penalty:     0.10000000 evidence:   -463.73 loss_train:    984.04 loss_test:    781.40 
    dim_hid:    2  penalty:     1.00000000 evidence:  -1517.19 loss_train:   3131.02 loss_test:   2026.05 
    dim_hid:    2  penalty:    10.00000000 evidence:   -360.24 loss_train:    772.09 loss_test:    200.21 
    dim_hid:    2  penalty:   100.00000000 evidence:   -492.58 loss_train:   1065.89 loss_test:    296.84 
    Average evidence for 2 hidden variables is -528.89
    dim_hid:    5  penalty:     0.00000001 evidence:   -656.54 loss_train:    659.92 loss_test:    209.60 
    dim_hid:    5  penalty:     0.00001000 evidence:   -455.38 loss_train:    658.62 loss_test:    196.34 
    dim_hid:    5  penalty:     0.00010000 evidence:   -372.82 loss_train:    664.52 loss_test:    186.70 
    dim_hid:    5  penalty:     0.00100000 evidence:   -396.01 loss_train:    669.88 loss_test:    180.97 
    dim_hid:    5  penalty:     0.01000000 evidence:   -387.02 loss_train:    666.12 loss_test:    164.11 
    dim_hid:    5  penalty:     0.10000000 evidence:   -329.33 loss_train:    669.39 loss_test:    157.40 
    dim_hid:    5  penalty:     1.00000000 evidence:   -350.58 loss_train:    698.85 loss_test:    181.49 
    dim_hid:    5  penalty:    10.00000000 evidence:   -346.29 loss_train:    754.86 loss_test:    168.26 
    dim_hid:    5  penalty:   100.00000000 evidence:   -469.17 loss_train:   1021.86 loss_test:    297.56 
    Average evidence for 5 hidden variables is -418.12
    dim_hid:   10  penalty:     0.00000001 evidence:   -736.65 loss_train:    603.23 loss_test:    243.34 
    dim_hid:   10  penalty:     0.00001000 evidence:   -574.01 loss_train:    643.53 loss_test:    216.76 
    dim_hid:   10  penalty:     0.00010000 evidence:   -523.29 loss_train:    717.26 loss_test:    210.48 
    dim_hid:   10  penalty:     0.00100000 evidence:   -454.41 loss_train:    617.84 loss_test:    205.06 
    dim_hid:   10  penalty:     0.01000000 evidence:   -394.40 loss_train:    657.95 loss_test:    220.28 
    dim_hid:   10  penalty:     0.10000000 evidence:   -363.81 loss_train:    620.53 loss_test:    226.50 
    dim_hid:   10  penalty:     1.00000000 evidence:   -351.10 loss_train:    681.07 loss_test:    196.11 
    dim_hid:   10  penalty:    10.00000000 evidence:   -325.45 loss_train:    679.49 loss_test:    158.09 
    dim_hid:   10  penalty:   100.00000000 evidence:   -453.78 loss_train:   1019.95 loss_test:    333.66 
    Average evidence for 10 hidden variables is -464.10
    dim_hid:   25  penalty:     0.00000001 evidence:  -1120.89 loss_train:    661.25 loss_test:    261.48 
    dim_hid:   25  penalty:     0.00001000 evidence:   -680.72 loss_train:    642.70 loss_test:    224.28 
    dim_hid:   25  penalty:     0.00010000 evidence:   -591.49 loss_train:    663.32 loss_test:    206.81 
    dim_hid:   25  penalty:     0.00100000 evidence:   -528.53 loss_train:    694.61 loss_test:    230.91 
    dim_hid:   25  penalty:     0.01000000 evidence:   -447.23 loss_train:    664.28 loss_test:    260.64 
    dim_hid:   25  penalty:     0.10000000 evidence:   -402.18 loss_train:    679.08 loss_test:    268.98 
    dim_hid:   25  penalty:     1.00000000 evidence:   -324.37 loss_train:    632.49 loss_test:    202.41 
    dim_hid:   25  penalty:    10.00000000 evidence:   -316.28 loss_train:    732.56 loss_test:    134.60 
    dim_hid:   25  penalty:   100.00000000 evidence:   -404.03 loss_train:   1008.48 loss_test:    295.41 
    Average evidence for 25 hidden variables is -535.08
    dim_hid:   50  penalty:     0.00000001 evidence:  -1721.87 loss_train:    636.41 loss_test:    228.12 
    dim_hid:   50  penalty:     0.00001000 evidence:   -794.76 loss_train:    626.81 loss_test:    237.96 
    dim_hid:   50  penalty:     0.00010000 evidence:   -720.51 loss_train:    648.92 loss_test:    262.43 
    dim_hid:   50  penalty:     0.00100000 evidence:   -571.81 loss_train:    620.49 loss_test:    256.97 
    dim_hid:   50  penalty:     0.01000000 evidence:   -499.19 loss_train:    644.63 loss_test:    222.60 
    dim_hid:   50  penalty:     0.10000000 evidence:   -388.95 loss_train:    654.41 loss_test:    240.41 
    dim_hid:   50  penalty:     1.00000000 evidence:   -305.43 loss_train:    668.08 loss_test:    191.33 
    dim_hid:   50  penalty:    10.00000000 evidence:   -249.68 loss_train:    732.74 loss_test:    157.55 
    dim_hid:   50  penalty:   100.00000000 evidence:   -277.11 loss_train:    943.14 loss_test:    285.10 
    Average evidence for 50 hidden variables is -614.37
    dim_hid:  100  penalty:     0.00000001 evidence:  -2328.20 loss_train:    630.10 loss_test:    301.71 
    dim_hid:  100  penalty:     0.00001000 evidence:  -1119.58 loss_train:    649.91 loss_test:    258.75 
    dim_hid:  100  penalty:     0.00010000 evidence:   -874.78 loss_train:    676.37 loss_test:    243.17 
    dim_hid:  100  penalty:     0.00100000 evidence:   -686.63 loss_train:    630.66 loss_test:    207.71 
    dim_hid:  100  penalty:     0.01000000 evidence:   -504.46 loss_train:    669.84 loss_test:    222.03 
    dim_hid:  100  penalty:     0.10000000 evidence:   -340.08 loss_train:    620.15 loss_test:    179.03 
    dim_hid:  100  penalty:     1.00000000 evidence:   -192.07 loss_train:    638.93 loss_test:    183.10 
    dim_hid:  100  penalty:    10.00000000 evidence:   -108.04 loss_train:    791.90 loss_test:    172.25 
    dim_hid:  100  penalty:   100.00000000 evidence:    -78.79 loss_train:    996.19 loss_test:    254.96 
    Average evidence for 100 hidden variables is -692.52
    


```python
plt.scatter(evidence_trace, generror_trace)
plt.title('Bayes evidence vs Test error')
plt.xlabel('Evidence')
plt.ylabel('Test error')
```




    <matplotlib.text.Text at 0x1f21eb1e4a8>




![png](output_18_1.png)



```python
plt.scatter(evidence_trace, generror_trace)
plt.title('Bayes evidence vs Test error (Zoom)')
plt.xlabel('Evidence')
plt.ylabel('Test error')
plt.ylim(100, 600)
plt.xlim(-1600, -150)
```




    (-1600, -150)




![png](output_19_1.png)

