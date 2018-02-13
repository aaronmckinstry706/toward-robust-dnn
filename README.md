# toward-robust-dnn
A project to reproduce and extend the work done in "Towards Deep Neural Network Architectures Robust to Adversarial Examples" (at https://arxiv.org/abs/1412.5068 ). 

The first step in this set of experiments is to reproduce the results of the original paper. To this end, I'm re-implementing their networks in pytorch. Below, I'm taking notes on the exact parameters, architecture, etc. that they used. 

For what I have in mind, the simplest thing to do is to first reproduce the results of Figure 4 on MNIST.

## The Nearly-Exact Parameters of Madry et al.'s Experiments for Figure 4

MNIST architecture:

> The initial network has a convolutional layer with 2 filters, followed by another convolutional layer with 4 filters, and a fully connected hidden layer with 64 units. Convolutional layers are followed by 2x2 max-pooling layers and adversarial examples are constructed with $\varepsilon = 3$. 

What are the filter sizes? According to their code on the [challenge website][https://github.com/MadryLab/mnist_challenge/blob/master/model.py]:

> `W_conv1 = self._weight_variable([5,5,1,32])`
> `W_conv2 = self._weight_variable([5,5,32,64])`

So, the filter sizes are 5x5. (The code quoted above is the largest architecture with which they performed an experiment--namely, the initial architecture but with the number of hidden units and filters scaled up by a factor of 16.)

Next is to figure out training parameters. How, exactly, did they train this model? What were the adversarial attack parameters? 

From the quote before, we know $\varepsilon$ for both FGSM and PGD attack. The question is, what is $\alpha$ in their PGD attack? In the [config][https://github.com/MadryLab/mnist_challenge/blob/master/config.json] file, we see their attack parameters:

> `"epsilon": 0.3,`
> `"a": 0.01,`

Although this is for their final trained network, we will take this as what they did in their original experiments, so for our experiments we assume $\alpha=0.01$. Meanwhile, in the same file, we have their training parameters:

> `"max_num_training_steps": 100000,`
> `"num_output_steps": 100,`
> `"num_summary_steps": 100,`
> `"num_checkpoint_steps": 300,`
> `"training_batch_size": 50,`
> `"num_eval_examples": 10000,`

Now we arrive at their optimization algorithm. From the code:

> `rain_step = tf.train.AdamOptimizer(1e-4).minimize(model.xent, global_step=global_step)`

So they use Adam optimization. Did they use dropout, batch normalization, or any other techniques of which I'm not aware? Their code does not seem to use any such techniques, so I'm assuming that they used a plain CNN in their experiments. 

And with that, our investigations are complete! Time to get coding!

JK JK JK filling in some gaps here. What about the kernel stride? Their code:

> `  def _conv2d(x, W):`
> `      return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')`

Now we also know that using same-padding, too. 
