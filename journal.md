# A Journal of Thoughts for This Project

## Mar. 10, 2018

Training is taking wayyyy to long. Last time, I tried increasing the learning rate from $10^{-4}$ to $10^{-3}$--but that was just preventing it from converging at all. Apparently this is just an issue with PyTorch's ADAM optimization implementation, so it's not a simple fix, by any means. 

Simplest solution: write a stopwatch to time how long every operation takes. It will be the same stopwatch as the one we implemented in Production Quality Software--just in Python instead of Java. The method descriptions are below. 
* `start()` will start the stopwatch;
* `stop()` will stop the stopwatch;
* `lap()` will record a lap time (defined as the time difference between now and the last time `lap()` was called, or--if `lap()` hasn't been called--the time that `start()` was called), assuming the stopwatch has been started and is not currently stopped;
* `lap_times()` will return the list of lap times. 
The stopwatch will fail gracefully--that is, we won't raise an exception if a method is called at the wrong time (i.e., someone calls `stop()` or `lap()` before they've called `start()`, or someone calls `lap()` when the stopwatch is stopped). 

The way we implemented this in Java was by keeping track of the last recorded time (the last time `start()` was called or the last time `lap()` was called--whichever is later). 

There's a problem in this definition. My use case is as follows: `start(), lap(), lap(), ..., stop(), start(), lap(), ...`. It's the sequence `stop(), start(), lap()` that is the troublesome issue here. In that instance, I need to take two differences and add them together in order to get the proper lap time. How did I solve this again? I...don't know. 

Let's think. What if we stored a running total of the current lap time, *and* the last time that either `start()` or `lap()` was called? This would allow us to track the lap time between stopping and starting the stopwatch. Let's try this. 

Okay, the running total worked in combination with a state variable `_is_stopped`. Separately, I need to consider whether to use `time.perf_counter()`, which gets system time, or `time.process_time()`, which gets the total time the current process has been running (i.e., this doesn't include time waiting for other processes to finish on the CPU while the current process waits). Since the CUDA code runs (I think) in a separate process--or, at least, it probably isn't included in the current process' time by whatever underlying OS utilities exist), I think it's best to use system time. So, we'll use `time.perf_counter()`. 

Now, on to actually timing a few different portions of training. 

So I timed the following code segments, for each update step:
* the whole iteration;
* the attack (generating a batch of adversarial examples);
* the forward pass through the network;
* the backward pass through the network; and
* the network update.
The results are below:
> step: [2.500127602000248, 2.432737095000448, 2.378695799000525, 2.361024169999837, 2.368802637999579]
  setup: [0.00775164000060613, 0.00043373199969209963, 0.005929208999987168, 0.005944301999988966, 0.005837287999383989]
  attack: [2.490778893999959, 2.4309864920005566, 2.3714755950004474, 2.353649265000058, 2.361683899000127]
  forward: [0.00047837999954936095, 0.0003414690008867183, 0.0003465170002527884, 0.0004134620003242162, 0.00039096100044844206]
  backward: [0.0005620250003630645, 0.0005620550000458024, 0.0005325439997250214, 0.0005207569993217476, 0.00048380599946540315]
  update: [0.0004085689997737063, 0.00031210599991027266, 0.0003128460002699285, 0.00036908299989590887, 0.00030489300024783006]
These results make it clear that the attack is responsible for the most time spent in the update step. 

This is expected, since the bulk of network training is the network computation itself, and there are `batch_size=50` consecutive attacks (since they can't be run as a single computation on GPU; thanks PyTorch), each with `num_steps=40`, making $50\cdot 40 = 2000$ consecutive backpropagation and update steps. This is a *massive* slowdown. It also explains the relationship between the forward/backward pass times and the attack time: taking the latest iteration, we have $2000(\mathrm{forward time}\ +\ \mathrm{backward time}) = 2000*(.0009) \approx 2$, which is close to $2.36$. (As a side note, the discrepancy between $2$ and $2.36$ indicates that there are not-insignificant time savings to be achieved in the attack function itself.)

Now that we know this is the bulk of our computation, we have decision to make. Either we time the existing attack function, or we start implementing the replay memory. Either way, we have to break this script into a full-fledged Python project. 

## Mar. 11, 2018

I've decided to keep the project structure simple: a top-level directory with a few modules. This isn't going to be a package that someone will install; it's a script that people may download and run. This means that it does not need any requirements.txt, setup.py, or other file which is only relevant when installing a module and/or when managing a large project. 

That being said, how am I breaking up my code?
* I'll put the network `SimpleCNN` in `models.py`--but that seems odd, since the project really isn't about the models. However, that's the nomenclature, so that's what I'll use. 
* I'd like to put the attacks in two separate modules, since they each have a ton of test functions: `pgd_attack` and `fgsm_attack`. 
* The class `Stopwatch` should go in a module named `timing.py`. 
* The loading of MNIST training data: that should go in a `utilities.py` module. 
* The training functions should go in a file `training.py`. 
Having said all this, let me sit on it for a bit. Project structure is important for workflow. 

What's my goal? First, to reproduce the original guy's work--but that's taking too long, so I need a different training method to make things faster. So I'll want to put multiple training methods in my project, which means my training methods should be in their own module file. An alternative to this is that, since training a model is more like my project's `run` method (i.e., if someone else looks at the project, this is the module they execute), each training method might have its own file. Regardless of which of those two options I'm choosing, though, loading the data will be separated from the training process, and so it should go in its own `utilities.py` module. 

However, when someone sees this project, what do I want them to be able to do? I want them to be able to load a model and test its performance against various adversarial attacks, in addition to training a model from scratch. Hmmm. That's a later iteration of the project, though. For now, just having different training with different scripts is just fine. 

Given all of that, I will now make the project structure as below. 
* `SimpleCNN` in `models.py`. 
* `pgd_attack` and `fgsm_attack` in separate modules. 
* `Stopwatch`--and the loading of MNIST data--goes in `utilities.py`. 
* Each training method will be put in its own script, to be run separately. 

Side note: I want training methods to periodically save the models during training. I haven't done this yet. Within each training script, if I put the training inside of a method, I need to abstract the metaparameters in the training--or do I? The training is meant to be a script. 

Hmmm. The fundamental issue is this: how to I track various different variations of the training procedure which I'm trying out? There are really two different things which I want to track: experiments, and code development for each experiment. Then, for each experiment, there are (first) the many small variations and (second) the final experiment which you show the world. How do you track these things? Obviously, the small variations come first. Hmmm. It's not clear how to track this stuff in GitHub easily. One simple option is to use naming conventions for branches: `xxx-experiments`, and then `xxx-experiment-yyy` for different branches. 

Maybe I'm trying to generalize the workflow of a (computer) scientist too much, when it can't really be generalized. It's probably more appropriate to come up with a model for my own workflow on this project. I'll keep the master branch as the official, this-is-what-I-want-the-world-to-see code--as it should be. This will contain finalized experiments only. There will be two types of branches: development branches, and experiment branches. Development branches will be dealing with writing significant new chunks of code, refactoring existing code (i.e., software development stuff). Experiment branches will be dedicated to running a thousand small variations on the same experiment. 

Hmmmm. Maybe I can split this up. I've been assuming that I have to use all Jupyter, or no Jupyter. Maybe I can do a hybrid. I use Git as normal for development work, but then I use a notebook to run and capture my experiments. 

I'll sit and think about this some more, but I like this last idea in particular. It's a good idea. 

Alright, after sitting, I still think it's a good idea, so I'll go with it. I'll use Jupyter notebooks to run experiments, and I'll use Git to keep track of development on the code base. 
