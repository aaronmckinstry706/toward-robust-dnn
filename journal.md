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
