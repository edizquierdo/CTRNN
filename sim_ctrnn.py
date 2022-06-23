import ctrnn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

size = int(sys.argv[1])
duration = float(sys.argv[2])
stepsize = float(sys.argv[3])

time = np.arange(0.0,duration,stepsize)

nn = ctrnn.CTRNN(size)

nn.randomizeParameters()

nn.initializeState(np.zeros(size))

outputs = np.zeros((len(time),size))

# Run simulation
step = 0
for t in time:
    nn.step(stepsize)
    outputs[step] = nn.Outputs
    step += 1

# Plot activity
for i in range(size):
    plt.plot(time,outputs)
plt.xlabel("Time")
plt.ylabel("Output")
plt.title("Neural activity")
plt.show()

nn.save("ctrnn")
