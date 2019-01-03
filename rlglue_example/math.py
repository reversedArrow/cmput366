import numpy as np 
import matplotlib.pyplot as plt 



ph = 0.55

V = np.zeros(101)

Policy = np.zeros(101)

delta = 1

theta = 0.0001

while delta > theta:
    delta = 0

    for s in range(1,100):
        v = V[s]
        reward = 0
        action = np.zeros(min(s,100-s)+1)
        for a in range(1,min(s,100-s)+1):
            if s+a >= 100:
                reward = 1

            action[a] = ph*(reward + V[s+a]) + (1-ph)*V[s-a]
        V[s] = action[1:].max()

        Policy[s] = action[1:].argmax() + 1 

        delta = max(delta,abs(v-V[s]))

plt.subplot(2,1,1)
plt.plot(np.arange(1,100),V[1:-1])

plt.subplot(2,1,2)
plt.step(np.arange(1,100),Policy[1:-1])

plt.show()



