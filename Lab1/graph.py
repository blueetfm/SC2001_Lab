import numpy as np
import matplotlib.pyplot as plt

x=np.array(range(1,101))
y=100000*x+100000*(np.log2(100000/x))
f,axes = plt.subplots(1,1,figsize=(16,10))
plt.plot(x,y)
plt.xlabel('Size of S, S')
plt.ylabel('Time Complexity, W(n)')
plt.title('Theoretical W(n) against S', fontsize=20)
plt.show()