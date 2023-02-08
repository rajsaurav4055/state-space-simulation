import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

A = [[0.0, 1.0, 0.0],[0.0, 0.0, 1.0],[-11.0, -17.0, -5.0]]
B = [[0.0],[0.0],[1.0]]
C = [13.0, 9.0, 0.0]
D = [0.0]
sys = signal.StateSpace(A,B,C,D)

#Define step input signal
t = np.linspace(0,10,1000)
u_step = np.ones_like(t)
u_step[t < 1] = 0
t,y_step,x_step= signal.lsim(sys,u_step,t)

#Ramp input signal
u_ramp = t
u_ramp[t < 1] = 0

#Simulate ramp response
t,y_ramp,x_ramp = signal.lsim(sys,u_ramp, t)


#Plot the step responses
plt.subplot(4,1,1)
plt.plot(t,y_step, label='Step Input')
plt.title('Response for Step Input Signal')
plt.legend(['y_step','u'],loc='best')
plt.ylabel('Output')
plt.xlabel('Time')
plt.show()

#plot the ramp response
plt.subplot(4,1,3)
plt.plot(t,y_ramp, label='Ramp Input')
plt.title('Response for Ramp Input Signal')
plt.legend(['y_ramp','u'],loc='best')
plt.ylabel('Output')
plt.xlabel('Time')

plt.show()