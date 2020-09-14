import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from numpy import sin,cos
G = 9.8
M = 1.0
L = 1.0
b = 0.0
u0 = 7.054
def func_dzdt(t,state):
    dzdx = np.zeros_like(state)
    dzdx[0] = state[1]
    dzdx[1] = (-M*G*L*sin(state[0]) -b *state[1] + u0) / (M*L*L)  
    return dzdx

t_span = [0,20]

state = np.radians([0,0])
dt = 0.001
t = np.arange(t_span[0],t_span[1],dt)
solver = solve_ivp(func_dzdt,t_span,state,t_eval = t)
y = solver.y

print(y)
plt.plot(y[0],y[1])
plt.xlabel('theta')
plt.ylabel('Dtheta')
plt.title('single pendulum')
plt.show()


def gen():
    for tt,theta,omega in zip(t,y[0,:],y[1,:]):
        x1 = L*sin(theta)
        y1 = -L*cos(theta)
        if abs(omega) <0.01:
            eqx1 = x1
            eqy1= y1
          # print(np.rad2deg(theta))
        yield tt,x1,y1,eqx1,eqy1

fig, ax = plt.subplots()
ax.set_xlim(-L, L)
ax.set_ylim(-L, L)
ax.set_aspect('equal')


eqpoint, = ax.plot([], [],'bo-', linewidth=2)
line, = ax.plot([], [], 'ro-', linewidth=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
eqpoint_x, eqpoint_y= [], []
def animate(data):
    t, x1, y1,eqx1,eqy1= data
    # eqpoint_x.append([0,eqx1])
    # eqpoint_y.append([0,eqy1])
    # eqpoint.set_data(eqpoint_x,eqpoint_y)
    

    line.set_data([0, x1], [0, y1])
    time_text.set_text(time_template % (t))
    return [eqpoint,line,time_text]

ani = FuncAnimation(fig,animate,gen, interval = 1,blit=True,save_count=len(t))
# ani.save('single.mp4',writer='ffmpeg')
plt.show()
