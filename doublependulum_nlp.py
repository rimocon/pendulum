import numpy as np 
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy import sin,cos

# 初期値
G = 9.8     #[m/s^2]
L1 = 1.0    #[m]
L2 = 1.0    # [m]
M1 = 1.0    # [kg]
M2 = 1.0    # [kg]
K1 = 1.0    # Coefficient of friction
K2 = 1.0    # Coefficient of friction

A1 = 6.3
A2 = 8.1
B0 = 4.4
B = 21.8
freq = 1/14

def difeq(t,state):
    #tau1 = B0 + B*sin(t)
    # tau1 = A1*np.sign(np.sin(2*np.pi*freq*t))
    # tau2 = A2*np.sign(np.sin(2*np.pi*freq*t))
    
    tau1 = 17.84
    tau2 = 9.5

    a11 = M2*L2**2 + (M1+M2)*L1**2 + 2*M2*L1*L2*cos(state[2])
    a12 = M2*L2**2 + M2*L1*L2*cos(state[2])
    a21 = a12
    a22 = M2*L2**2
    b1 = (tau1 + M2*L1*L2*sin(state[2])*state[3]**2
    + 2*M2*L1*L2*sin(state[2])*state[1]*state[3]
    - M2*L2*G*cos(state[0]+state[2])
    - (M1 + M2)*L1*G*cos(state[0])
    - K1*state[1])
    b2 = (tau2 - M2*L1*L2*sin(state[2])*state[1]**2
    - M2*L2*G*cos(state[0]+state[2]) 
    - K2*state[3])
    delta = a11*a22 - a12*a21

    dzdx = np.zeros_like(state)
    dzdx[0] = state[1]
    dzdx[1] = (b1*a22 - b2*a12) / delta
    dzdx[2] = state[3]
    dzdx[3] = (b2*a11 - b1*a21) / delta
    return dzdx


# initial value
th1 = 0.0
w1 = 0.0
th2 = 0.0
w2 = 0.0
state = np.radians([th1, w1, th2, w2])

# 微分間隔と時間設定
dt = 0.01
t_span = [0,20]
t = np.arange(t_span[0],t_span[1],dt)

# solverで解く
solver = solve_ivp(difeq,t_span,state,t_eval = t)
y = solver.y
# 確認用
print(y)

'''
fig,ax =plt.subplots()
ax.plot(y[0], y[1],'b,',label='theta1,Dtheta1')
ax.plot(y[2], y[3],'r,',label='theta2,Dtheta2')
# ax.set_xlim(-0.2,0.4)
# ax.set_ylim(-0.06,-0.05)
plt.xlabel('theta1')
plt.ylabel('Dtheta 1')
plt.legend()
plt.title('double pendulum')
plt.show()
'''

#animation
def gen():
    for tt,th1,th2 in zip(t,y[0,:],y[2,:]):
        x1 = L1*cos(th1)
        y1 = L1*sin(th1)
        x2 = L2*cos(th1+th2) + x1
        y2 = L2*sin(th1+th2) + y1
        yield tt,x1,y1,x2,y2

fig, ax = plt.subplots()
ax.set_xlim(-(L1+L2), L1+L2)
ax.set_ylim(-(L1+L2), L1+L2)
ax.set_aspect('equal')

# locusが軌跡,lineが線
locus, = ax.plot([], [], 'b-', linewidth=2)
line, = ax.plot([], [], 'ro-', linewidth=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
xlocus, ylocus= [], []

def animate(data):
    t, x1, y1, x2, y2 = data
    xlocus.append(x2)
    ylocus.append(y2)

    locus.set_data(xlocus, ylocus)
    line.set_data([0, x1, x2], [0, y1, y2])
    time_text.set_text(time_template % (t))
    return [locus,line,time_text]

# 15sのアニメーション
ani = FuncAnimation(fig, animate, gen, interval = 0.0,blit=True,save_count = len(t))

# 秒感100frame
# 動画を保存する場合コメントアウト
# ani.save('double.mp4',writer='ffmpeg',fps = 200)
plt.show()