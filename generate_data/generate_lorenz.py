# runge kutta 4 explicit
def RK4(yt,h,f):
    k1=h*f(yt)
    k2=h*f(yt+0.5*k1)
    k3=h*f(yt+0.5*k2)
    k4=h*f(yt+k3)
    dy=(k1+2*k2+2*k3+k4)/6
    return yt+dy

## Equations of the Lorenz system
def l63(Xt):
    import numpy as np
    sigma=10
    rho=28
    beta=8/3
    xdot=sigma*(Xt[1]-Xt[0])
    ydot=Xt[0]*(rho-Xt[2])-Xt[1]
    zdot=Xt[0]*Xt[1]-beta*Xt[2]
    return np.array([xdot,ydot,zdot])