import numpy as np

def initial_parameters(r0v, v0v: np.ndarray) -> np.ndarray | float:
    
    def zero_pi(x: float) -> float:
        if (x < 0):
            x += 2*np.pi
        return x
    
    mu = 8
    hv = np.cross(r0v, v0v)
    h = np.linalg.norm(hv)
    uh = hv/h
    r0 = np.linalg.norm(r0v)
    ev = -r0v/r0 - 1/mu*np.cross(hv, v0v)
    e = np.linalg.norm(ev)
    ue = ev/e
    up = np.cross(uh, ue)
    
    i = np.arccos(hv[2]/h)
    ascnode = zero_pi(np.arctan2(hv[0], -hv[1]))
    omega = zero_pi(np.arctan2(ev[2]/np.sin(i), 
                               ev[0]*np.cos(ascnode) + ev[1]*np.sin(ascnode)))
    i = np.rad2deg(i)
    ascnode = np.rad2deg(ascnode)
    omega = np.rad2deg(omega)
    
    p = h**2/mu
    a = p/(1 - e**2)
    b = a*np.sqrt(1 - e**2)
    rmin = p/(1 + e)
    rmax = p/(1 - e)
    n = np.sqrt(mu/a**3)
    T = 2*np.pi/n
    
    Qxp = np.array([ue, up, uh])
    r0v_per = np.matmul(Qxp, r0v)
    x0_per = r0v_per[0]
    y0_per = r0v_per[1]
    
    f0 = np.arccos((p/r0 - 1)/e)
    E0 = np.arccos((1 - r0/a)/e)
    M0 = E0 - e*np.sin(E0)
    
    tau = -M0/n
    if tau < 0:
        tau_p = tau + T
    else:
        tau_p = tau
        
    return (uh, ue, up, i, ascnode, omega, a, e, b, rmin, rmax, n, T,
            x0_per, y0_per, f0, E0, M0, tau_p, tau, Qxp)

def final_values(r0v, v0v: np.ndarray, delta: float) -> np.ndarray | float:
    
    z = initial_parameters(r0v, v0v)
    a = z[6]
    n = z[11]
    e = z[7]
    tau = z[19]
    Qxp = z[20]
    
    M = n*(delta - tau)
    E = np.pi
    
    for i in range(0, 100):
        E = M + e*np.sin(E)
    
    dE = n/(1 - e*np.cos(E))
    x_per = a*(np.cos(E) - e)
    y_per = a*np.sin(E)*np.sqrt(1 - e**2)
    xdot_per = -a*np.sin(E)*dE
    ydot_per = a*np.cos(E)*np.sqrt(1 - e**2)*dE
    
    r = np.matmul(Qxp.T, np.array([x_per, y_per, 0]))
    v = np.matmul(Qxp.T, np.array([xdot_per, ydot_per, 0]))
    
    return (M, E, dE, x_per, y_per, xdot_per, ydot_per, r, v)
