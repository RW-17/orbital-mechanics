import numpy as np

def trajectory_type(r1v, r2v, t1, t2):
    
    r1 = np.linalg.norm(r1v)
    r2 = np.linalg.norm(r2v)
    mu = 1.32712438e+20
    c = np.linalg.norm(r2v - r1v)
    
    if np.cross(r1v, r2v)[2] < 0:
        case = 'Retrograde'
    else:
        case = 'Direct'
    
    amin = (r1 + r2 + c)/4
    tau = (t2 - t1)*np.sqrt(mu/amin**3)*24*3600
    slambda = np.sqrt((r1 + r2 - c)/(r1 + r2 + c))
    
    q = 1.0
    E1 = np.arccos(1 - 2*q*slambda**2)
    E2 = np.arccos(1 - 2*q)
    
    def orbit(q: float, n: int) -> float:
        
        E1 = np.arccos(1 - 2*q*slambda**2)
        E2 = np.arccos(1 - 2*q)
        if n == 1:
            return ((E2 - np.sin(E2)) - (E1 - np.sin(E1)))/q**(3/2) - tau
        
        if n == 2:
            return ((E2 - np.sin(E2)) + (E1 - np.sin(E1)))/q**(3/2) - tau

        if n == 3:
            return (2*np.pi - (E2 - np.sin(E2)) 
                    + (E1 - np.sin(E1)))/q**(3/2) - tau
        if n == 4:
            return (2*np.pi - (E2 - np.sin(E2)) 
                    - (E1 - np.sin(E1)))/q**(3/2) - tau
            
    
    def derivative(q: float, n: int) -> float:
        
        k1 = 1 - 2*q*slambda**2
        k2 = 1 - 2*q
        k = slambda**2
        num_a = 2*k*(k1 - 1)/np.sqrt(1 - k1**2)
        num_b = 4*q/np.sqrt(1 - k2**2)
        num_c = np.arccos(-k1) + np.sqrt(1 - k1**2)
        num_d = np.arccos(-k2) + np.sqrt(1 - k2**2) 
        
        if n == 1:
            num1 = num_a + num_b
            num2 = 3*(num_c - num_d)
        if n == 2:
            num1 = -num_a + num_b
            num2 = 3*(-num_c - num_d + 2*np.pi)
        if n == 3:
            num1 = -num_a - num_b
            num2 = 3*(-num_c + num_d + 2*np.pi)
        if n == 4:
            num1 = (num_a - num_b)
            num2 = 3*(num_c + num_d)
            
        return num1/q**(1.5) - num2/(2*q**2.5)
            
    tau_d = (E2 - np.sin(E2)) + (E1 - np.sin(E1))
    tau_r = (E1 - np.sin(E1)) + (E2 - np.sin(E2))
    
    if case == 'Direct':
        if tau < tau_d:
            orbit_type = 'I'
            q -= 10e-7
            for i in range(0, 100):
                q = q - orbit(q, 1)/derivative(q, 1)
        else:
            orbit_type = 'IV'
            q -= 10e-7
            for i in range(0, 100):
                q = q - orbit(q, 4)/derivative(q, 4)
                
    if case == 'Retrograde':
        if tau < tau_r:
            orbit_type = 'II'
            q -= 10e-7
            for i in range(0, 100):
                q = q - orbit(q, 2)/derivative(q, 2)

        else:
            orbit_type = 'III'
            q -= 10e-7
            for i in range(0, 100):
                q = q - orbit(q, 3)/derivative(q, 3)

    return case, c, amin, tau, slambda, orbit_type, q, amin/q, r1, r2

def velocity(r1v, r2v, t1, t2, v_earth: np.ndarray) -> float | np.ndarray:
    
    mu = 1.32712438e+20

    c = trajectory_type(r1v, r2v, t1, t2)[1]
    a = trajectory_type(r1v, r2v, t1, t2)[7]
    r1 = trajectory_type(r1v, r2v, t1, t2)[8]
    r2 = trajectory_type(r1v, r2v, t1, t2)[9]
    orbit_type = trajectory_type(r1v, r2v, t1, t2)[5]
    
    v1 = np.sqrt(mu*(2/r1 - 1/a))
    G = mu/(r1*r2 + np.dot(r1v,r2v))
    A = c**2
    B = -v1**2 + 2*G*np.dot(r2v - r1v, r1v)/r1
    C = G**2
    disc = np.sqrt(B**2 - 4*A*C)
    
    if orbit_type == 'IV':
        eta = np.sqrt((-B - disc)/(2*A))
    if orbit_type == 'II':
        eta = -np.sqrt((-B - disc)/(2*A))
    if orbit_type == 'I':
        eta = np.sqrt((-B + disc)/(2*A))
    if orbit_type == 'III':
        eta = -np.sqrt((-B + disc)/(2*A))
    
    v1v = eta*(r2v - r1v) + G/eta*r1v/r1
    delta_v = np.linalg.norm(v1v - v_earth)
    
    return G, A, B, C, eta, v1v, delta_v
