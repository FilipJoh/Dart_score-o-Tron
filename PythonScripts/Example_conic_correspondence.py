import matplotlib.pyplot as plt
from copy import deepcopy

import numpy as np
import cv2

class Conic(object):
    def __init__(self, C, Z=0):
        # Ax**2 + Bxy + Cy**2 + Dx + Ey + F = 0.
        self.C = C
        self.Z = Z
    
    def sample(self, npoints=10):
        # Given: p * M * p.T = 0
        pass
    
    def pointwise_transform(self, H):
        new_C = np.inv(H).T.dot(self.C).dot(np.inv(H))
        
        discriminant = 2*new_C[0, 1]**2 - 4*new_C[0, 0]*new_C[1, 1]
        if np.det(new_C) == 0:
            ## Degenerate
            new_conic = None
        elif discriminant < 0:
            # Ellipse
            new_conic = Ellipse(0,0,0,0,0)
        elif discriminant == 0:
            # Parabola
            new_conic = None
        elif discriminant > 0:
            # Hyperbola
            new_conic = None
        
        new_conic.C = new_C
        new_conic.Z = self.Z
        new_conic.__set_parametrization()
        return new_conic
    
    def __get_parametrization():
        pass
    
    def __set_parametrization():
        pass
    
    def plot(self):
        pass
        
        
        
        
        
        

class Ellipse(Conic):
    def __init__(self, C_x, C_y, R_x, R_y, theta, Z=0):
        self.C_x = C_x
        self.C_y = C_y
        self.R_x = R_x
        self.R_y = R_y
        self.theta = theta
        
        (A, B, C, D, E, F) = self.__get_parametrization()
        matrix_representation = np.asarray(((A, B/2, D/2),(B/2, C, E/2),(D/2, E/2, F)))
        
        super().__init__(matrix_representation, Z)
        
    def __get_parametrization(self):
        A = (self.R_y**2) * (np.cos(self.theta)**2) + (self.R_x**2) * (np.sin(self.theta))**2
        B = 2 * np.sin(self.theta) * np.cos(self.theta) * (self.R_y**2 - self.R_x**2)
        C = (self.R_y**2) * (np.sin(self.theta))**2 + (self.R_x**2) * (np.cos(self.theta))**2
        D = -(2*A*self.C_x + B*self.C_y)
        E = -(2*C*self.C_y + B*self.C_x)
        F = A*self.C_x**2 + B*self.C_x*self.C_y + C*self.C_y**2 - self.R_x**2*self.R_y**2
        return (1, B/A, C/A, D/A, E/A, F/A)
    
    def set_parametrization(self):
        A = self.C[0, 0]
        B = 2*self.C[0, 1]
        C = self.C[1, 1]
        D = 2*self.C[0,2]
        E = 2*self.C[1,2]
        F = self.C[2,2]
        
#        delta = A*C - (B**2) / 2
#        self.C_x = (0.5*(B)*0.5*(E) - C*0.5*D) / (delta)
#        self.C_y = (0.5*(B)*0.5*(D) - A*0.5*E) / (delta)
        X_0 = np.linalg.solve(np.asarray(((A, B/2), (B/2, C))), np.asarray((-D/2, -E/2)))
        self.C_x = X_0[0]
        self.C_y = X_0[1]
        
#        self.C_x = (1 / (B**2 - 4*A*C)) * (2*C*D - B*E)
#        self.C_y = (1 / (B**2 - 4*A*C)) * (2*A*E - B*D)
        
        M = np.asarray(((A, B/2), (B/2, C)))
        e, v = np.linalg.eig(M)
        self.R_x = np.sqrt(e[1])
        self.R_y = np.sqrt(e[0])
        self.theta = np.arctan(v[1,0] / v[0,0])
        if self.theta < 0:
            self.theta +=  2*np.pi
        pass
    
    def plot(self):
        p = self.sample(1000)
        plt.scatter(p[0,:], p[1,:], label='parametrical')
        
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        for i in range(x.size):
            for j in range(y.size):
                vec = np.asarray([x[i], y[j], 1])
                val = (vec.T).dot(self.C).dot(vec)
                if val < 0:
                    plt.scatter(x[i], y[j], color='black', s=5)
#        p = self.sample(1000)
#        plt.scatter(p[0,:], p[1,:], label='before')
#        plt.axis([-10,10,-10,10])
#        self.__set_parametrization()
#        p = self.sample(1000)
#        plt.scatter(p[0,:], p[1,:], label='after')
#        
        plt.grid('on')
        plt.axis([-10,10,-10,10])
        plt.axhline(y=0, color='k')
        plt.axvline(x=0, color='k')
        plt.legend()
        
        
        
    
    def sample(self, npoints=10):
        # Generate uniformely distributed angles
        angles = 2*np.pi*np.random.rand(npoints)
        
        points = np.zeros((2, npoints))
        for i, angle in enumerate(angles):
            # get coordinates for unrotated, origin-centered ellipse
            x = self.R_x*np.cos(angle)
            y = self.R_y*np.sin(angle)
            
            # rotate coordinates
            x_rot = x*np.cos(self.theta) - y*np.sin(self.theta)
            y_rot = x*np.sin(self.theta) + y*np.cos(self.theta)
            
            # offset coordinates
            x_translated = x_rot + self.C_x
            y_translated = y_rot + self.C_y
            
            points[0, i] = x_translated
            points[1, i] = y_translated
        
        return points
    
    def prent(self):
        print('C_x: {}'.format(self.C_x))
        print('C_y: {}'.format(self.C_y))
        print('R_x: {}'.format(self.R_x))
        print('R_y: {}'.format(self.R_y))
        print('Theta: {}'.format(self.theta))


e = Ellipse(0, 0, 5, 2, 0.1*np.pi)
print('before')
e.prent()
plt.figure()
e.plot()
plt.title('before parameter extraction')
#print('after')
#e.set_parametrization()
#e.prent()
#plt.figure()
#e.plot()
#plt.title('after parameter extraction')
        
#P1 = np.asarray(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)))
#
#K1 = []
#R1 = []
#t1 = []
#
#P2 = []