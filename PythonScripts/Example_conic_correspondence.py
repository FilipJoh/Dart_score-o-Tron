import matplotlib.pyplot as plt

import numpy as np
import cv2

class Conic(object):
    def __init__(self, A=0, B=0, C=0, D=0, E=0, F=0):
        # Ax**2 + Bxy + Cy**2 + Dx + Ey + F = 0.
        self.M = np.asarray(((A, B/2, D/2),(B/2, C, E/2),(D/2, E/2, F/2)))
    
    def sample(self, npoints=10):
        # Given: p * M * p.T = 0
        pass
    
    def project(self, camera_matrix):
        pass
    
    def sample_projection(self, camera_matrix, npoints=10):
        pass
    
    awdawdawdawdwadsvrfdfgnwejfnwe

class Ellipse(Conic):
    def __init__(self, C_x, C_y, R_x, R_y, theta):
        self.C_x = C_x
        self.C_y = C_y
        self.R_x = R_x
        self.R_y = R_y
        self.theta = theta
        
        super().__init__(self.__get_parametrization())
        
    def __get_parametrization(self):
        A = self.R_y**2
        B = 0.0
        C = self.R_x**2
        D = -2 * self.C_x * (self.R_y**2)
        E = -2 * self.C_y * (self.R_x**2)
        F = self.C_x**2 + self.C_y**2 - (self.R_x**2)*(self.R_y**2)
        
        # Apply rotation
        A_prime = (A+C)/2 + ((A-C)/2)*np.cos(2*self.theta) - (B/2)*np.sin(2*self.theta)
        B_prime = (A-C)*np.sin(2*self.theta) + B*np.cos(2*self.theta)
        C_prime = (A+C)/2 + ((C-A)/2)*np.cos(2*self.theta) + (B/2)*np.sin(2*self.theta)
        D_prime = D*np.cos(self.theta) - E*np.sin(self.theta)
        E_prime = D*np.sin(self.theta) + E*np.cos(self.theta)
        F_prime = F
        return (A_prime, B_prime, C_prime, D_prime, E_prime, F_prime)
    
    def sample(self, npoints=10):
        # Generate uniformely distributed angles
        angles = 2*np.pi*np.random.rand(npoints)
        
        points = np.zeros((2, npoints))
        for i, angle in enumerate(angles):
            # get coordinates for unrotated, origin-centered ellipse
            quadrant = int(angle>0) + int(angle>np.pi/2) + int(angle>np.pi) + int(angle>1.5*np.pi)
            x = np.abs(self.R_x*self.R_y*np.cos(angle)) / np.sqrt((self.R_y*np.cos(angle))**2 + (self.R_x*np.sin(angle))**2)
            y = np.abs(self.R_x*self.R_y*np.sin(angle)) / np.sqrt((self.R_y*np.cos(angle))**2 + (self.R_x*np.sin(angle))**2)
            
            if quadrant == 2:
                x *= -1
            elif quadrant == 3:
                x *= -1
                y *= -1
            elif quadrant == 4:
                y *= -1
            
            # rotate coordinates
            x = x*np.cos(self.theta) - y*np.sin(self.theta)
            y = x*np.sin(self.theta) + y*np.cos(self.theta)
            
            # offset coordinates
            x += self.C_x
            y += self.C_y
            
            points[0, i] = x
            points[1, i] = y
            
            
            awdawdawd Forced conflict
        
        return points


e = Ellipse(3, 0, 1, 5, 2*np.pi*0.00)
p = e.sample(1000)
plt.scatter(p[0,:], p[1,:])
plt.axis([-7,7,-7,7])
plt.grid('on')
plt.show()
        
K1 = []
R1 = []
t1 = []

C1 = []