from ctypes import *
import numpy as np

n_beta=10

class vect(Structure):
    _fields_ = [
        ('theta', c_double),
        ('phi', c_double),
        ('r', c_double),
    ]
    def __init__(self, t, p, r):
        self.theta=t
        self.phi=p
        self.r=r
    def __str__(self):
        return str(self.theta)+" "+str(self.phi)+" "+str(self.r)
        

class kite(Structure):
    _fields_ = [
        ('position', vect),
        ('velocity', vect),
        ('wind', c_void_p),
        ('C_l', c_double),
        ('C_d', c_double),
        ('psi', c_double),
        ('time', c_double)
    ]
        
        
    def __init__(self, initial_pos, initial_vel, wind_type, C_l, C_d, psi,continuous,time=0):
        self.position=initial_pos
        self.velocity=initial_vel
        self.C_l=C_l
        self.C_d=C_d
        self.psi=psi
        self.time=time
        self.continuous=continuous
        if wind_type=='turbo':
            libkite.init_turbo_wind(pointer(self))
        if wind_type=='lin':
            libkite.init_lin_wind(pointer(self), 5, 0.250)
        if wind_type=='const':
            libkite.init_const_wind(pointer(self), 10)
    def reset(self, initial_pos, initial_vel, wind_type):
        self.position=initial_pos
        self.velocity=initial_vel
        self.time=0
        if wind_type=='turbo':
            libkite.reset_turbo_wind(pointer(self))
        if wind_type=='lin':
            libkite.init_lin_wind(pointer(self), 5, 0.250)
            
    def __str__(self):
        return "Position: "+str(self.position.theta)+","+str(self.position.phi)+","+str(self.position.r)+", Velocity"+ str(self.velocity.theta)+","+str(self.velocity.phi)+","+str(self.velocity.r)
    def simulate(self, step, force_):
        return libkite.simulation_step(pointer(self), step,force_)
   # def evolve_system(self, attack_angle, bank_angle, integration_steps, step):
     #   self.update_coefficients(attack_angle, bank_angle)
       # return libkite.simulate(pointer(self), integration_steps, step)
    def evolve_system_2(self,integration_steps, step, force_):
        return libkite.simulate(pointer(self), integration_steps, step, force_)
    def beta(self):#, continuous=True):
        if self.continuous:
            return libkite.getbeta(pointer(self))
        else:
            return max(min(np.digitize(libkite.getbeta(pointer(self)), np.linspace(-np.pi/2, np.pi/2, n_beta+1))-1, n_beta-1), 0)
    def alt(self, continuous=True):       #qui dovevo usare continuous = True 
        altitude=self.position.r*np.cos(self.position.theta)
        if continuous:
            return altitude
        else:
            return np.digitize(altitude, np.linspace(0, 100, n_alt))
    def vrel(self):
        a=libkite.getvrel(pointer(self))
        return a.theta, a.phi, a.r
    def accelerations(self,force_):
        a=libkite.getaccelerations(pointer(self),force_)
        return a.theta, a.phi, a.r
    def reward(self, learning_step, force_):
        return libkite.getreward(pointer(self),force_)*learning_step
    def update_coefficients(self, attack_angle, bank_angle):
        self.C_l, self.C_d = coefficients[attack_angle,0], coefficients[attack_angle,1]
        self.psi = np.deg2rad(bank_angles[bank_angle])
    def update_coefficients_cont(self, c_l,c_d, bank_angle):
        self.C_l, self.C_d = c_l, c_d
        self.psi = np.deg2rad(bank_angle)#[bank_angle])
    def fullyunrolled(self):
        return self.position.r>100
        
    def fullyrolled(self): 
        return self.position.r<23
    def v_kite(self):
        return libkite.getvelocity(pointer(self))
    def force_on_kite_r(self): 
        return libkite.getforce_r(pointer(self))
    def tether_tension(self,force_): 
        return libkite.get_tension(pointer(self),force_)
    def effective_wind_speed(self): 
        return libkite.get_effective_wind_speed(pointer(self))

def setup_lib(lib_path):
    lib = cdll.LoadLibrary(lib_path)
    lib.simulation_step.argtypes = [POINTER(kite), c_double, c_double]
    lib.simulation_step.restype = c_int
    lib.simulate.argtypes =[POINTER(kite), c_int, c_double,c_double]   #aggiunto un c_doube qui oer assare la forza quando faccio partire simulate 
    lib.simulate.restype=c_int
    lib.init_const_wind.argtypes=[POINTER(kite), c_double]
    lib.init_lin_wind.argtypes=[POINTER(kite), c_double, c_double]
    lib.getbeta.argtypes = [POINTER(kite)]
    lib.getbeta.restype=c_double
    lib.getvrel.argtypes = [POINTER(kite)]
    lib.getvrel.restype=vect
    lib.getaccelerations.argtypes = [POINTER(kite),c_double]
    lib.getaccelerations.restype=vect
    lib.getreward.argtypes = [POINTER(kite),c_double]
    lib.getreward.restype=c_double
    lib.getvelocity.argtypes=[POINTER(kite)]
    lib.getvelocity.restype=c_double
    lib.getforce_r.argtypes=[POINTER(kite)]
    lib.getforce_r.restype=c_double
    lib.get_tension.argtypes=[POINTER(kite),c_double]
    lib.get_tension.restype=c_double
    lib.get_effective_wind_speed.argtypes = [POINTER(kite)] 
    lib.get_effective_wind_speed.restype=c_double
    return lib
    
    return lib

libkite=setup_lib("./libkite.so")            

