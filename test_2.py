import numpy as np 
from learning.models import * 
from argparse import ArgumentParser 
import sys 
import os 
import matplotlib.pyplot as plt 
from learning.utils import *
import torch as T
import pandas as pd 
import scipy.interpolate as interpolate
import learning.pykite as pk


ACTION_NOISE = 0.2
ATTACK_INF_LIM = -5 
ATTACK_SUP_LIM = 18 
BANK_INF_LIM = -3 
BANK_SUP_LIM=3
R_A = 10


def test(args): 

    np.random.seed(14)
    
    if args.cpu != 0:
        device ='cpu'
        
    else:
    
        device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') 
    print(device)    
    if(args.range_actions) is not None: 
    
        if(len(args.range_actions[0])>1): 
        
            range_actions = np.array([args.range_actions[0][0],args.range_actions[0][2]])
            range_actions = range_actions.astype(np.float64)
            
        else: 
        
            range_actions=np.array([args.range_actions[0][0],args.range_actions[0][0]])
            range_actions = range_actions.astype(np.float64)
            
    else: 
        
        range_actions = np.array([1,1]) 
        
    force= args.force
    
    #if args.range_force is not None:
    
        #range_force = args.range_force
    
    dir_nets = os.path.join(args.path,"nets")
    
    Cl_angles = read_file('env/coefficients/CL_angle.txt') 
    
    Cl_values = read_file('env/coefficients/CL_value.txt') 
    
    Cd_angles = read_file('env/coefficients/CD_angle.txt') 
    
    Cd_values = read_file('env/coefficients/CD_value.txt') 

    Cl_data = pd.DataFrame({'cl_angles':Cl_angles,'cl_values':Cl_values}) 
    
    Cd_data = pd.DataFrame({'cd_angles':Cd_angles,'cd_values':Cd_values}) 
    
    f_cl  = interpolate.interp1d(Cl_data.cl_angles, Cl_data.cl_values, kind='linear')
    
    f_cd  = interpolate.interp1d(Cd_data.cd_angles, Cd_data.cd_values, kind='linear')
    
    EPISODES = int(args.episodes) 
    
    warmup_ = args.warmup 
    
    c_lr = args.critic_lr
    
    a_lr = args.actor_lr
    
    step = args.step
    
    learning_step= args.step 
    
    integration_step = 0.001 
    
    duration = args.duration
    
    episode_duration= int(duration) 
    
    horizon = int(episode_duration/learning_step) 
    
    integration_steps_per_learning_step=int(learning_step/integration_step)
    
    #dir_name = os.path.join(args.path,"data")
    
    dir_nets = os.path.join(args.path,"nets")
    
    dir_test = os.path.join(args.path,"test")
    
    pict_name = os.path.join(dir_test,"average_reward.png") 
    
    if not os.path.exists(dir_test):
    
        os.makedirs(dir_test)
        
    kWh_history = [] 
    
    score_history = [] 
    
    
    #finish_flag = 0
    initial_velocity = pk.vect(0, 0, 0)
    theta_init = 1.01823
    phi_init =   0.153866
    init_attack = 3.646512171128421
    init_bank = -2.5552693009376526
    c_l = f_cl(init_attack)
    c_d = f_cd(init_attack)
    init_psi = 0.0159
    initial_position = pk.vect(theta_init, phi_init, 100)
    
    k = pk.kite(initial_position, initial_velocity, args.wind,c_l,c_d,init_psi,continuous=True)
    #if args.range_force is not None:
    
        #agent =Agent(5,3,critic_lr=c_lr,actor_lr=a_lr,gamma=1,warmup=0,chkpt_dir=dir_nets,device=device)
        
    #else:
    
    agent =Agent(4,2,critic_lr=c_lr,actor_lr=a_lr,gamma=1,warmup=0,chkpt_dir=dir_nets,device=device)
    agent.load_actor()
    
    time_history = [] 
    
    r = [] 
    
    theta = [] 
    
    phi = [] 
    
    alpha = [] 
    
    bank = [] 
    
    beta = [] 
    
    power = [] 
    
    cumulative_time = []
    
    force_during_sim = []
    
    KWh_per_second = []
    
    eff_wind_speed = []
    int_time = 0
    
    
    for i in range(EPISODES): 
    
        #print(i, end='\r')
    
        done = False 
        
        score = 0 
        
        k.reset(initial_position, initial_velocity, args.wind) 
        
        initial_beta = k.beta() 
        
        #if args.range_force is not None:
        
            #state = (np.random.uniform(ATTACK_INF_LIM,ATTACK_SUP_LIM), np.random.uniform(BANK_INF_LIM,BANK_SUP_LIM), initial_beta,k.position.r,0.)
            
        #else:
        
        state = (np.random.uniform(ATTACK_INF_LIM,ATTACK_SUP_LIM), np.random.uniform(BANK_INF_LIM,BANK_SUP_LIM), initial_beta,k.position.r)
        

        
        c_l = f_cl(state[0])
        
        c_d = f_cd(state[0])
        
        k.update_coefficients_cont(c_l,c_d,state[1])
                    
        state = np.asarray(state)
        
        count = 0
        
        time= 0
        
       # force = 0
        
        kWh = 0
        
        while not done: 
        
            time += learning_step
            
            int_time +=1 
            
            count +=1 
            
            action = agent.choose_action(state,0.0,test=True)
            
            new_attack, new_bank = state[0]+action[0]*range_actions[0], state[1]+action[1]*range_actions[1]
            
            #force += action[2]*range_force
            
            #if args.range_force is not None:
            
               # force += action[2]*range_force
                
               # force = np.clip(force,0,force_)
                
           # else:
            
              #  force= action[2]*force_/2 +force_/2
                
               # force = np.clip(force,0,force_)
                
            #print("force = ",force) 
            new_attack = np.clip(new_attack,ATTACK_INF_LIM,ATTACK_SUP_LIM) 
            
            new_bank = np.clip(new_bank,BANK_INF_LIM,BANK_SUP_LIM)
            
            
            
            c_l = f_cl(new_attack)
            
            c_d = f_cd(new_attack)
            
            k.update_coefficients_cont(c_l,c_d,new_bank)
            
            sim_status = k.evolve_system_2(integration_steps_per_learning_step,integration_step,force)
            
            energy = k.reward(step,force)
            
            kWh +=energy
            
            if i<10: 
            
                r.append(k.position.r) 
                
                theta.append(k.position.theta) 
                
                phi.append(k.position.phi)
            
            if not sim_status == 0: 
            
                finish_flag = 0
                
                reward = 20 -k.position.r 
                 
                #print("Not ended, position = ",k.position.r," episodes = ",i)
                
                done = True 
                
                new_state = state 
                
                print("Episode ",i," final position = ", k.position.r)
                
            else: 
            
               # if args.range_force is not None:
                
                   # new_state = np.asarray((new_attack, new_bank, k.beta(),k.position.r,force))
                    
                #else:
                
                new_state = np.asarray((new_attack, new_bank, k.beta(),k.position.r))
                
                reward=0
                
            if i<10: 
            
                alpha.append(state[0]) 
                bank.append(state[1]) 
                beta.append(state[2]) 
                force_during_sim.append(force)
                power.append((reward/learning_step)*3600)   
                eff_wind_speed.append(k.effective_wind_speed())
                
                
                
                
            
            if (count== int(horizon) -1 or k.fullyrolled()): 
            
                done = 1 
                
                reward = 20-k.position.r #+energy +1/count
                
                if(k.fullyrolled()): 
                
                    reward= 0  #ener
                    
                    print("Episode ",i," finished!, energy consumed :",score)
                    
                else: 
                
                    print("Episode ",i," out of time!, energy consumed :",score,"pos ",k.position.r)
                    
                
            
            
            score +=energy
            
            state = new_state 
            
        score_history.append(score) 
        
        cumulative_time.append(int_time) 
        
        time_history.append(time) 
        
        KWh_per_second.append(score/time)
        
    r = np.array(r)
     
    theta = np.array(theta)
    
    phi = np.array(phi)
     
    alpha = np.array(alpha)
    
    bank = np.array(bank)
    
    beta = np.array(beta)
    
    x=np.multiply(r, np.multiply(np.sin(theta), np.cos(phi)))
    
    y=np.multiply(r, np.multiply(np.sin(theta), np.sin(phi)))
    
    z=np.multiply(r, np.cos(theta))
        
    np.save(os.path.join(dir_test,"durations.npy"), np.array(time_history))
    np.save(os.path.join(dir_test,"cumulative_durations.npy"), np.array(cumulative_time))
    np.save(os.path.join(dir_test,"power.npy"), np.array(power))
    np.save(os.path.join(dir_test,"x.npy"), x)
    np.save(os.path.join(dir_test,"y.npy"), y)
    np.save(os.path.join(dir_test,"z.npy"), z)
    np.save(os.path.join(dir_test,"alpha.npy"), alpha)
    np.save(os.path.join(dir_test,"bank.npy"), bank)
    np.save(os.path.join(dir_test,"beta.npy"), beta)
    np.save(os.path.join(dir_test,"wind.npy"), eff_wind_speed)    
    np.save(os.path.join(dir_test,"applied_force.npy"), force_during_sim)         
        
    x = [i+1 for i in range(EPISODES)]
    
    plot_average_reward(x,score_history,pict_name) 
    
    #print
    
    #kw_data = os.path.join(dir_test,"kwh.png")            
    
    mean_performance = os.path.join(args.path,"mean_performance.txt")
    
    with open(mean_performance,'w') as f: 
                
        f.write("average reward = "+str(np.mean(np.asarray(score_history)))+"\n")
        f.write("average time = "+str(np.mean(np.asarray(time_history)))+"\n")
        f.write("average power =" +str(np.mean(np.asarray(KWh_per_second)))+"\n")

    
    
        
            
            
                
            
                

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", default="./results/const/")
    parser.add_argument("--wind", default="const") #const, lin or turbo
    parser.add_argument('--step',type=float,default=0.1)
    parser.add_argument('--critic_lr',type = float, default = 0.0001) 
    parser.add_argument('--actor_lr',type = float, default = 0.0001) 
    parser.add_argument("--episodes", type=int, default=6000)
    parser.add_argument('--range_actions',action='append',default = None) 
    parser.add_argument('--force',type=int,default=1200) 
    parser.add_argument('--range_force',type=int,default=None)
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--warmup",type=int,default=50000)
    parser.add_argument("--cpu",type=int,default=0)
    args = parser.parse_args()
    test(args)




