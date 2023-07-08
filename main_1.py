import numpy as np 
#from learning.models_radam import * 
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
import pickle


ACTION_NOISE = 0.25    #prima era 0.2
ATTACK_INF_LIM = -5 
ATTACK_SUP_LIM = 18 
BANK_INF_LIM = -3 
BANK_SUP_LIM=3
R_A = 10


def main(args): 


    np.random.seed(21) 
    
    range_actions = np.array([1,1]) 
    
    force = args.force 
    
    continue_ = bool(args.continue_)
    
    initialization = int(args.initialization) 
    
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
    
    mem_size = args.mem_size
    
    c_lr = args.critic_lr
    
    a_lr = args.actor_lr
    
    step = args.step
    
    integration_step = 0.001 
    
    duration = args.duration
    
    episode_duration= int(duration) 
    
    horizon = int(episode_duration/step) 
    
    integration_steps_per_learning_step=int(step/integration_step)
    
    dir_name = os.path.join(args.path,"data")
    
    dir_nets = os.path.join(args.path,"nets")
    
    dir_training_status  = os.path.join(args.path,"training_status")
    
    data_name= os.path.join(dir_name,"data_training.txt") 
               
    pict_name = os.path.join(dir_name,"average_reward.png") 
    
    if not os.path.exists(dir_name):
    
        os.makedirs(dir_name)
        
    if not os.path.exists(dir_training_status): 
        
        os.makedirs(dir_training_status) 
        
    replay_file = os.path.join(dir_training_status, "replay_memory.pickle")
    
    training_info = os.path.join(dir_training_status, "training_info.pickle")    
    
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
    
    agent =Agent(4,2,critic_lr=c_lr,actor_lr=a_lr,gamma=0.99,warmup=warmup_,max_size=mem_size,chkpt_dir=dir_nets)#,device="cpu") 
    
    if(args.continue_): 
        print("ciao")
        agent.load_training_model(folder = dir_training_status) 
        agent.warmup=0
        with open(replay_file,'rb') as f: 
        
            replay_memory : ReplayBuffer = pickle.load(f) 
            
        agent.memory = replay_memory 
        
        with open(training_info, 'rb') as f: 
        
            t_data : training_state = pickle.load(f)
            
            agent.trajectory_critic_1["parameters"] = t_data.trajectory_critic_1["parameters"] 
            agent.trajectory_critic_1["gradients"] = t_data.trajectory_critic_1["gradients"]
            agent.trajectory_critic_2["parameters"] = t_data.trajectory_critic_1["parameters"]
            agent.trajectory_critic_2["gradients"] = t_data.trajectory_critic_2["gradients"]
            agent.trajectory_actor["parameters"] = t_data.trajectory_actor["parameters"]
            agent.trajectory_actor["gradients"] = t_data.trajectory_actor["gradients"]
            
        init_episode = t_data.n_training +1 
        
    else: 
    
        init_episode = 0 
        
        t_data = training_state()
        
        if (initialization==1): 
        
            #agent.he_init()
            print("no initialization") 
        else: 
            agent.manual_initialization() 
            
    counter_hist= []    
    true_score_hist=[]
    best_count=8000
    for i in range(init_episode, EPISODES): 
    
        print(i, end='\r') 
        
        done = False 
        
        score = 0 
        
        k.reset(initial_position, initial_velocity, args.wind) 
        
        initial_beta = k.beta() 
        
        state = (np.random.uniform(ATTACK_INF_LIM,ATTACK_SUP_LIM), np.random.uniform(BANK_INF_LIM,BANK_SUP_LIM), initial_beta,k.position.r)    
        
        c_l = f_cl(state[0])
        
        c_d = f_cd(state[0])
        
        k.update_coefficients_cont(c_l,c_d,state[1])
                    
        state = np.asarray(state)
        
        count = 0
        
        kWh = 0
        
        true_score = 0
        
        while not done: 
        
            reward = 0 
            
            t_data.counter +=1 
            
            count += 1 
            
            action = agent.choose_action(state,ACTION_NOISE)
            
            new_attack, new_bank = state[0]+action[0]*range_actions[0], state[1]+action[1]*range_actions[1]
            
            new_attack = np.clip(new_attack,ATTACK_INF_LIM,ATTACK_SUP_LIM) 
            
            new_bank = np.clip(new_bank,BANK_INF_LIM,BANK_SUP_LIM)
            
            c_l = f_cl(new_attack)
            
            c_d = f_cd(new_attack)
            
            k.update_coefficients_cont(c_l,c_d,new_bank)
            
            sim_status = k.evolve_system_2(integration_steps_per_learning_step,integration_step,force)
            
            energy = k.reward(step,force)
            
            kWh +=energy
            
            if not sim_status == 0: 
            
                reward = 25 -k.position.r 
                
                true_score = reward 
                
                done = True 
                
                new_state = state 
            
            else: 
            
                new_state = np.asarray((new_attack, new_bank, k.beta(),k.position.r))
                
                reward = -k.v_kite()/100
                
            if (reward == 0 and k.position.r==100): #considerare di metterlo in tutte le posizioni 
            
                reward = -0.1 
                
            if (count == int(horizon) -1 or k.fullyrolled()): 
            
                done = 1 
                
                reward = 25 -k.position.r 
                
                true_score = reward 
                
                if (k.fully_rolled()): 
                
                    reward = 2 
                    
                    true_score = reward 
                    
                    print("finished") 
                    
            if done: 
           
                reward += kWh 
                
                true_score += kWh 
               
            agent.store_transition(state, action, reward, new_state, done)
           
            agent.train(combined=True, clip=args.clip) 
           
            score += reward 
           
            state = new_state 
           
       
        counter_hist.append(count) 
       
        true_score_hist.append(true_score) 
       
        t_data.final_position.append(k.position.r) 
       
        t_data.kwh_history.append(kWh)
            
        t_data.score_history.append(score) 
        
        avg_score = np.mean(true_score_hist[-R_A:])
        avg_count = np.mean(np.asarray(counter_hist[-R_A:]))
        
        if t_data.counter > agent.warmup +R_A: 
        
        
            if avg_score > t_data.best_score: 
            
                t_data.best_score = avg_score 
                
                agent.save_models()
                
                print("saving model at episode ", i)
                
                
            if avg_count < best_count: 
                #name__ = "fast_model"+str(hh)
                name__="fast_model"
                fast_model=os.path.join(args.path,name__)
                best_count=avg_count
                if not os.path.exists(fast_model): 
        
                    os.makedirs(fast_model) 
                if (avg_score-t_data.best_score>-0.0001):
                    print("saving fast model at episode: ",i)
                    agent.save_training_model(folder=fast_model)
                   # hh+=1
                
            
                    
        if (i%50==0 or i==EPISODES-1): 
            t_data.trajectory_critic_1["parameters"] = agent.trajectory_critic_1["parameters"]
            t_data.trajectory_critic_1["gradients"]=agent.trajectory_critic_1["gradients"]
            t_data.trajectory_critic_2["parameters"] = agent.trajectory_critic_2["parameters"]
            t_data.trajectory_critic_2["gradients"]=agent.trajectory_critic_2["gradients"]
            t_data.trajectory_actor["parameters"] = agent.trajectory_critic_2["parameters"]
            t_data.trajectory_actor["gradients"]=agent.trajectory_critic_2["gradients"]
            agent.save_training_model(folder=dir_training_status) 
            
            t_data.n_training = i
            
            with open(replay_file, 'wb') as f: 
        
                pickle.dump(agent.memory,f)
                
                
            with open(training_info, 'wb') as f:
        
                pickle.dump(t_data,f)
                
                

 
    x = [i+1 for i in range(EPISODES)]
    
    plot_average_reward(x,t_data.score_history,pict_name) 
    
    with open(data_name,'w') as f: 
               
        for i in range(0,len(t_data.score_history)): 
                   
            f.write(str(t_data.score_history[i])+"\n") 
            
    data_name= os.path.join(dir_name,"kWh.txt") 
    with open(data_name,'w') as f: 
               
        for i in range(0,len(t_data.score_history)):
        
            f.write(str(t_data.kwh_history[i])+"\n")  
            
    pict_name = os.path.join(dir_name,"traj_actor.png")         
    plot_trajectory(t_data.trajectory_actor,pict_name)#,ylim=(0,5))
    pict_name = os.path.join(dir_name,"traj_critic_1.png")    
    plot_trajectory(t_data.trajectory_critic_1,pict_name)
    pict_name = os.path.join(dir_name,"traj_critic_2.png")    
    plot_trajectory(t_data.trajectory_critic_2,pict_name)
                   
               
            
                            
            
                       
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", default="./results/const/")
    parser.add_argument("--wind", default="const") #const, lin or turbo
    parser.add_argument('--step',type=float,default=0.1)
    parser.add_argument('--critic_lr',type = float, default = 0.0001) 
    parser.add_argument('--actor_lr',type = float, default = 0.0001) 
    parser.add_argument("--episodes", type=int, default=6000)
   # parser.add_argument('--range_actions',action='append',default = None) 
    parser.add_argument('--force',type=int,default=1200) 
  #  parser.add_argument('--range_force',type=int,default=None)
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--warmup",type=int,default=50000)
    parser.add_argument("--mem_size",type=int, default=1000000)
    parser.add_argument("--initialization",type=int,default=0)
    parser.add_argument("--continue_",type=bool, default=False)
    parser.add_argument("--clip",type=int, default=0)
    parser.add_argument("--step_penalty",type=float,default=0)
    args = parser.parse_args()
    main(args)            
    
    
        
    
    
    
    
    
    

    
