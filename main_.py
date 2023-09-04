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
import pickle

ATTACK_INF_LIM = -5 
ATTACK_SUP_LIM = 18 
BANK_INF_LIM = -3 
BANK_SUP_LIM=3
R_A = 10

def scheduling(value, t, T, exp=0.1):
    if t>T:
        return value/((t-T)**exp)
    else:
        return value


def main(args): 

    np.random.seed(17)
    
    if(args.range_actions) is not None: 
    
        if(len(args.range_actions[0])>1): 
        
            range_actions = np.array([args.range_actions[0][0],args.range_actions[0][2]])
            range_actions = range_actions.astype(np.float64)
            
        else: 
        
            range_actions=np.array([args.range_actions[0][0],args.range_actions[0][0]])
            range_actions = range_actions.astype(np.float64)
            
    else: 
        
        range_actions = np.array([1,1]) 
        
    ACTION_NOISE = args.noise 
    
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
    
    type_ = args.type_
    
    step = args.step
    
    learning_step= args.step 
    
    integration_step = 0.001 
    
    duration = args.duration
    
    episode_duration= int(duration) 
    
    horizon = int(episode_duration/learning_step) 
    
    integration_steps_per_learning_step=int(learning_step/integration_step)
    
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
    
    agent =Agent(3,2,critic_lr=c_lr,actor_lr=a_lr,gamma=args.gamma,warmup=warmup_,max_size=mem_size,chkpt_dir=dir_nets)
    
    #alpha = (0.00001**(1/10000))/(0.1**(1/10000))
    
    penalty = 10
    if(args.continue_):
        
        agent.load_training_model(folder=dir_training_status) 
        agent.warmup=0
        with open(replay_file, 'rb') as f: 
        
            replay_memory = pickle.load(f)
            
        agent.memory = replay_memory 
        
        with open(training_info, 'rb') as f:
        
            t_data = pickle.load(f)
            
            agent.trajectory_critic_1["parameters"] = t_data.trajectory_critic_1["parameters"] 
            agent.trajectory_critic_1["gradients"] = t_data.trajectory_critic_1["gradients"]
            agent.trajectory_critic_2["parameters"] = t_data.trajectory_critic_1["parameters"]
            agent.trajectory_critic_2["gradients"] = t_data.trajectory_critic_2["gradients"]
            agent.trajectory_actor["parameters"] = t_data.trajectory_actor["parameters"]
            agent.trajectory_actor["gradients"] = t_data.trajectory_actor["gradients"]
            
        #EPISODES = EPISODES -t_data.n_training 
        init_episode = t_data.n_training +1 
        
        #penalty_1 = t_data.penalty_1 
        
        #penalty_2 = t_data.penalty_2 
        
    
    else: 
        
    
        init_episode = 0
        
        #penalty_1 = -0.1 
        
        #penalty_2 = -0.01 
        
        t_data = training_state_2()
        
        if (initialization==1):
            agent.he_init()
        else:
        
            agent.manual_initialization()
            
 
            
    trained_="positions"
    
    if args.wind=="turbo":
        
        final_phi=np.load(os.path.join(trained_,"final_phi.npy"))
        final_theta=np.load(os.path.join(trained_,"final_theta.npy"))
        
    elif args.wind=="lin": 
        
        final_phi=np.load(os.path.join(trained_,"final_phi_lin.npy"))
        final_theta=np.load(os.path.join(trained_,"final_theta_lin.npy"))
    else:
    
        final_phi=np.load(os.path.join(trained_,"final_phi_const.npy"))
        final_theta=np.load(os.path.join(trained_,"final_theta_const.npy"))
        
    true_score_hist = []
    
    
    
    for i in range(init_episode, EPISODES): 
    
        done = False 
        
        num=np.random.randint(1000)
        
        theta_init=final_theta[num]
        phi_init=final_phi[num]
 
        initial_position = pk.vect(theta_init, phi_init, 100)

        k.reset(initial_position, initial_velocity,args.wind) 
        
        initial_beta = k.beta()      
        
        state = (np.random.uniform(ATTACK_INF_LIM,ATTACK_SUP_LIM), np.random.uniform(BANK_INF_LIM,BANK_SUP_LIM), initial_beta)
        
        c_l = f_cl(state[0])
        
        c_d = f_cd(state[0])
        
        k.update_coefficients_cont(c_l,c_d,state[1])
                    
        state = np.asarray(state)
        
        count = 0
        
        #force = 1200
        
        kWh = 0
        
        true_score = 0
        
        count = 0
        
        score= 0
        
        hh= 1
        
        while not done: 
        
            reward = 0 
            
            t_data.counter +=1 
            
            count += 1 
            
            action = agent.choose_action(state, ACTION_NOISE)
            
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
                
                if type_==0:
                    reward = 25 -k.position.r 
                else: 
                    reward= scheduling(-penalty,horizon,horizon-count,exp=0.2)
                
                true_score = reward 
                
                done = True 
                
                new_state = state 
                
            else: 
            
                new_state = np.asarray((new_attack, new_bank, k.beta()))
                 
                reward= - k.v_kite()/1000    #radial velocity
                 
            if(reward == 0 and k.position.r==100): 
            
                reward= -0.001*(hh)
                if hh<1000:
                    hh*=2
                
            else: 
                if (reward==0): 
                
                    reward=-0.0001*(hh)
                    if hh<1000:
                        hh*=2
                else: 
                    hh=1
                    
            if (count==int(horizon) -1 or k.fullyrolled()):
            
                done =1 
                
                if type_==0:
                    reward = 25 -k.position.r 
                else: 
                    reward= scheduling(-penalty,horizon,horizon-count,exp=0.2)
              
                true_score = reward
                
                if(k.fullyrolled()): 
                
                    reward= 2  #energy +1/count
                    reward += (0.1+kWh)*10
                    true_score=reward
                    print("finished")
                    
                    
            agent.store_transition(state,action,reward,new_state,done)
            
            agent.train(combined=True,clip=args.clip) 
            
            score+=((args.gamma)**count)*reward 
            
            state = new_state
            
            
        true_score_hist.append(true_score)  
        
        t_data.true_score_history.append(true_score) 
                    
        t_data.final_position.append(k.position.r)      
         
        t_data.kwh_history.append(kWh)
            
        t_data.score_history.append(score)         
        
        avg_score = np.mean(true_score_hist[-R_A:])  
        
        
        
        if t_data.counter > agent.warmup +R_A: 
            
            #if (t_data.counter > 2*agent.warmup and i< 10000): 
             #   penalty_1 *= alpha 
              #  penalty_2 *= alpha 
        
            if avg_score > t_data.best_score: 
            
                t_data.best_score = avg_score 
                
                agent.save_models()
                
                print("saving model at episode ", i)
                
                
        if(i%20==0):
            print("episode = ",i,"position= ",k.position.r, "energy=",kWh,"score = ",score)        
                
                
        if (i%50==0 or i==EPISODES-1): 
            t_data.trajectory_critic_1["parameters"] = agent.trajectory_critic_1["parameters"]
            t_data.trajectory_critic_1["gradients"]=agent.trajectory_critic_1["gradients"]
            t_data.trajectory_critic_2["parameters"] = agent.trajectory_critic_2["parameters"]
            t_data.trajectory_critic_2["gradients"]=agent.trajectory_critic_2["gradients"]
            t_data.trajectory_actor["parameters"] = agent.trajectory_critic_2["parameters"]
            t_data.trajectory_actor["gradients"]=agent.trajectory_critic_2["gradients"]
            agent.save_training_model(folder=dir_training_status) 
            
            t_data.n_training = i
            
            #t_data.penalty_1 = penalty_1 
            
            #t_data.penalty_2 = penalty_2
            
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
    parser.add_argument("--episodes", type=int, default=12000)
    parser.add_argument('--range_actions',action='append',default = None) 
    parser.add_argument('--force',type=int,default=1200) 
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--noise",type=float,default=0.25)
    parser.add_argument("--warmup",type=int,default=50000)
    parser.add_argument("--mem_size",type=int, default=1000000)
    parser.add_argument("--initialization",type=int,default=0)
    parser.add_argument("--continue_",type=bool, default=False)
    parser.add_argument("--clip",type=int, default=0)
    parser.add_argument("--type_",type=int, default=0)
    #parser.add_argument("--step_penalty",type=float,default=0)
    parser.add_argument("--gamma",type=float,default=0.99)
    args = parser.parse_args()
    main(args)
