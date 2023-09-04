import numpy as np
import matplotlib.pyplot as plt 
import torch as T 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import os 
import learning.pykite as pk 
#from utility import *
from learning.utils import *
import pandas as pd 
import argparse
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt 
from classes_lstm import* 
from learning.models import training_state

import pickle
#from torch.utils.tensorboard import SummaryWriter 
#from ut import mean_and_std,get_params_and_gradients_norm
ACTION_NOISE = 0.2
ATTACK_INF_LIM = -5 
ATTACK_SUP_LIM = 18 
BANK_INF_LIM = -3 
BANK_SUP_LIM=3
R_A = 10

def main(args): 

    
   
    Cl_angles = read_file('env/coefficients/CL_angle.txt') 
    
    Cl_values = read_file('env/coefficients/CL_value.txt') 
    
    Cd_angles = read_file('env/coefficients/CD_angle.txt') 
    
    Cd_values = read_file('env/coefficients/CD_value.txt') 

    Cl_data = pd.DataFrame({'cl_angles':Cl_angles,'cl_values':Cl_values}) 
    
    Cd_data = pd.DataFrame({'cd_angles':Cd_angles,'cd_values':Cd_values}) 
    
    
    f_cl  = interpolate.interp1d(Cl_data.cl_angles, Cl_data.cl_values, kind='linear')
    
    f_cd  = interpolate.interp1d(Cd_data.cd_angles, Cd_data.cd_values, kind='linear')
    
    continue_ = bool(args.continue_)
    
    EPISODES = int(args.episodes) 
    
    warmup_ = args.warmup 
    
    mem_size = args.mem_size
    
    c_lr = args.critic_lr
    
    a_lr = args.actor_lr
    
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
    
    theta_init = 0.61823
    
    phi_init =   0.153866
    
    init_attack = 3.646512171128421
    
    init_bank = -2.5552693009376526
    
    c_l = f_cl(init_attack)
    
    c_d = f_cd(init_attack)
    
    init_psi = 0.0159
    
    initial_position = pk.vect(theta_init, phi_init, 100)
    
    range_actions = [1,1]   #1200] 
    
    range_actions = np.array(range_actions)
    
    k = pk.kite(initial_position, initial_velocity, args.wind,c_l,c_d,init_psi,continuous=True)
   
    
    
    
   
    
    
    hist_len = args.hist_len 
    
    train_interval = args.train_interval
    
    agent = Agent(3,2,hist_len,c_lr= c_lr, a_lr = a_lr,gamma=args.gamma,warmup=warmup_,chkpt_dir=dir_nets,device=None)
    
    if(args.continue_):
        print("continue =",args.continue_)
        agent.load_training_model(folder=dir_training_status) 
        agent.warmup=0
        with open(replay_file, 'rb') as f: 
        
            replay_memory = pickle.load(f)
            
        agent.memory = replay_memory 
        
        with open(training_info, 'rb') as f:
        
            t_data = pickle.load(f)
            
            #agent.trajectory_critic_1["parameters"] = t_data.trajectory_critic_1["parameters"] 
            #agent.trajectory_critic_1["gradients"] = t_data.trajectory_critic_1["gradients"]
            #agent.trajectory_critic_2["parameters"] = t_data.trajectory_critic_1["parameters"]
            #agent.trajectory_critic_2["gradients"] = t_data.trajectory_critic_2["gradients"]
            #agent.trajectory_actor["parameters"] = t_data.trajectory_actor["parameters"]
            #agent.trajectory_actor["gradients"] = t_data.trajectory_actor["gradients"]
        init_episode = t_data.n_training +1 
    else:
        init_episode=0
        t_data = training_state()
        
      #print("EPISODI RIMSTI",EPISODES) 
    print("replay numero dati ",agent.memory.mem_cntr) 
    print("dato dalla replay", agent.memory.state_memory[2]) 
    print("t _ data numero ",t_data.n_training) 
    print("t_data.counter",t_data.counter) 
    print("len of final positions and score ", len(t_data.final_position), len(t_data.score_history))
                
    true_score_hist = [] 
    counter_hist = [] 
    force = args.force
    
    #agent.load_models()
   # agent.warmup = 0 
    
 
    
    
    true_score_hist = [] 
     
    force = args.force
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



    


    
    for i in range(init_episode, EPISODES): 
        
       
        done = 0 
        
        score = 0 
        
        num=np.random.randint(1000)

        theta_init=final_theta[num]
        phi_init=final_phi[num]

        initial_position = pk.vect(theta_init, phi_init, 100)

        k.reset(initial_position,initial_velocity,args.wind) 
        
        initial_beta = k.beta()  #mi pare non ci sia piu bisogno di mettere cont=true
        
        #state = (np.random.uniform(-5,18),np.random.uniform(-3,3),initial_beta,k.position.theta,k.position.phi, k.position.r)
        state = (np.random.uniform(ATTACK_INF_LIM,ATTACK_SUP_LIM), np.random.uniform(BANK_INF_LIM,BANK_SUP_LIM), initial_beta)

        
        c_l = f_cl(state[0])
                   
        c_d = f_cd(state[0])
        
        k.update_coefficients_cont(c_l,c_d,state[1])
        
        state = np.asarray(state)
    
        o_buff = np.zeros([hist_len,3]) 
        a_buff = np.zeros([hist_len,2])
        o_buff[0,:] = state 
        o_buff_len = 0
    
        count = 0   
        
        kWh = 0

        true_score = 0

        
        while not done: 
        
            reward=0

            t_data.counter +=1

            
            count += 1 
            
            #if(count%700): 
               # print("state =", state)
            
            action = agent.choose_action(state, o_buff, a_buff, o_buff_len,ACTION_NOISE)
            
            #if(total_count%150==0): 
                #cond = agent.warmup<agent.time_step
                
                #print("fine warmup = ",cond)
                #print("a[0] = ",a[0]," a[1] = ",a[1]," a[2] = ",a[2]) 
            
            #action= np.zeros_like(a)
            
            #action[2] = a[2]*range_actions[2]/2 +range_actions[2]/2 
            #action[1]= a[1]*range_actions[1]
            #action[0] = a[0]*range_actions[0]
            
            new_attack, new_bank = state[0]+action[0]*range_actions[0],state[1]+action[1]*range_actions[1]
            
          
            
            new_attack = np.clip(new_attack,ATTACK_INF_LIM,ATTACK_SUP_LIM)

            new_bank = np.clip(new_bank,BANK_INF_LIM,BANK_SUP_LIM)

            
            c_l = f_cl(new_attack)
        
            c_d = f_cd(new_attack)
        
            k.update_coefficients_cont(c_l,c_d,new_bank)
    
            sim_status = k.evolve_system_2(integration_steps_per_learning_step,integration_step,force)
            
            if not sim_status == 0: 
            
                #reward = 100-(k.position.r) 
                
                reward = 25 -k.position.r

                true_score=reward
                done = 1 
                
                new_state = state 
                if(i%50==0):
                    print("Number Episode : ",i," Radial Position = ",k.position.r, "count = ", count)
                
            else: 
            
                new_state = np.asarray((new_attack, new_bank, k.beta()))
                
                reward= - k.v_kite()/1000
            
                #reward = 0 
                
                #aggiungere dopo if(k.position.r == 100 ) mettere penalita 
            if(reward == 0 and k.position.r==100): #era reward invece di energy

                reward= -0.01
            else:
                if (reward==0):    #questo lo ho aggiunto ora, prima non c'era
                    reward=-0.0001

                
            if (count == int(horizon)-1 or k.fullyrolled()): 
            
                done = 1 
                
                reward = 25-k.position.r #+energy +1/count
                
                true_score = reward

                
                if(k.fullyrolled()): 
                
                    reward= 2  #energy +1/count
                    true_score=reward

                
                    print("Number Episode = ",i," Finished!!") 
                    
            if done:
                reward += kWh
                true_score += kWh

                    
            agent.store_transition(state,action,reward,new_state,done) 
            
            if((t_data.counter%train_interval)==0): 
            
                agent.train(loc_steps = train_interval) 
                
            if o_buff_len== hist_len:
            
                o_buff[:hist_len - 1] = o_buff[1:]
                a_buff[:hist_len - 1] = a_buff[1:]
                o_buff[hist_len - 1] = list(state)
                a_buff[hist_len - 1] = list(action) 
                
            else: 
            
                o_buff[o_buff_len + 1 - 1] = list(state)
                a_buff[o_buff_len + 1 - 1] = list(action)
                o_buff_len += 1
                
                
            state = new_state 
            
            score += reward 
            
        #score_history.append(score) 
        true_score_hist.append(true_score)
       # avg_score = np.mean(score_history[-10:])
       
        t_data.final_position.append(k.position.r)

        t_data.kwh_history.append(kWh)

        t_data.score_history.append(score)

        avg_score = np.mean(true_score_hist[-R_A:])
        
        if t_data.counter > agent.warmup +R_A:


            if avg_score > t_data.best_score:

                t_data.best_score = avg_score

                agent.save_models()

                print("saving model at episode ", i)
                
                
        if (i%50==0 or i==EPISODES-1):
            #t_data.trajectory_critic_1["parameters"] = agent.trajectory_critic_1["parameters"]
            #t_data.trajectory_critic_1["gradients"]=agent.trajectory_critic_1["gradients"]
            #t_data.trajectory_critic_2["parameters"] = agent.trajectory_critic_2["parameters"]
            #t_data.trajectory_critic_2["gradients"]=agent.trajectory_critic_2["gradients"]
            #t_data.trajectory_actor["parameters"] = agent.trajectory_critic_2["parameters"]
            #t_data.trajectory_actor["gradients"]=agent.trajectory_critic_2["gradients"]
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


    #parser.argparse.ArgumentParser() 
    
    parser = argparse.ArgumentParser() 
    
    parser.add_argument("--path", default="./results/const/")
    
    parser.add_argument("--wind", default="const") #const, lin or turbo
    
    parser.add_argument('--step',type=float,default=0.1)
    
    parser.add_argument('--episodes',type=int,default=10000)
    
   
    
    parser.add_argument('--critic_lr',type = float, default = 0.0001) 
    
    parser.add_argument('--actor_lr',type = float, default = 0.0001) 
    
    parser.add_argument('--force',type=int,default=1200) 
    
    parser.add_argument('--hist_len',type=int,default=3) 
    
    #parser.add_argument('--save_dir',default =  "lstm_prova_4/") 
    
    parser.add_argument("--duration", type=int, default=300)
    
    parser.add_argument("--noise",type=float,default=0.25)
    parser.add_argument("--warmup",type=int,default=50000)
    parser.add_argument("--mem_size",type=int, default=1000000)
    
    parser.add_argument('--train_interval', type= int, default=30) #was 50
    
    parser.add_argument("--gamma",type=float,default=0.99)
    
    parser.add_argument("--continue_",type=bool, default=False)
    
    args = parser.parse_args() 
    
    main(args) 
    
    
    
    
     
     
