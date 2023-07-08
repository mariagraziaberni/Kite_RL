import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from learning.utils import get_params_and_gradients_norm
import os
import numpy as np
from ewc import *
torch.autograd.set_detect_anomaly(True)
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(3,256, bias=True)
        self.layer2=nn.Linear(256,32, bias=True)
        self.layer3=nn.Linear(32,9, bias=False)
        self.activation=nn.ReLU()

    def forward(self, x):
        x=self.activation(self.layer1(x))
        x=self.activation(self.layer2(x))
        return self.layer3(x)


MAX_ACTION =1.0

class ReplayBuffer(): 

    def __init__(self, max_size, input_shape, n_actions): 
        
        self.mem_size = max_size 
        
        self.mem_cntr = 0 
        
        self.state_memory = np.zeros((self.mem_size, input_shape)) 
        
        self.new_state_memory = np.zeros((self.mem_size, input_shape)) 
        
        self.action_memory = np.zeros((self.mem_size, n_actions)) 
        
        self.reward_memory = np.zeros(self.mem_size) 
        
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.bool)
        
    def change_max_mem(self,num): 
    
        self.mem_size=num 
        
    def store_transition(self, state, action, reward, state_, done): 
        
        index = self.mem_cntr%self.mem_size 
        
        self.state_memory[index] = state
        
        self.new_state_memory[index] = state_ 
        
        self.terminal_memory[index] = done 
        
        self.reward_memory[index] = reward 
        
        self.action_memory[index] = action 
        
        self.mem_cntr +=1 
        
    def sample_buffer(self, batch_size,combined=False): 
    
        offset = 1 if combined else 0 
        
       # print("offset = ",offset)
        max_mem = min(self.mem_cntr, self.mem_size-offset)
        
        batch = np.random.choice(max_mem, batch_size -offset) 
        
        states = self.state_memory[batch] 
        
        states_ = self.new_state_memory[batch] 
        
        actions = self.action_memory[batch] 
        
        rewards = self.reward_memory[batch] 
        
        dones = self.terminal_memory[batch] 
        
        if combined: 
            
            index = self.mem_cntr%self.mem_size -1 
            
            last_state = self.state_memory[index] 
            
            last_state_ =  self.new_state_memory[index] 
            
            last_actions = self.action_memory[index] 
            
            last_rewards = self.reward_memory[index] 
            
            last_dones = self.terminal_memory[index] 
            
            states = np.vstack((states, last_state)) 
            
            states_ = np.vstack((states_,last_state_))
            
            actions = np.vstack((actions,last_actions))
            
            rewards = np.append(rewards, last_rewards) 
            
            dones = np.append(dones,last_dones) 
            
            
            
        
        
        return states, actions, rewards, states_, dones 
        

class training_state():

    def __init__(self): 
    
        self.score_history = [] 
        
        self.kwh_history = [] 
        
        self.counter = 0 
        
        self.best_score = -8000
        
        self.n_training = 0 
        
        self.final_position = [] 
        
        self.trajectory_actor = {"parameters": [], "gradients": []}
        
        self.trajectory_critic_1 = {"parameters": [], "gradients": []}
        
        self.trajectory_critic_2 = {"parameters": [], "gradients": []}
        
        
  
  
class CriticNetwork(nn.Module): 
        
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions, name= None, chkpt_dir = "net_dir"): 
        
        super(CriticNetwork, self).__init__ ()
        
        self.name = name
        
        if name is not None: 
            
            if not os.path.exists(chkpt_dir): 
                
                os.makedirs(chkpt_dir) 
                
            self.checkpoint_file= os.path.join(chkpt_dir,name +'_td3') 
            
        
        self.input_dims = input_dims 
        
        self.fc1_dims = fc1_dims 
        
        self.fc2_dims = fc2_dims 
        
        self.n_actions = n_actions 
        
        self.name = name 
        
        self.fc1 = nn.Linear(self.input_dims+n_actions, self.fc1_dims) 
        
        self.fc2 = nn.Linear(self.fc1_dims+n_actions, self.fc2_dims) 
        
        self.q1 = nn.Linear(self.fc2_dims,1) 
        
        
        
      
        
    def forward(self, state, action): 
    
        q1_action_value = self.fc1(torch.cat([state,action],dim=1))
        
        q1_action_value = F.relu(q1_action_value) 
        
        q1_action_value = self.fc2(torch.cat([q1_action_value,action],dim=1))
        
        q1_action_value = F.relu(q1_action_value) 
        
        q1 = self.q1(q1_action_value) 
        
        return q1 
        
    def init_weights(self): 
        #torch.manual_seed(2)
        
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        
        f3 = 0.003
        
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)

        nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        
        nn.init.uniform_(self.q1.weight.data, -f3, f3)
        
        nn.init.uniform_(self.q1.bias.data, -f3, f3)
        
    def init_He(self): 
        
        f1 = np.sqrt(2/(self.fc1.weight.data.size()[0]))
        
        f2 = np.sqrt(2/(self.fc2.weight.data.size()[0]))
        
        nn.init.normal_(self.fc1.weight.data,mean=0.0, std=f1)
        
        nn.init.constant_(self.fc1.bias.data,0) 
        
        nn.init.normal_(self.fc2.weight.data, mean=0.0, std=f2)
        
        nn.init.constant_(self.fc2.bias.data, 0) 
        
        f3 = 1 / np.sqrt(self.q1.weight.data.size()[0])
        
        nn.init.uniform_(self.q1.weight.data, -f3, f3)
        
        #nn.init.uniform_(self.mu.bias.data, -f3, f3)
        nn.init.constant_(self.q1.bias.data, 0) 
        
        
        
    def save_checkpoint(self): 
        
        if self.name is not None:
    
            print("...saving...") 
        
            torch.save(self.state_dict(),self.checkpoint_file)
        
        
    def load_checkpoint(self): 
    
        if self.name is not None:
    
            print("..loading...") 
        
            self.load_state_dict(torch.load(self.checkpoint_file)) 
            
            
    def save_training_checkpoint(self, folder = None): 
    
        if folder is not None: 
        
            checkpoint_file= os.path.join(folder,self.name +'_td3')
        
            torch.save(self.state_dict(),checkpoint_file)    
            
    def load_training_checkpoint(self, folder = None): 
    
        if folder is not None: 
        
            checkpoint_file= os.path.join(folder,self.name +'_td3')
            
            self.load_state_dict(torch.load(checkpoint_file)) 
            
   
            
            
            
        
                 
        
            
        
        
        
    
class ActorNetwork(nn.Module): 
        
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions, name=None, chkpt_dir = "net_dir", max_action=None): 
        
        super(ActorNetwork, self).__init__ ()
        
        self.name = name
        
        self.max_action = max_action
        
        if name is not None: 
            
            if not os.path.exists(chkpt_dir): 
                
                os.makedirs(chkpt_dir) 
                
            self.checkpoint_file= os.path.join(chkpt_dir,name +'_td3') 
        
        self.input_dims = input_dims 
        
        self.fc1_dims = fc1_dims 
        
        self.fc2_dims = fc2_dims 
        
        self.n_actions = n_actions 
        
        self.name = name 
        
        #self.checkpoint_dir = chkpt_dir 
        
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims) 
        
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims) 
        
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)  
        
        
        
    def forward(self, state): 
    
        prob = self.fc1(state)
        
        prob= F.relu(prob) 
        
        
        prob= self.fc2(prob)
   
        prob = F.relu(prob) 
      
        if self.max_action is None:
        
            mu = torch.tanh(self.mu(prob))*MAX_ACTION
            
        else: 
        
            mu = torch.tanh(self.mu(prob))*self.max_action
        
        return mu
        
    def init_weights(self): 
        #torch.manual_seed(0)
        
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        
        #f_new = 1 / np.sqrt(self.new_layer.weight.data.size()[0])
        
        f3 = 0.003
        
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)

        nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        
        #nn.init.uniform_(self.new_layer.weight.data, -f_new, f_new)

        #nn.init.uniform_(self.new_layer.bias.data, -f_new, f_new)
        
        nn.init.uniform_(self.mu.weight.data, -f3, f3)
        
        nn.init.uniform_(self.mu.bias.data, -f3, f3)
        
    def init_He(self): 
        
        f1 = np.sqrt(2/(self.fc1.weight.data.size()[0]))
        
        f2 = np.sqrt(2/(self.fc2.weight.data.size()[0]))
        
        nn.init.normal_(self.fc1.weight.data,mean=0.0, std=f1)
        
        nn.init.constant_(self.fc1.bias.data,0) 
        
        nn.init.normal_(self.fc2.weight.data, mean=0.0, std=f2)
        
        nn.init.constant_(self.fc2.bias.data, 0) 
        
        f3 = 1 / np.sqrt(self.mu.weight.data.size()[0])
        
        nn.init.uniform_(self.mu.weight.data, -f3, f3)
        
        #nn.init.uniform_(self.mu.bias.data, -f3, f3)
        nn.init.constant_(self.mu.bias.data, 0) 
        
        
        
        
        
        
        
        
    def save_checkpoint(self): 
    
        if self.name is not None:
    
            print("...saving...") 
        
            torch.save(self.state_dict(),self.checkpoint_file)
        
        
    def load_checkpoint(self): 
    
        if self.name is not None:
    
            print("...loading...") 
        
            self.load_state_dict(torch.load(self.checkpoint_file)) 
            
            
    def save_training_checkpoint(self, folder = None): 
    
        if folder is not None: 
        
            checkpoint_file= os.path.join(folder,self.name +'_td3')
        
            torch.save(self.state_dict(),checkpoint_file)    
            
    def load_training_checkpoint(self, folder = None): 
    
        if folder is not None: 
        
            checkpoint_file= os.path.join(folder,self.name +'_td3')
            
            self.load_state_dict(torch.load(checkpoint_file)) 
            
            
         
         
class Agent: 
    
    def __init__(self, input_dims, n_actions, critic_lr=0.0005, actor_lr = 0.0005, tau = 0.005, gamma = 0.99, update_actor_interval =2, warmup = 50000, 
                 max_size=1000000, layer1_size= 400, layer2_size = 300, batch_size=100,target_noise = 0.2, chkpt_dir= "const_training_standards",device = None, max_action = None):
                 
        #torch.manual_seed(0)        
        
        self.input_dims = input_dims 
        
        self.n_actions = n_actions 
        
        self.gamma = gamma 
        
        self.tau = tau 
        
        self.memory = ReplayBuffer(max_size, self.input_dims, self.n_actions) 
        
        self.batch_size = batch_size 
        
        self.learn_step_cntr = 0 
        
        self.time_step = 0 
        
        self.warmup = warmup 
        
        if device is not None: 
        
            self.device = device
           
        else:
            
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
            
        print("device from model=",device)
            
        self.max_action = MAX_ACTION 
        
        self.min_action = -MAX_ACTION
        
        self.update_actor_iter = update_actor_interval
        
        self.actor = ActorNetwork(self.input_dims, layer1_size, layer2_size, self.n_actions, name = "actor",chkpt_dir=chkpt_dir,max_action = self.max_action).to(self.device)
        
        self.critic_1 = CriticNetwork(self.input_dims, layer1_size, layer2_size, self.n_actions, name = "critic_1",chkpt_dir=chkpt_dir).to(self.device) 
        
        self.critic_2 = CriticNetwork(self.input_dims, layer1_size, layer2_size, self.n_actions, name = "critic_2",chkpt_dir=chkpt_dir).to(self.device)
    
        self.target_actor = ActorNetwork(self.input_dims, layer1_size, layer2_size, self.n_actions, name = "target_actor",chkpt_dir=chkpt_dir, max_action = self.max_action).to(self.device)
    
        self.target_critic_1 = CriticNetwork(self.input_dims, layer1_size, layer2_size, self.n_actions, name = "target_critic_1",chkpt_dir=chkpt_dir).to(self.device) 
    
        self.target_critic_2 = CriticNetwork(self.input_dims, layer1_size, layer2_size, self.n_actions , name = "target_critic_2",chkpt_dir=chkpt_dir).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr = actor_lr) 
        
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(),lr = critic_lr) 
        
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(),lr = critic_lr) 
        
    
        self.target_noise = target_noise 
        
        self.update_network_parameters(tau=1) 
        
        self.min_value = 0
        
        self.max_value = 0
        
        self.trajectory_actor = {"parameters": [], "gradients": []}
        
        self.trajectory_critic_1 = {"parameters": [], "gradients": []}
        
        self.trajectory_critic_2 = {"parameters": [], "gradients": []}
        
        self.c1_precision = None 
        self.c1_mean = None 
        self.c2_precision =None 
        self.c2_mean = None 
        self.a_precision=None 
        self.a_mean=None
        
        #self.trajectory_target_actor = {"parameters": [], "gradients": []}
        
        #self.trajectory_target_critic_1 = {"parameters": [], "gradients": []}
        
        #self.trajectory_target_critic_2 = {"parameters": [], "gradients": []}
        
    def change_memory(self,num):
    
        self.memory.change_max_mem(num)
        
    def manual_initialization(self): 
    
        self.actor.init_weights()
        
        self.critic_1.init_weights()
        
        self.critic_2.init_weights()

        self.update_network_parameters(tau=1) 
        
    def he_init(self):
    
        self.actor.init_He()
        
        self.critic_1.init_He()
        
        self.critic_2.init_He()

        self.update_network_parameters(tau=1) 
      
        
    
        
    def compute_ewc_matrices(self):
    
        c1,c2,a,mc1,mc2,ma= test_ewc() 
        self.c1_precision = c1
        self.c1_mean = mc1
        self.c2_precision =c2
        self.c2_mean = mc2
        self.a_precision=a
        self.a_mean=ma 
    
        
         
        
    def critic_1_penalty(self,model: nn.Module):
        loss=0 
        for n, p in model.named_parameters():
            _loss = (self.c1_precision[n]).to(self.device) * (p - (self.c1_mean[n]).to(self.device)) ** 2
            loss += _loss.sum()
        return loss.to(self.device)   

    def critic_2_penalty(self,model: nn.Module):
        loss=0 
        for n, p in model.named_parameters():
            #_loss = self.c2_precision[n] * (p - self.c2_mean[n]) ** 2
            _loss = (self.c2_precision[n]).to(self.device) * (p - (self.c2_mean[n]).to(self.device)) ** 2
            loss += _loss.sum()
        return loss.to(self.device) 
    
    def actor_penalty(self,model: nn.Module):
        loss=0 
        for n, p in model.named_parameters():
           # _loss = self.a_precision[n] * (p - self.a_mean[n]) ** 2
            _loss = (self.a_precision[n]).to(self.device) * (p - (self.a_mean[n]).to(self.device)) ** 2
            loss += _loss.sum() 
        return loss.to(self.device)  
  
    def choose_action(self, observation, expl_noise, test=False): 
    
    
        if (self.time_step < self.warmup and (not test)):
        
            mu = torch.tensor(np.random.normal(loc = 0, scale = 0.5, size = (self.n_actions,)))
            
        else: 
        
            self.actor.eval()
            
            state = torch.tensor(observation, dtype = torch.float).to(self.device)
            
            with torch.no_grad(): 
            
                mu = self.actor.forward(state).to(self.device) 
                
            if (not test): 
            
                
                
                
             
                    
                mu = mu + torch.tensor(np.random.normal(0, self.max_action*expl_noise,size=self.n_actions), dtype = torch.float).to(self.device) 
                
                
        action = torch.clamp(mu, self.min_action, self.max_action) 
        
        self.time_step +=1 
        
        return action.cpu().detach().numpy() 
        
        
    def store_transition(self, state, action, reward, new_state, done): 
    
        self.memory.store_transition(state, action, reward, new_state, done) 
        
        
    def train(self,combined=False,clip=False,importance=0.8): 
        
        if self.memory.mem_cntr < self.batch_size: 
            
            return 
        
        
        if(self.memory.mem_cntr == self.batch_size): 
            print("training begin \n")
            print(" ")
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size,combined=combined) 
        
        reward = torch.tensor(reward, dtype= torch.float).to(self.device) 
        
        done = torch.tensor(done).to(self.device) 
        
        state = torch.tensor(state, dtype= torch.float).to(self.device) 
        
        action = torch.tensor(action, dtype= torch.float).to(self.device) 
        
        state_ = torch.tensor(new_state, dtype= torch.float).to(self.device) 
        
        
        target_actions = self.target_actor.forward(state_) 
        
        #noise = torch.clamp(torch.randn_like(action)*self.noise*self.max_action,0.2*self.min_action,0.2*self.max_action)   
        
        noise = torch.clamp(torch.randn_like(action)*self.target_noise*self.max_action,self.target_noise*self.min_action,self.target_noise*self.max_action) 

        target_actions_ = target_actions + noise

        target_actions = torch.clamp(target_actions_, self.min_action, self.max_action)
        self.critic_1_optimizer.zero_grad() 
        
        self.critic_2_optimizer.zero_grad() 

        Q_tc1 = self.target_critic_1.forward(state_,target_actions) 
        
        Q_tc2 = self.target_critic_2.forward(state_,target_actions) 
        
        Q1 = self.critic_1.forward(state,action) 
        
        Q2 = self.critic_2.forward(state,action) 
        
        Q_tc1[done] = 0.0 
        
        Q_tc2[done] = 0.0 
        
        Q_tc1= Q_tc1.view(-1) 
        
        Q_tc2 = Q_tc2.view(-1) 
        
        critic_target_value = torch.min(Q_tc1,Q_tc2) 
        
        target = reward +self.gamma*critic_target_value
        
        target = target.view(self.batch_size,1) 
        
       
        
        q1_loss = F.mse_loss(Q1,target) 
       
        q2_loss = F.mse_loss(Q2,target) 
       
       
        critic_loss= q1_loss+ q2_loss  #+(importance)*self.critic_1_penalty(self.critic_1)  #+(importance)*self.critic_2_penalty(self.critic_2)
        
        #critic_1_loss = critic_loss +a_   #(importance)*self.critic_1_penalty(self.critic_1)  
        #critic_1_loss = critic_loss+importance*self.critic_1_penalty(self.critic_1)
        
        critic_loss.backward()   #retain_graph=True) 
        #if clip: 
        #    nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=clip, norm_type=2)
          #  nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=clip, norm_type=2)
        self.critic_1_optimizer.step() 
        
        self.critic_2_optimizer.step()  # zero_grad() 
       # critic_loss_2 = critic_loss_+(importance)*self.critic_2_penalty(self.critic_2)
        #critic_1_loss = critic_loss+importance*self.critic_1_penalty(self.critic_1)
        #critic_loss_2 = critic_1_loss -a_ +(importance)*self.critic_2_penalty(self.critic_2)
        #critic_loss_2.backward() 
        
       # self.critic_2_optimizer.step() 
        
        self.learn_step_cntr +=1 
        
        params_norm, grad_norms = get_params_and_gradients_norm(self.critic_1.named_parameters())
        self.trajectory_critic_1["parameters"].append(params_norm) 
        self.trajectory_critic_1["gradients"].append(grad_norms)
        
        params_norm, grad_norms = get_params_and_gradients_norm(self.critic_2.named_parameters())
        self.trajectory_critic_2["parameters"].append(params_norm) 
        self.trajectory_critic_2["gradients"].append(grad_norms)
        
        
        #update actor 
        
        if self.learn_step_cntr % self.update_actor_iter != 0: 
         
            return 
            
        self.actor_optimizer.zero_grad() 
        
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))

        actor_loss = -torch.mean(actor_q1_loss) +(importance)*self.actor_penalty(self.actor)
        
        actor_loss.backward() 
        if clip:
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=clip, norm_type=2)
        self.actor_optimizer.step() 
        
        params_norm, grad_norms = get_params_and_gradients_norm(self.actor.named_parameters())
        self.trajectory_actor["parameters"].append(params_norm) 
        self.trajectory_actor["gradients"].append(grad_norms)
        
        self.update_network_parameters() 


    def update_network_parameters(self, tau = None): 
    
        if tau is None: 
         
            tau = self.tau
        
        actor_params = self.actor.named_parameters() 
        
        critic_1_params = self.critic_1.named_parameters()
        
        critic_2_params = self.critic_2.named_parameters()
        
        
        target_actor_params = self.target_actor.named_parameters()
        
        target_critic_1_params = self.target_critic_1.named_parameters()
        
        target_critic_2_params = self.target_critic_2.named_parameters()
        

        critic_1 = dict(critic_1_params)
        
        critic_2 = dict(critic_2_params)
        
        actor = dict(actor_params)
        
       # target_critic_dict = dict(target_critic_params)
        target_actor = dict(target_actor_params)
        
        target_critic_1 = dict(target_critic_1_params)
        
        target_critic_2 = dict(target_critic_2_params)
         
        
        
        for name in critic_1:
            critic_1[name] = tau*critic_1[name].clone()+ \
                                      (1-tau)*target_critic_1[name].clone()

        self.target_critic_1.load_state_dict(critic_1)
        
        for name in critic_2:
            critic_2[name] = tau*critic_2[name].clone()+ \
                                      (1-tau)*target_critic_2[name].clone()

        self.target_critic_2.load_state_dict(critic_2)
        
        
        for name in actor:
            actor[name] = tau*actor[name].clone()+ \
                                      (1-tau)*target_actor[name].clone()
                                      
        self.target_actor.load_state_dict(actor)
        
        
    def save_models(self): 
        
        self.actor.save_checkpoint()
        
        self.target_actor.save_checkpoint()
        
        self.critic_1.save_checkpoint()
        
        self.critic_2.save_checkpoint()
        
        self.target_critic_1.save_checkpoint()
        
        self.target_critic_2.save_checkpoint()
        
        
    def load_models(self): 
        
        self.actor.load_checkpoint()
        
        self.target_actor.load_checkpoint()
        
        self.critic_1.load_checkpoint()
        
        self.critic_2.load_checkpoint()
        
        self.target_critic_1.load_checkpoint()
        
        self.target_critic_2.load_checkpoint()
        
    def load_actor(self): 
    
        self.actor.load_checkpoint()
        
    def save_training_model(self, folder= None): 
    
        self.actor.save_training_checkpoint(folder = folder)
        
        self.target_actor.save_training_checkpoint(folder = folder)
        
        self.critic_1.save_training_checkpoint(folder = folder)
        
        self.critic_2.save_training_checkpoint(folder = folder)
        
        self.target_critic_1.save_training_checkpoint(folder = folder)
        
        self.target_critic_2.save_training_checkpoint(folder = folder)

    def load_training_model(self, folder = None): 
    
        self.actor.load_training_checkpoint(folder = folder)
        
        self.target_actor.load_training_checkpoint(folder = folder)
        
        self.critic_1.load_training_checkpoint(folder = folder)
        
        self.critic_2.load_training_checkpoint(folder = folder)
        
        self.target_critic_1.load_training_checkpoint(folder = folder)
        
        self.target_critic_2.load_training_checkpoint(folder = folder)

    
                       
        
