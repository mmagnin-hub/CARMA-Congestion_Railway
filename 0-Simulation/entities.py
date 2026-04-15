import numpy as np
from abc import ABC, abstractmethod
import random

# ==============================================================
# Traveler
# ==============================================================

# ==============================================================
# Base class for shared attributes and methods
# ==============================================================
class TravelerBase(ABC):
    """Abstract base for traveler entities."""
    def __init__(self, traveler_type: int):
        self.traveler_type = traveler_type

# ==============================================================
# TravelerGroup (shared type-level parameters and learning)
# ==============================================================
class TravelerGroup(TravelerBase):
    def __init__(self, type_id: int, phi: np.ndarray,delta_t: int, t_star: int, u_value: np.ndarray, K: int, T: int, delta: float=0.9, eta: float=0.1, alpha: float=42, beta: float=42, gamma: float=42):
        '''
        TODO: describe all the variables
        '''
        # Fixed attributes
        super().__init__(type_id)
        self.phi = np.array(phi)
        self.delta_t = delta_t
        self.t_star = t_star
        self.u_value = np.array(u_value)
        self.delta = delta
        self.eta = eta  
        self.U = self.u_value.shape[0]
        self.K = K
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Dynamic attributes
        self.travelers: list[Traveler] = []
        # Shared policy structures (group-level learning)
        self.zeta = np.zeros((self.U * (self.K+1) * self.T * (self.K+1))) 
        self.V = np.zeros((self.U * (self.K+1)))
        self.Q = np.zeros((self.U * (self.K+1) * self.T * (self.K+1)))
        self.P = np.zeros((self.U * (self.K+1) * self.T * (self.K+1), self.U * (self.K+1)))
        self.pi = np.random.rand(self.U * (self.K+1), self.T * (self.K+1))
        self.action_mask = np.zeros_like(self.pi) # action_mask: 1 if action is allowed, 0 otherwise
        for i in range(self.U * (self.K+1)): 
            k = i % (self.K+1)
            for t in range(self.T):
                for b in range(self.K + 1):
                    action_index = t*(self.K+1) + b
                    if b <= k:  
                        self.action_mask[i, action_index] = 1
        self.pi *= self.action_mask
        self.pi = self.pi / self.pi.sum(axis=1, keepdims=True)

    def register(self, traveler: 'Traveler'):
        self.travelers.append(traveler)

    def update_transition_matrix(self):
        '''
        Update the transition matrix P based on simulated transitions.
        '''
        self.P.fill(0.0)  # KZ: reinit transition
        for traveler in self.travelers:
            idx_state_init = traveler.u_start * ((self.K+1) * self.T * (self.K+1)) + traveler.k_start * (self.T * (self.K+1)) + traveler.t * (self.K+1) + traveler.b
            idx_state_final = traveler.u_curr *(self.K+1) + traveler.k_curr
            self.P[idx_state_init,idx_state_final] += 1

        # update transition matrix
        for row in range(self.P.shape[0]):
            row_sum = self.P[row, :].sum()
            if row_sum > 0:
                self.P[row, :] /= row_sum
        return
    
    def update_policy(self, system: 'System'):
        # Computing the expected total reward from state (u,k) taking action (t,b), zeta
        self.immediate_reward(system)
        self.Q = self.zeta + self.delta * np.dot(self.P, self.V)

        # Update policy with smoothing
        new_pi = self.perturbed_best_response_dynamic()
        self.pi = (1 - self.eta) * self.pi + self.eta * new_pi

        # Update value function
        Q_reshape = self.Q.reshape(self.pi.shape) 
        self.V = np.sum(Q_reshape * self.pi, axis=1)
        return
    
    def immediate_reward(self, system: 'System'):
        '''
        TODO: optimize and adapt comments
        '''

        # Precompute absolute time distance
        dt = np.abs(np.arange(self.T) - self.t_star)* self.delta_t       # shape (T,)
        early_mask = np.arange(self.T) <= self.t_star                    # shape (T,)

        # Expand for broadcasting
        dt_exp = dt[None, None, :, None]                   # shape (1,1,T,1)
        psi    = system.psi[None, None, :, None]

        # u values
        u_val = self.u_value[:, None, None, None]          # shape (U,1,1,1)

        # Time-dependent betas/gammas
        beta_t = np.where(early_mask, self.beta, self.gamma)[None,None,:,None]

        # Compute the base fast-lane and slow-lane costs
        time_reward = -u_val * dt_exp * beta_t               # shape (U,K1,T,K1)

        # Create b and b_star arrays for comparison
        b_vals = np.arange(self.K + 1)[None, None, None, :]        # shape (1,1,1,K1)
        b_vals = np.broadcast_to(b_vals, (self.U, self.K+1, self.T, self.K+1))  # shape (U,K1,T,K1)

        b_star = system.b_star[None, None, :, None]        # shape (1,1,T,1)
        b_star = np.broadcast_to(b_star, (self.U, self.K+1, self.T, self.K+1))  # shape (U,K1,T,K1)

        # Masks for the three block conditions
        mask_first_class  = b_vals > b_star
        mask_equal   = b_vals == b_star

        # expend psi with b dimmension
        psi_exp = np.where(mask_equal, psi, np.where(mask_first_class, 1, 0))  # shape (U,K1,T,K1)

        # expend capacity following the same logic as psi
        capacity = np.where(mask_equal, system.first_class_capacity * psi + system.second_class_capacity * (1 - psi), np.where(mask_first_class, system.first_class_capacity, system.second_class_capacity))  # shape (U,K1,T,K1)

        # count the number of travelers in each departure time slot
        number_of_travelers = np.zeros((1,1,self.T,1))
        for t in range(self.T):
            number_of_travelers[:,:,t,:] = sum(1 for traveler in system.travelers if traveler.t == t) 
            # important to use system.travelers and not self.travelers to get the total number of travelers in each departure time slot, not just the ones in the current group

        crowdedness_reward= -self.alpha * psi_exp * np.power(number_of_travelers / capacity, 4)

        # Allocate output
        zeta = np.zeros((self.U, self.K+1, self.T, self.K+1))
        zeta += time_reward
        zeta += crowdedness_reward

        self.zeta = zeta.reshape(-1)

        return 

    def perturbed_best_response_dynamic(self):
        Q_mtx = self.Q.reshape(self.pi.shape)
        pi = np.zeros(self.pi.shape)

        for i, Q_row in enumerate(Q_mtx):
            idx_col = self.action_mask[i] > 0
            Q_row = Q_row[idx_col]  
            expQ = np.exp(Q_row)
            pi[i, idx_col] = expQ / np.sum(expQ)
        return pi
    
    def __repr__(self):
        return f"TravelerGroup(type={self.traveler_type}, n={len(self.travelers)})"
    
# ==============================================================
# Inidividual Traveler (agent-level state and actions)
# ==============================================================
class Traveler(TravelerBase):
    def __init__(self, 
                 group: TravelerGroup,
                 k_init: int, 
                 id: int
                 ):
        '''
        TODO: describe all the variables
        '''
        super().__init__(group.traveler_type)
        # Group level
        self.group = group
        group.register(self)

        # Fixed attributess
        self.id = id
        
        # Dynamic attributes
        self.u_curr = 0
        self.k_curr = k_init 
        self.u_start = 0
        self.k_start = 0
        self.t: int = self.group.t_star
        self.b = 0
        self.enter_first_class: bool = False

    def action(self):
        idx_row = self.u_curr * (self.group.K+1) + self.k_curr
        prob = self.group.pi[idx_row]
        idx_col = np.random.choice(self.group.T * (self.group.K+1), p=prob)
        self.t = idx_col // (self.group.K+1)
        self.b = idx_col % (self.group.K+1)
        return 

    def paid_karma_bid(self):
        """Deduct the bid from the traveler's karma balance."""
        if self.enter_first_class:
            self.k_curr -= self.b
        return
    
    def get_new_karma(self, karma):
        """
        Receive redistributed karma from the system.
        (See System.karma_redistribution())
        """
        self.k_curr += karma
        return
    
    def update_urgency(self):
        """Update urgency level based on transition matrix phi."""
        self.u_curr = np.random.choice(self.group.U, p=self.group.phi[self.u_curr])
        return   

    def store_start_state(self):
        """ Store start of the day to compare at the end of the day. """
        self.u_start = self.u_curr
        self.k_start = self.k_curr  
        return


# ==============================================================
# System
# ==============================================================

class System: 
    def __init__(self, first_class_capacity: int, second_class_capacity: int, K: int, T: int, travelers: list[Traveler]):
        '''
        TODO: describe all the variables
        '''
        # Fixed attributes
        self.first_class_capacity = first_class_capacity
        self.second_class_capacity = second_class_capacity
        self.K = K  
        self.T = T
        self.N = len(travelers)

        # Dynamic attributes
        self.b_star = np.zeros(self.T)                
        self.psi = np.zeros(self.T)        
        self.travelers = travelers

    def karma_redistribution(self):
        """
        Uniform redistribution of used karma among all travelers.
        TODO: 
        - Later : redistribute based on traveler states.
        """
        total_karma_used = sum(traveler.b for traveler in self.travelers if traveler.enter_first_class)
        karma_per_traveler = total_karma_used // len(self.travelers)
        leftover_karma = total_karma_used % len(self.travelers) 
        indexes_with_extra = set(random.sample(range(len(self.travelers)), leftover_karma))
        for i, traveler in enumerate(self.travelers):
            extra = 1 if i in indexes_with_extra else 0
            traveler.get_new_karma(karma_per_traveler + extra)
        return

    def simulate_class_attribution(self):
        """
        Loop over each departure time group and compute:
        - who enters the 1st vs 2nd.
        """
        # Reset dynamic attributes
        self.b_star = np.zeros(self.T)              
        self.psi = np.zeros(self.T) 
        for traveler in self.travelers: traveler.enter_first_class = False
        # sort travelers for each departure time
        group_travelers = self.group_travelers_by_departure() 
        for t, travelers in enumerate(group_travelers): 
            self.b_star[t], self.psi[t] = self.determine_threshold_bid(travelers)
            self.assign_lanes(t, travelers)
            count_second_class_users = sum(1 for traveler in travelers if not traveler.enter_first_class)
        return 
    
    def group_travelers_by_departure(self):
        """
        Create a list of travelers for each departure time slot.
        """
        group_travelers = [[] for _ in range(self.T)]
        for traveler in self.travelers:
            group_travelers[traveler.t].append(traveler)
        return group_travelers

    def determine_threshold_bid(self, travelers: list[Traveler]):
        """
        Compute threshold bid (b_star) for the first class.
        """
        bids = sorted([traveler.b for traveler in travelers], reverse=True)
        if len(bids) > self.first_class_capacity:
            b_star = bids[self.first_class_capacity - 1]
            traveler_count_at_b_star = sum(1 for traveler in travelers if traveler.b == b_star)
            traveler_count_over_b_star = sum(1 for traveler in travelers if traveler.b > b_star)
            free_spot_at_b_star = self.first_class_capacity - traveler_count_over_b_star
            psi = free_spot_at_b_star/traveler_count_at_b_star 
        elif len(bids) == 0:
            b_star = 0
            psi = 1
        else:
            b_star = bids[-1]
            psi = 1
        return b_star, psi
    
    def assign_lanes(self, t, travelers):
        '''
        Update each traveler's lane choice (enter_first_class) based on b_star and psi.
        '''
        for traveler in travelers:
            if traveler.b > self.b_star[t]:
                traveler.enter_first_class = True
            elif traveler.b == self.b_star[t]:
                traveler.enter_first_class = random.random() < self.psi[t] # MM: add a warning print
        return




