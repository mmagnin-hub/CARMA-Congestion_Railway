import numpy as np
from abc import ABC, abstractmethod
import random
import scipy.sparse as sp
import numba as nb

@nb.jit(nopython=True)
def _update_p(p, u_start, k_start, t_start, b_start, u_curr, k_curr, U, K, T, state_action_mask):
    p.fill(0.0)
    for i in range(len(u_start)):
        idx = u_start[i] * ((K+1) * T * (K+1)) + k_start[i] * (T * (K+1)) + t_start[i] * (K+1) + b_start[i]
        j = u_curr[i] * (K+1) + k_curr[i]
        p[idx, j] += 1
    for row in range(p.shape[0]):
        s = 0.0
        for col in range(p.shape[1]):
            s += p[row, col]
        if s > 0:
            for col in range(p.shape[1]):
                p[row, col] /= s
                
        # elif s == 0: # MM uniformly distribute if no transitions observed
        #     for col in range(p.shape[1]):
        #         p[row, col] = 1/(p.shape[1])
    p[state_action_mask == 0, :] = 0.0 

@nb.jit(nopython=True)
def _compute_zeta(U, K, T, u_value, delta_t, t_star, beta, gamma, alpha, psi, b_star, number_of_travelers, first_class_capacity, second_class_capacity):
    zeta = np.zeros((U, K+1, T, K+1), dtype=np.float32)

    dt = np.abs(np.arange(T) - t_star) * delta_t

    early_mask = np.arange(T) <= t_star

    beta_t = np.where(early_mask, beta, gamma)

    u_val = u_value[:, None, None, None]

    dt_exp = dt[None, None, :, None]

    psi_exp = psi[None, None, :, None]

    b_star_exp = b_star[None, None, :, None]

    b_vals = np.arange(K + 1)[None, None, None, :]

    b_vals = np.broadcast_to(b_vals, (U, K+1, T, K+1))

    b_star_exp = np.broadcast_to(b_star_exp, (U, K+1, T, K+1))

    mask_first_class = b_vals > b_star_exp

    mask_equal = b_vals == b_star_exp

    psi_exp = np.where(mask_equal, psi_exp, np.where(mask_first_class, 1.0, 0.0))

    capacity = np.where(mask_equal, first_class_capacity * psi_exp + second_class_capacity * (1 - psi_exp), np.where(mask_first_class, first_class_capacity, second_class_capacity))

    number_of_travelers_exp = number_of_travelers[None, None, :, None]

    crowdedness_penalty = alpha * psi_exp * np.power(number_of_travelers_exp / capacity, 4)

    time_penalty = u_val * dt_exp * beta_t[None, None, :, None]

    zeta = - time_penalty - crowdedness_penalty

    return zeta.reshape(-1)

@nb.jit(nopython=True)
def _perturbed(Q_reshaped, action_mask):
    pi = np.zeros_like(Q_reshaped, dtype=np.float32)

    for i in range(Q_reshaped.shape[0]):
        idx_col = np.where(action_mask[i] > 0)[0]
        Q_row = Q_reshaped[i, idx_col]
        expQ = np.exp(Q_row)
        denom = np.sum(expQ)
        if denom == 0 or not np.isfinite(denom):
            pi[i, idx_col] = 1.0 / len(idx_col)
        else:
            pi[i, idx_col] = expQ / denom
    return pi

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
        self.zeta = np.zeros((self.U * (self.K+1) * self.T * (self.K+1)), dtype=np.float32) 
        self.V = np.zeros((self.U * (self.K+1)), dtype=np.float32)
        self.Q = np.zeros((self.U * (self.K+1) * self.T * (self.K+1)), dtype=np.float32)
        self.p = np.zeros((self.U * (self.K+1) * self.T * (self.K+1), self.U * (self.K+1)),dtype=np.float32) 
        self.pi = np.random.rand(self.U * (self.K+1), self.T * (self.K+1)).astype(np.float32)
        self.action_mask = np.zeros_like(self.pi, dtype=np.float32) # action_mask: 1 if action is allowed, 0 otherwise
        self.state_action_mask = np.zeros((self.U * (self.K+1) * self.T * (self.K+1)), dtype=np.float32) # state_action_mask: 1 if state-action pair is allowed, 0 otherwise
        self.state_distribution = np.zeros(self.U * (self.K+1), dtype=np.float32)
        for i in range(self.U * (self.K+1)): 
            k = i % (self.K+1)
            for t in range(self.T):
                for b in range(self.K + 1):
                    action_index = t*(self.K+1) + b
                    if b <= k:  
                        self.action_mask[i, action_index] = 1
                        self.state_action_mask[i * (self.T * (self.K+1)) + action_index] = 1
        self.pi *= self.action_mask
        self.pi = self.pi / self.pi.sum(axis=1, keepdims=True)

    def register(self, traveler: 'Traveler'):
        self.travelers.append(traveler)
    
    def update_group_attributes(self, system: 'System', n_day: int):
        self.update_transition_matrix()
        self.update_immediate_reward(system)
        self.update_Q()
        self.update_policy(n_day)
        self.update_V()
        return

    def update_transition_matrix(self):
        '''
        Update the transition matrix P based on simulated transitions.
        '''
        u_start = np.array([tr.u_start for tr in self.travelers], dtype=np.int32)
        k_start = np.array([tr.k_start for tr in self.travelers], dtype=np.int32)
        t_start = np.array([tr.t for tr in self.travelers], dtype=np.int32)
        b_start = np.array([tr.b for tr in self.travelers], dtype=np.int32)
        u_curr = np.array([tr.u_curr for tr in self.travelers], dtype=np.int32)
        k_curr = np.array([tr.k_curr for tr in self.travelers], dtype=np.int32)
        # debug
        for i in range(len(u_start)):
            if self.travelers[i].enter_first_class and k_start[i] - b_start[i] > k_curr[i]:
                print("Warning: observed transition where k_start - b_start > k_curr for traveler entering first class. This should not happen if travelers are correctly paying their karma bids.")
            elif not self.travelers[i].enter_first_class and k_start[i] > k_curr[i]:
                print("Warning: observed transition where k_start > k_curr for traveler not entering first class. This should not happen if travelers are correctly paying their karma bids.")

        _update_p(self.p, u_start, k_start, t_start, b_start, u_curr, k_curr, self.U, self.K, self.T, self.state_action_mask)
        return
    
    def update_immediate_reward(self, system: 'System'):
        number_of_travelers = np.array([sum(1 for traveler in system.travelers if traveler.t == t) for t in range(self.T)], dtype=np.float32)
        self.compute_immediate_reward(system, number_of_travelers)
        return
    
    def update_Q(self):
        self.Q = self.zeta + self.delta * np.dot(self.p, self.V) # MM sum product over the last axis of P and V
        return
    
    def update_policy(self, n_day: int):
        pi_tilde = self.perturbed_best_response_dynamic()
        self.pi = self.pi + 1/(n_day + 1) * (pi_tilde - self.pi)
        return
    
    def update_V(self):
        Q_reshape = self.Q.reshape(self.pi.shape) 
        self.V = np.sum(Q_reshape * self.pi, axis=1)
        return
    
    def compute_immediate_reward(self, system: 'System', number_of_travelers: np.ndarray):
        '''
        TODO: optimize and adapt comments
        '''
        self.zeta = _compute_zeta(self.U, self.K, self.T, self.u_value, self.delta_t, self.t_star, self.beta, self.gamma, self.alpha, system.psi, system.b_star, number_of_travelers, system.first_class_capacity, system.second_class_capacity)
        return 

    def perturbed_best_response_dynamic(self):
        Q_mtx = self.Q.reshape(self.pi.shape)
        return _perturbed(Q_mtx, self.action_mask)
    
    def update_state_distribution(self):
        self.state_distribution.fill(0.0)
        for traveler in self.travelers:
            idx_state = traveler.u_curr * (self.K+1) + traveler.k_curr
            self.state_distribution[idx_state] += 1
        self.state_distribution = self.state_distribution / len(self.travelers)
        return

    def compute_expected_value_function(self):
        self.expected_value_function = np.dot(self.state_distribution, self.V)
        return
    
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
        if self.k_curr > self.group.K:
            print(f"Warning: Traveler {self.id} with {self.k_curr } karma exceeded max K. Capping to {self.group.K}.")
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
            psi = free_spot_at_b_star/(traveler_count_at_b_star)
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




