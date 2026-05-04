from entities import TravelerGroup, Traveler, System
import numpy as np
import pickle
import os

def main():
    # -------------------------------------------------------------
    # 1. Define model dimensions and parameters
    # -------------------------------------------------------------
    U = 2 # u in {0, 1} (urgency levels)             
    T = 10 # t in {0, 1, ..., 9} (time slots)         
    delta_t = 15          
    n_travelers = 9000
    K = 100 # k in {0, 1, ..., 100} (karma levels)
    k_init = 5 # 10 
    n_groups = 1
    t_star = 8
    phi = np.array([[0.8, 0.2],
                    [0.8, 0.2]])

    u_value = np.array([1.0, 6.0]) # from the paper

    # per hour penalty
    delta = 0.99 # discount factor  
    eta = 0.1 # smoothing weight MM will be remove 
    alpha = 0.05 # crowdedness penalty weight
    beta = 4/60 # early arrival weight (per minute)
    gamma = 16/60 # late arrival weight (per minute)

    # -------------------------------------------------------------
    # 2. Create traveler groups
    # -------------------------------------------------------------
    groups = []
    
    g1 = TravelerGroup(
        type_id=0,
        phi=phi,
        delta_t= delta_t,
        t_star=t_star,
        u_value=u_value,
        K=K,
        T=T,
        delta=delta,
        eta=eta,
        alpha=alpha,
        beta=beta,
        gamma=gamma
    )
    groups.append(g1)
    # -------------------------------------------------------------
    # 3. Create travelers and split across groups
    # -------------------------------------------------------------
    travelers = []
    per_group = n_travelers // n_groups

    traveler_id = 0
    for group in groups:
        for _ in range(per_group):
            traveler = Traveler(group=group, k_init=k_init, id=traveler_id)
            travelers.append(traveler)
            traveler_id += 1

    # -------------------------------------------------------------
    # 4. Initialize the System with all travelers
    # -------------------------------------------------------------
    first_class_capacity = 12 * delta_t # seat per class
    second_class_capacity = 48 * delta_t # 

    system = System(
        first_class_capacity=first_class_capacity,
        second_class_capacity=second_class_capacity,
        K=K,
        T=T,
        travelers=travelers
    )

    # -------------------------------------------------------------
    # 5. Simulation loop
    # -------------------------------------------------------------
    threshold = 1e-4
    total_day = 200
    n_day = total_day

    # For storing old policies: (states × actions × groups)
    pi_old = np.zeros((U*(K+1), T*(K+1), n_groups))
    error_vec = np.zeros((n_day, n_groups))
    expected_value_vec = np.zeros((n_day, n_groups))

    converge = False

    while not converge and n_day > 0:
        print("----- Remaining day", n_day, "-----")
        
        for g in groups:
            pi_old[:, :, g.traveler_type] = g.pi.copy()

        converge = True
        n_day -= 1

        # 1. Travelers act
        for tr in travelers:
            tr.store_start_state()
            tr.action()

        # 2. System queues
        system.simulate_class_attribution()

        total_karma = 0
        # 3. Payment
        for tr in travelers:
            total_karma += tr.k_curr # "after" redistribution
            tr.paid_karma_bid()
            

        assert not (np.isclose(total_karma, n_travelers * k_init) == False), f"Total karma {total_karma} does not match initial total karma {n_travelers * k_init}"

        # 4. Redistribution
        system.karma_redistribution()

        # 5. Update urgency
        for tr in travelers:
            tr.update_urgency()

        # 6. Update each group (independent policies)
        for g in groups:
            g.update_group_attributes(system, (total_day-n_day)) 

            g.update_state_distribution()
            g.compute_expected_value_function()

            # check that the transition matrix is valid and get the state location
            row_sums = g.p.sum(axis=1)

            assert not ((~np.isclose(row_sums, 1)) & (~np.isclose(row_sums, 0))).any(), \
                f"Transition matrix rows do not sum to 0 for unfeasible state-action pairs or 1 but to {row_sums}"


        # 7. Convergence
        print("b_star", system.b_star)
        for g in groups:
            expected_value_vec[n_day, g.traveler_type] = g.expected_value_function
            print(f"Group {g.traveler_type} expected value:", g.expected_value_function)
            err = np.linalg.norm(g.pi - pi_old[:, :, g.traveler_type])
            print(f"Group {g.traveler_type} error:", err)
            error_vec[n_day, g.traveler_type] = err
            if err > threshold:
                converge = False
            
    # -------------------------------------------------------------
    # 6. Download results
    # -------------------------------------------------------------
    path_name = "results_1/" 
    os.makedirs(path_name, exist_ok=True)
    with open(path_name + "groups.pkl", "wb") as f:
        pickle.dump(groups, f)

    with open(path_name + "error_vec.pkl", "wb") as f:
        pickle.dump(error_vec, f)

    with open(path_name + "expected_value_vec.pkl", "wb") as f:
        pickle.dump(expected_value_vec, f)

    with open(path_name + "simulation_params.pkl", "wb") as f:
        pickle.dump((n_day, n_groups, K, n_travelers), f)

    with open(path_name + "system.pkl", "wb") as f:
        pickle.dump(system, f)


if __name__ == "__main__":
    main()

