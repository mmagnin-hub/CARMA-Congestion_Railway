import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import math


def plot_policy_convergence(error_vec, n_day, n_groups):
    # Keep only the rows actually used
    error_used = error_vec[n_day:, :]  

    # Reverse vertically so the first iteration is at the left
    error_rev = error_used[::-1, :]

    plt.figure(figsize=(10, 5))
    for g in range(n_groups):
        plt.plot(error_rev[:, g], label=f"Group t*={g}")

    plt.title("Policy Convergence Error (Lower is Better)")
    plt.xlabel("Day")
    plt.ylabel("Policy L2 Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_final_policies(groups, n_groups):
    plt.figure(figsize=(15, 10))
    for g in range(n_groups):
        plt.subplot(2, 3, g + 1)
        plt.imshow(groups[g].pi + 1e-12, aspect="auto", norm=LogNorm())
        plt.title(f"Policy for Group {groups[g].traveler_type}")
        plt.xlabel("Action index (t × b)")
        plt.ylabel("State index (u × k)")
        plt.colorbar(label="Policy Probability (log scale)")
    plt.tight_layout()
    plt.show()

# def plot_final_policies_linear(groups, n_groups):
#     plt.figure(figsize=(15, 10))
#     for g in range(n_groups):
#         plt.subplot(2, 3, g + 1)
#         plt.imshow(groups[g].pi, aspect="auto", cmap='inferno') # cmap to enhance visibility
#         plt.title(f"Policy for Group {groups[g].traveler_type}")
#         plt.xlabel("Action index (t × b)")
#         plt.ylabel("State index (u × k)")
#         plt.colorbar(label="Policy Probability")
#     plt.tight_layout()
#     plt.show()

def plot_final_policies_linear(groups, n_groups):
    plt.figure(figsize=(15, 10))
    
    for g in range(n_groups):
        plt.subplot(2, 3, g + 1)
        
        pi = groups[g].pi
        
        plt.imshow(
            pi,
            aspect="auto",
            cmap="viridis",        # brighter colormap
            vmin=0,                # ensures full brightness range
            vmax=np.max(pi)        # avoids washed-out scaling
        )
        
        plt.title(f"Policy for Group {groups[g].traveler_type}")
        plt.xlabel("Action index (t × b)")
        plt.ylabel("State index (u × k)")
        plt.colorbar(label="Policy Probability")
    
    plt.tight_layout()
    plt.show()


def plot_specific_state_policy(groups, n_groups, K, specific_u, specific_k):
    state_index = specific_u * (K + 1) + specific_k

    gap = 2  # visual gap between blocks

    plt.figure(figsize=(10, 5))

    for g in range(n_groups):
        plt.subplot(1, 2, g + 1)

        prob_compressed = []
        action_pos = []
        action_labels = []

        non_zero_idx = np.where(groups[g].pi[state_index, :] > 0)[0]

        current_x = 0
        prev_idx = None

        for idx in non_zero_idx:
            if prev_idx is not None and idx != prev_idx + 1:
                current_x += gap  # compress zero region

            prob_compressed.append(groups[g].pi[state_index, idx])
            action_pos.append(current_x)
            action_labels.append(idx)   # ORIGINAL action index

            current_x += 1
            prev_idx = idx

        plt.bar(action_pos, prob_compressed)
        plt.yscale('log')
        plt.ylim(1e-6, 1)

        # restore original meaning of x-axis
        plt.xticks(action_pos, action_labels, rotation=90)

        plt.title(
            f"Policy for Group {groups[g].traveler_type} "
            f"at State (u={specific_u}, k={specific_k})"
        )
        plt.xlabel("Action index (t × b)")
        plt.ylabel("Action Probability (log scale)")

    plt.tight_layout()
    plt.show()

def plot_specific_state_policy_linear(groups, n_groups, K, specific_u, specific_k):
    state_index = specific_u * (K + 1) + specific_k

    gap = 2  # visual gap between blocks

    plt.figure(figsize=(10, 5))

    for g in range(n_groups):
        plt.subplot(1, 2, g + 1)

        prob_compressed = []
        action_pos = []
        action_labels = []

        non_zero_idx = np.where(groups[g].pi[state_index, :] > 0)[0]

        current_x = 0
        prev_idx = None

        for idx in non_zero_idx:
            if prev_idx is not None and idx != prev_idx + 1:
                current_x += gap  # compress zero region

            prob_compressed.append(groups[g].pi[state_index, idx])
            action_pos.append(current_x)
            action_labels.append(idx)   # ORIGINAL action index

            current_x += 1
            prev_idx = idx

        plt.bar(action_pos, prob_compressed)

        # fixed linear probability scale for all groups
        plt.ylim(0, 1)

        # restore original meaning of x-axis
        plt.xticks(action_pos, action_labels, rotation=90)

        plt.title(
            f"Policy for Group {groups[g].traveler_type} "
            f"at State (u={specific_u}, k={specific_k})"
        )
        plt.xlabel("Action index (t × b)")
        plt.ylabel("Action Probability")

    plt.tight_layout()
    plt.show()


#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# NEW PLOTS

import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import math 
from matplotlib.colors import LogNorm

def plot_policy(group, u=None, k=None, t=None, b=None, b_star=None):
    """
    Plot policy heatmap with flexible slicing + optimal b* markers.

    Parameters
    ----------
    group : TravelerGroup
    u, k : int or list or None
    t, b : int or list or None
    b_star : list or array of size T (optimal b for each t)
    """

    U, K, T = group.U, group.K, group.T

    def to_list(x, max_val):
        if x is None:
            return list(range(max_val))
        if isinstance(x, int):
            return [x]
        return list(x)

    u_list = to_list(u, U)
    k_list = to_list(k, K+1)
    t_list = to_list(t, T)
    b_list = to_list(b, K+1)

    state_indices = [ui*(K+1)+ki for ui in u_list for ki in k_list]
    action_indices = [ti*(K+1)+bi for ti in t_list for bi in b_list]

    pi = group.pi[np.ix_(state_indices, action_indices)]

    # ---- labels ----
    state_labels = []
    prev_ui = None
    for ui in u_list:
        for ki in k_list:
            state_labels.append(f"u={ui}" if ui != prev_ui else "")
            prev_ui = ui

    action_labels = []
    prev_ti = None
    for ti in t_list:
        for bi in b_list:
            action_labels.append(f"t={ti}" if ti != prev_ti else "")
            prev_ti = ti

    # ---- plot ----
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(pi, cmap="viridis",
                     xticklabels=action_labels,
                     yticklabels=state_labels)
    
    # ---- Y axis (u,k) ----
    yticks = list(range(len(state_indices)))
    y_major_positions = []
    y_minor_labels = []

    for i, ui in enumerate(u_list):
        for j, ki in enumerate(k_list):
            pos = i * len(k_list) + j
            y_minor_labels.append(str(ki))  # k labels (numbers only)
            if j == 0:
                y_major_positions.append(pos)

    # major = u
    ax.set_yticks(y_major_positions)
    ax.set_yticklabels([f"u={ui}" for ui in u_list], rotation=0)

    # minor = k
    ax.set_yticks(yticks, minor=True)
    ax.set_yticklabels(y_minor_labels, minor=True)


    # ---- X axis (t,b) ----
    xticks = list(range(len(action_indices)))
    x_major_positions = []
    x_minor_labels = []

    for i, ti in enumerate(t_list):
        for j, bi in enumerate(b_list):
            pos = i * len(b_list) + j
            x_minor_labels.append(str(bi))  # b labels (numbers only)
            if j == 0:
                x_major_positions.append(pos)

    # major = t
    ax.set_xticks(x_major_positions)
    ax.set_xticklabels([f"t={ti}" for ti in t_list], rotation=0)

    # minor = b
    ax.set_xticks(xticks, minor=True)
    ax.set_xticklabels(x_minor_labels, minor=True)

    # spacing between major and minor labels
    ax.tick_params(axis='x', which='major', pad=10)
    ax.tick_params(axis='x', which='minor', pad=2)

    ax.tick_params(axis='y', which='major', pad=10)
    ax.tick_params(axis='y', which='minor', pad=2)

    # size difference
    ax.tick_params(axis='both', which='major', length=8, width=2, labelsize=10)
    ax.tick_params(axis='both', which='minor', length=3, width=0.8, labelsize=8)

    # ---- add vertical lines for b_star ----
    if b_star is not None:
        for i, ti in enumerate(t_list):
            if ti >= len(b_star):
                continue
            b_opt = b_star[ti]

            if b_opt in b_list:
                # position inside the block
                bi_index = b_list.index(b_opt)
                x_pos = i * len(b_list) + bi_index + 0.5
                ax.axvline(x=x_pos, color="red", linewidth=2)

    plt.title("Policy Heatmap π(u,k → t,b)")
    plt.xlabel("Actions (t,b)")
    plt.ylabel("States (u,k)")
    plt.tight_layout()
    plt.show()

def plot_transition_matrix(group, u=None, k=None, t=None, b=None,
                           b_star=None, max_plots=6):

    U, K, T = group.U, group.K, group.T

    def to_list(x, max_val):
        if x is None:
            return list(range(max_val))
        if isinstance(x, int):
            return [x]
        return list(x)

    u_list = to_list(u, U)
    k_list = to_list(k, K+1)
    t_list = to_list(t, T)
    b_list = to_list(b, K+1)

    combos = [(ui, ki, ti, bi)
              for ui in u_list
              for ki in k_list
              for ti in t_list
              for bi in b_list]

    if len(combos) > max_plots:
        raise ValueError(
            f"Too many combinations ({len(combos)}), max is {max_plots}"
        )

    n = len(combos)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    axes = np.array(axes).ravel()

    for ax, (ui, ki, ti, bi) in zip(axes, combos):

        idx = (
            ui * ((K+1)*T*(K+1)) +
            ki * (T*(K+1)) +
            ti * (K+1) +
            bi
        )

        row = group.p[idx]
        matrix = row.reshape(U, K+1)

        # ---- crop k' axis ----
        k_min = max(0, ki - 5)
        k_max = min(K, ki + 5)

        outside_left = matrix[:, :k_min]
        outside_right = matrix[:, k_max+1:]

        if (np.any(outside_left > 1e-12) or np.any(outside_right > 1e-12)):
            raise ValueError(
                f"Non-zero transition probability outside k±5 window "
                f"for (u={ui},k={ki},t={ti},b={bi})"
            )

        matrix_cropped = matrix[:, k_min:k_max+1]

        # ---- plot ----
        xticks = list(range(k_min, k_max + 1))
        step = max(1, len(xticks)//6)

        sns.heatmap(
            matrix_cropped,
            cmap="rocket",
            ax=ax,
            vmin=0, vmax=1,
            xticklabels=xticks[::step],
            yticklabels=list(range(U))
        )

        # ---- title with b_star ----
        if b_star is not None and ti < len(b_star):
            ax.set_title(f"(u={ui},k={ki},t={ti},b={bi}) | b*={b_star[ti]}")
        else:
            ax.set_title(f"(u={ui},k={ki},t={ti},b={bi})")

        #ax.set_xlabel("k'")
        # ax.set_ylabel("u'")

    plt.tight_layout()
    plt.show()