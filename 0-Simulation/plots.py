import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
 


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

def plot_policy(group, u=None, k=None, t=None, b=None):
    """
    Plot policy heatmap with flexible slicing.

    Parameters
    ----------
    group : TravelerGroup
    u, k : int or list or None
    t, b : int or list or None
    """

    U, K, T = group.U, group.K, group.T

    # ---- helper to normalize inputs ----
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

    # ---- build indices ----
    state_indices = [ui*(K+1)+ki for ui in u_list for ki in k_list]
    action_indices = [ti*(K+1)+bi for ti in t_list for bi in b_list]

    # ---- slice matrix ----
    pi = group.pi[np.ix_(state_indices, action_indices)]

    # ---- labels ----
    # Only show labels at transitions (when value changes)
    state_labels = []
    prev_ui = None
    for ui in u_list:
        for ki in k_list:
            if ui != prev_ui:
                state_labels.append(f"u={ui}")
            else:
                state_labels.append("")
            prev_ui = ui
    
    action_labels = []
    prev_ti = None
    for ti in t_list:
        for bi in b_list:
            if ti != prev_ti:
                action_labels.append(f"t={ti}")
            else:
                action_labels.append("")
            prev_ti = ti

    # ---- plot ----
    plt.figure(figsize=(10, 6))
    sns.heatmap(pi, cmap="viridis", xticklabels=action_labels, yticklabels=state_labels)
    plt.title("Policy Heatmap π(u,k → t,b)")
    plt.xlabel("Actions (t,b)")
    plt.ylabel("States (u,k)")
    plt.tight_layout()
    plt.show()


def plot_transition_matrix(group, u=None, k=None, t=None, b=None, max_plots=6):
    """
    Plot transition matrices for selected (u,k,t,b).

    If too many combinations are requested, raises an error.
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

    combos = [(ui, ki, ti, bi)
              for ui in u_list
              for ki in k_list
              for ti in t_list
              for bi in b_list]

    if len(combos) > max_plots:
        raise ValueError(
            f"Too many (u,k,t,b) combinations ({len(combos)}). "
            f"Reduce selection (max {max_plots})."
        )

    # ---- plotting ----
    fig, axes = plt.subplots(1, len(combos), figsize=(5*len(combos), 4))

    if len(combos) == 1:
        axes = [axes]

    for ax, (ui, ki, ti, bi) in zip(axes, combos):

        idx = (
            ui * ((K+1)*T*(K+1)) +
            ki * (T*(K+1)) +
            ti * (K+1) +
            bi
        )

        row = group.p[idx]  # shape (U*(K+1),)

        matrix = row.reshape(U, K+1)

        sns.heatmap(matrix, cmap="magma", ax=ax, vmin=0, vmax=1)

        ax.set_title(f"(u={ui},k={ki},t={ti},b={bi})")
        ax.set_xlabel("k'")
        ax.set_ylabel("u'")

    plt.tight_layout()
    plt.show()