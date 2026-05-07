import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
from scipy.optimize import curve_fit

st.set_page_config(page_title="Nuclear Decay Simulator", layout="wide")
st.title("☢️ Nuclear Decay Simulator")

# Initialize session state for the grid
if "grid" not in st.session_state:
    st.session_state.grid = None
if "step_count" not in st.session_state:
    st.session_state.step_count = 0
if "history" not in st.session_state:
    st.session_state.history = []
if "playing" not in st.session_state:
    st.session_state.playing = False

# Sidebar controls
st.sidebar.header("Parameters")
N = st.sidebar.slider("Grid Size (N)", min_value=1, max_value=100, value=10,
                       help="Grid size will be N × N")
p = st.sidebar.slider("Decay Probability (p)", min_value=0.01, max_value=10.0, value=10.0, step=0.01, format="%.2f%%",
                       help="Probability of decay per step for each active particle (in percent)")

delay_ms = st.sidebar.slider("Play Delay (ms)", min_value=1, max_value=1000, value=200,
                              help="Delay between decay steps when playing (in milliseconds)")

show_exp_fit = st.sidebar.checkbox("Display exponential fit", value=False,
                                    help="Show exponential fit curve overlaid on the decay graph")

grid_size = N

# Initialize grid
if st.sidebar.button("Initialize Grid"):
    st.session_state.grid = np.full((grid_size, grid_size), 2, dtype=int)  # 2 = active (dark grey)
    st.session_state.step_count = 0
    st.session_state.history = [grid_size ** 2]  # Start with all particles active

# Create grid if it doesn't exist
if st.session_state.grid is None:
    st.session_state.grid = np.full((grid_size, grid_size), 2, dtype=int)  # 2 = active (dark grey)

# Decay function
def decay_step(grid, probability):
    """
    Perform one decay step on the grid.
    States: 2 = active (dark grey), 1 = just decayed (red), 0 = inactive 2+ steps (white)
    """
    new_grid = grid.copy()
    
    # First, age the recently decayed particles: 1 -> 0
    new_grid[grid == 1] = 0
    
    # Then check for new decays among active particles
    active_indices = np.where(grid == 2)  # Get indices of active particles (state 2)
    
    for i, j in zip(active_indices[0], active_indices[1]):
        if np.random.random() < probability:
            new_grid[i, j] = 1  # Decay: become red (state 1)
    
    return new_grid

# Main interface
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Simulation Grid & Decay Curve")
    
    grid_col, graph_col = st.columns(2)
    
    with grid_col:
        st.write("Active Particles")
        # Create visualization
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        
        # Custom colormap: white (0) = inactive 2+ steps, red (1) = just decayed, dark grey (2) = active
        colors = ['white', '#ff0000', '#2b2b2b']
        cmap = ListedColormap(colors)
        
        # Display grid (flip vertically for better visualization)
        im = ax.imshow(st.session_state.grid, cmap=cmap, vmin=0, vmax=2, origin='upper')
        
        # Add gridlines for borders
        grid_size_val = st.session_state.grid.shape[0]
        ax.set_xticks(np.arange(-0.5, grid_size_val, 1), minor=False)
        ax.set_yticks(np.arange(-0.5, grid_size_val, 1), minor=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(which='major', color='black', linestyle='-', linewidth=0.5)
        
        active_count = np.sum(st.session_state.grid == 2)
        ax.set_title(f"Step {st.session_state.step_count} | Active: {active_count}", fontsize=20)
        
        st.pyplot(fig)
        plt.close(fig)
    
    with graph_col:
        st.write("Particle Decay Over Time")
        fig_history, ax_history = plt.subplots(figsize=(6, 6), dpi=100)
        ax_history.plot(st.session_state.history, linewidth=2, color='#2b2b2b', label='Data')
        ax_history.fill_between(range(len(st.session_state.history)), st.session_state.history, alpha=0.3, color='#2b2b2b')
        
        half_life_steps = None
        # Exponential fit if enabled
        if show_exp_fit and len(st.session_state.history) > 3:
            try:
                def exponential(x, a, b):
                    return a * np.exp(-b * x)
                
                x_data = np.array(range(len(st.session_state.history)))
                y_data = np.array(st.session_state.history, dtype=float)
                
                # Initial guess based on data
                a_guess = y_data[0]
                b_guess = 0.01
                
                # Fit the exponential curve
                popt, _ = curve_fit(exponential, x_data, y_data, p0=[a_guess, b_guess], maxfev=10000)
                
                # Generate smooth curve
                x_smooth = np.linspace(0, len(st.session_state.history) - 1, 300)
                y_smooth = exponential(x_smooth, *popt)
                
                ax_history.plot(x_smooth, y_smooth, linewidth=2.5, color='orange', linestyle='--', label='Exponential Fit')
                ax_history.legend(fontsize=12)
                
                # Calculate half-life estimate from fit parameter b
                b_value = popt[1]
                if b_value > 0:
                    half_life_steps = np.log(2) / b_value
            except Exception:
                half_life_steps = None
        
        ax_history.set_xlabel("Time (steps)", fontsize=16)
        ax_history.set_ylabel("Number of Active Particles", fontsize=16)
        ax_history.tick_params(axis='both', labelsize=14)
        ax_history.grid(True, alpha=0.3)
        ax_history.set_ylim(bottom=0)
        st.pyplot(fig_history)
        plt.close(fig_history)
        
        if show_exp_fit and half_life_steps is not None:
            st.write(f"Estimated half-life: {half_life_steps:.1f} steps")
        elif show_exp_fit:
            st.write("Estimated half-life: not available yet")

with col2:
    st.subheader("Controls")
    
    col_play, col_stop = st.columns(2)
    with col_play:
        if st.button("▶ Play", use_container_width=True):
            st.session_state.playing = True
    with col_stop:
        if st.button("⏹ Stop", use_container_width=True):
            st.session_state.playing = False
    
    if st.button("Decay Step", use_container_width=True):
        st.session_state.grid = decay_step(st.session_state.grid, p / 100.0)
        st.session_state.step_count += 1
        st.session_state.history.append(np.sum(st.session_state.grid == 2))
        st.rerun()
    
    st.divider()
    st.metric("Step Count", st.session_state.step_count)
    st.metric("Active Particles", np.sum(st.session_state.grid == 2))
    st.metric("Decay Percentage", f"{(1 - np.sum(st.session_state.grid == 2) / (grid_size ** 2)) * 100:.1f}%")

# Auto-play loop
if st.session_state.playing:
    time.sleep(delay_ms / 1000.0)  # Convert milliseconds to seconds
    st.session_state.grid = decay_step(st.session_state.grid, p / 100.0)
    st.session_state.step_count += 1
    st.session_state.history.append(np.sum(st.session_state.grid == 2))
    st.rerun()
