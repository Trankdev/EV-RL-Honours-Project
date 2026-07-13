import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_comparison():
    # 1. SETUP DATA
    # ---------------------------------------------------------
    # Load baseline result from run_baseline_pure_traci.py
    if os.path.exists("baseline_result.txt"):
        with open("baseline_result.txt", "r") as f:
            lines = f.readlines()
            baseline_time = float(lines[0].strip())
            baseline_civilian_wait = float(lines[1].strip()) if len(lines) > 1 else 0
        print(f"✅ Loaded baseline - Ambulance: {baseline_time}s, Civilian wait: {baseline_civilian_wait:.2f}s")
    else:
        print("⚠️ baseline_result.txt not found! Run 'run_baseline_pure_traci.py' first.")
        baseline_time = 0
        baseline_civilian_wait = 0
    
    # Load optimized agent result from test_optimized.py
    if os.path.exists("optimized_result.txt"):
        with open("optimized_result.txt", "r") as f:
            lines = f.readlines()
            rl_time = float(lines[0].strip())
            rl_civilian_wait = float(lines[1].strip()) if len(lines) > 1 else 0
        print(f"✅ Loaded optimized - Ambulance: {rl_time}s, Civilian wait: {rl_civilian_wait:.2f}s")
    else:
        print("⚠️ optimized_result.txt not found! Run 'test_optimized.py' first.")
        rl_time = 0
        rl_civilian_wait = 0
    
    # Load training results if available
    if os.path.exists("training_results.csv"):
        df = pd.read_csv("training_results.csv")
        has_training_data = True
    else:
        print("⚠️ training_results.csv not found! Skipping learning curve.")
        has_training_data = False
    # ---------------------------------------------------------

    # 2. CREATE THE PLOTS
    if has_training_data and baseline_time > 0:
        # 3 subplots: learning curve, ambulance comparison, civilian comparison
        fig = plt.figure(figsize=(18, 5))
        
        # Learning Curve
        plt.subplot(1, 3, 1)
        sns.lineplot(data=df, x='step', y='system_mean_waiting_time', label='RL Agent')
        plt.axhline(y=baseline_civilian_wait, color='r', linestyle='--', label='Fixed-Time Baseline')
        plt.title("Learning Curve: Traffic Waiting Time")
        plt.xlabel("Training Steps")
        plt.ylabel("Avg Waiting Time (s)")
        plt.legend()
        plt.grid(True)

        # Ambulance comparison
        plt.subplot(1, 3, 2)
    else:
        # 2 subplots: ambulance and civilian comparison
        fig = plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
    
    data = {'System': ['Fixed-Time (Baseline)', 'PPO Agent (Optimized)'], 
            'Time (s)': [baseline_time, rl_time]}
    
    sns.barplot(x='System', y='Time (s)', data=data, palette=['red', 'green'])
    plt.title("Ambulance Transit Time Comparison")
    plt.ylabel("Transit Time in seconds (Lower is Better)")
    
    # Add text labels on bars
    if baseline_time > 0 and rl_time > 0:
        for i, v in enumerate([baseline_time, rl_time]):
            plt.text(i, v + 1, str(round(v, 2)), ha='center', fontweight='bold')
        
        # Calculate improvement
        improvement = ((baseline_time - rl_time) / baseline_time) * 100
        plt.text(0.5, max(baseline_time, rl_time) * 0.9, 
                f"Improvement: {improvement:.1f}%", 
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # Civilian waiting time comparison
    if has_training_data and baseline_time > 0:
        plt.subplot(1, 3, 3)
    else:
        plt.subplot(1, 2, 2)
    
    civilian_data = {'System': ['Fixed-Time (Baseline)', 'PPO Agent (Optimized)'], 
                     'Wait Time (s)': [baseline_civilian_wait, rl_civilian_wait]}
    
    sns.barplot(x='System', y='Wait Time (s)', data=civilian_data, palette=['red', 'green'])
    plt.title("Civilian Vehicle Avg Waiting Time")
    plt.ylabel("Avg Waiting Time in seconds (Lower is Better)")
    
    # Add text labels on bars
    if baseline_civilian_wait > 0 and rl_civilian_wait > 0:
        for i, v in enumerate([baseline_civilian_wait, rl_civilian_wait]):
            plt.text(i, v + 1, str(round(v, 2)), ha='center', fontweight='bold')
        
        # Calculate improvement
        civilian_improvement = ((baseline_civilian_wait - rl_civilian_wait) / baseline_civilian_wait) * 100
        plt.text(0.5, max(baseline_civilian_wait, rl_civilian_wait) * 0.9, 
                f"Improvement: {civilian_improvement:.1f}%", 
                ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_comparison()