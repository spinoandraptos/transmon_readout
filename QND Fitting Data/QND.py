import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# To adjust according to experiment
day ="2025-07-16"

start_time = datetime.strptime("15-52-00", "%H-%M-%S")
end_time   = datetime.strptime("15-54-00", "%H-%M-%S")

# data_directory = r"/home/spinoandraptos/Documents/CQT/Experiments/QND Fitting Data"
# CLEAR_directory = r"/home/spinoandraptos/Documents/CQT/Experiments/QND Fitting Data"

data_directory = f"C:\\Users\\qcrew\\Documents\\jon\\cheddar\\data\{day}"
CLEAR_directory = "C:\\Users\\qcrew\\Documents\\jon\\cheddar\\scripts\\CLEAR"

def get_QND(label, filepath1, filepath2, ax=None):

    with h5py.File(filepath1, 'r') as f:
        pre1 = f["single_shot_pre"][()]
        post1 = f["single_shot_post"][()]

    with h5py.File(filepath2, 'r') as f:
        pre2 = f["single_shot_pre"][()]
        post2 = f["single_shot_post"][()]
    
    pre_flat1 = pre1.flatten()   
    post_flat1 = post1.flatten() 
    pre_flat2 = pre2.flatten()   
    post_flat2 = post2.flatten() 

    pre_combined = np.concatenate([pre_flat1, pre_flat2])
    post_combined = np.concatenate([post_flat1, post_flat2])
  
    #Calculate matches/mismatches
    def get_counts(pre, post):
        total = len(pre)
        match = np.sum(pre == post)
        mismatch = total - match
        mismatch_rate = mismatch / total * 100
        return total, match, mismatch, mismatch_rate

    total1, match1, mismatch1, rate1 = get_counts(pre_flat1, post_flat1)
    total2, match2, mismatch2, rate2 = get_counts(pre_flat2, post_flat2)
    total_comb, match_comb, mismatch_comb, rate_comb = get_counts(pre_combined, post_combined)

    # Prepare data for grouped bar plot
    categories = ["Match", "Mismatch"]
    pre1_vals = [match1, mismatch1]
    pre2_vals = [match2, mismatch2]
    combined_vals = [match_comb, mismatch_comb]

    x = np.arange(len(categories))  # [0, 1]
    width = 0.25  # width of bars

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,6))
    else:
        fig = ax.figure

    bars1 = ax.bar(x - width, pre1_vals, width, label='Ground', color='tab:blue')
    bars2 = ax.bar(x,       pre2_vals, width, label='Excited', color='tab:orange')
    bars3 = ax.bar(x + width, combined_vals, width, label='Combined', color='tab:green')

    ax.set_ylabel('Counts')
    ax.set_title(f'{label} QND-ness')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    # Function to add annotations on bars
    def autolabel(bars, values, rates):
        for bar, val, rate in zip(bars, values, rates):
            height = bar.get_height()
            ax.annotate(f'{val}\n({rate:.1f}%)',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),  # 5 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    # Annotate bars with counts and percentages
    autolabel(bars1, pre1_vals, [100*val/sum(pre1_vals) for val in pre1_vals])
    autolabel(bars2, pre2_vals, [100*val/sum(pre2_vals) for val in pre2_vals])
    autolabel(bars3, combined_vals, [100*val/sum(combined_vals) for val in combined_vals])

    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":

    photon_list = []
    savepath = os.path.join(CLEAR_directory, "QND.png")

    file_entries = []
    for filename in os.listdir(data_directory):
        if filename.endswith(".hdf5"):
            try:
                timestamp_str = filename.split("_")[0]
                file_time = datetime.strptime(timestamp_str, "%H-%M-%S")
                if start_time < file_time < end_time:
                    file_entries.append((file_time, filename))
            except ValueError:
                continue  # skip files that don't match the format

    # Sort by file_time (ascending)
    file_entries.sort()
    sorted_filenames = [filename for _, filename in file_entries]

    # Group 1: files at index 0 and 2 (1st and 3rd)
    group1_files = [sorted_filenames[0], sorted_filenames[2]]

    # Group 2: files at index 1 and 3 (2nd and 4th)
    group2_files = [sorted_filenames[1], sorted_filenames[3]]

    fig, axs = plt.subplots(1, 2, figsize=(16,6))
    
    fig1, ax1 = get_QND("Const", os.path.join(data_directory, group1_files[0]), os.path.join(data_directory, group1_files[1]), ax=axs[0])
    fig2, ax2 = get_QND("CLEAR", os.path.join(data_directory, group2_files[0]), os.path.join(data_directory, group2_files[1]), ax=axs[1])
    plt.tight_layout()
    plt.savefig(savepath,dpi=300)
    plt.close()
    


