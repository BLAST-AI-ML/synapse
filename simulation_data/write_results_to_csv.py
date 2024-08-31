import os
import sys
import numpy as np
import pandas as pd

if __name__ == "__main__":

    # Directory containing exploration history
    exp_dir = './exploration'

    # History file prefix to match
    prefix = 'exploration_history_after_sim_'

    # Initialize variables
    largest_number = -1
    selected_file = None
    found_files = []

    # Iterate through files in the directory
    for filename in os.listdir(exp_dir):
        if filename.startswith(prefix) and filename.endswith('.npy'):
            # Extract the number from the filename
            try:
                # Extract the number of evaluations after the prefix
                number = int(filename[len(prefix):-4])
                found_files.append(filename)
                if number > largest_number:
                    largest_number = number
                    selected_file = filename
            except ValueError:
                continue

    if found_files:
        # Print found files
        print(f"Files found: {found_files}")
        print(f"Selected file: {selected_file}")
    else:
        print("No files found! Please check that the scan ran successfully!")
        sys.exit(1)

    # Load data from the selected file
    data = np.load(os.path.join(exp_dir, selected_file))
    df = pd.DataFrame(data)

    # Subselect columns into a new dataframe to write into its own CSV file
    select_cols = ['z_pos_um', 'TOD_fs3', 'f']
    df_res = df[select_cols].copy()
    df_res.rename(columns={'z_pos_um' : 'target_z_pos_um', 'f' : 'num_particles'}, inplace=True)

    res_file = './simulation_results.csv'
    df_res.to_csv(res_file)

    abs_res_path = os.path.abspath(res_file)
    print(f"Results written to {abs_res_path}.")
