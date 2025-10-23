#!/usr/bin/env python3
"""
Script to rerun analysis for a specified evaluations directory.
"""

import argparse
import os
import sys

sys.path.append("templates")
from analyze_simulation import analyze_simulation

from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
num_ranks = MPI.COMM_WORLD.Get_size()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluations_directory",
        type=str,
        required=True,
        help="Path to the evaluations directory containing the data to analyze",
    )
    args = parser.parse_args()

    # Validate that the directory exists
    if not os.path.exists(args.evaluations_directory):
        print(f"Error: Directory '{args.evaluations_directory}' does not exist.")
        sys.exit(1)

    # List all simulation folders in the evaluations directory
    simulation_folders = [
        f for f in os.listdir(args.evaluations_directory) if f.startswith("sim")
    ]
    simulation_folders.sort()
    n_sims = len(simulation_folders)

    # Loop through simulations in parallel
    for i in range(rank, n_sims, num_ranks):
        print(f"Rank {rank} analyzing simulation {simulation_folders[i]}")

        sim_folder = os.path.join(args.evaluations_directory, simulation_folders[i])
        analyze_simulation(sim_folder, upload_to_db=False)

        print(f"Rank {rank} finished analyzing simulation {simulation_folders[i]}")
