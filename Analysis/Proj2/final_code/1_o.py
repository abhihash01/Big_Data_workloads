from __future__ import division
from __future__ import print_function

import numpy as np
from mpi4py import MPI
from collections import defaultdict

import csv

def map_function(data):
    mapped = defaultdict(float)
    for row in data:
        customer_id, _, amount = row
        mapped[customer_id] += float(amount)
    return mapped

def reduce_function(key, values):
    return sum(values)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Read the input file
        with open('/gpfs/projects/AMS598/projects/proj2/sales_data.txt', 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            data = list(reader)

        # Distribute data among processes
        chunk_size = len(data) // size
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    else:
        chunks = None

    # Scatter data to all processes
    local_chunk = comm.scatter(chunks, root=0)

    # Map phase
    local_mapped = map_function(local_chunk)

    # Shuffle phase
    all_mapped = comm.gather(local_mapped, root=0)
    
    if rank == 0:
        shuffled = defaultdict(list)
        for mapped in all_mapped:
            for key, value in mapped.items():
                shuffled[key].append(value)
        
        # Distribute keys for reduce phase
        keys = list(shuffled.keys())
        keys_per_process = len(keys) // size
        key_chunks = [keys[i:i + keys_per_process] for i in range(0, len(keys), keys_per_process)]
    else:
        shuffled = None
        key_chunks = None

    # Scatter keys to all processes
    local_keys = comm.scatter(key_chunks, root=0)

    # Broadcast shuffled data to all processes
    shuffled = comm.bcast(shuffled, root=0)

    # Reduce phase
    local_reduced = {}
    for key in local_keys:
        local_reduced[key] = reduce_function(key, shuffled[key])

    # Gather all reduced results to root
    all_reduced = comm.gather(local_reduced, root=0)

    if rank == 0:
        # Combine all reduced results
        final_result = {}
        for reduced in all_reduced:
            final_result.update(reduced)

        # Sort and print the results
        for customer_id, total_amount in sorted(final_result.items()):
            print(f"{customer_id} {total_amount:.0f}")

if __name__ == "__main__":
    main()



