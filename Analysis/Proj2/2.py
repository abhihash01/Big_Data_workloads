from mpi4py import MPI
import csv
import random
from collections import defaultdict

def read_data():
    data_r1, data_r2, data_r3 = [], [], []

    with open('/gpfs/projects/AMS598/projects/proj2/data_R1.txt', newline='') as file:
        reader = csv.reader(file,delimiter='\t')
        for row in reader:
            data_r1.append(('R1', row))

    with open('/gpfs/projects/AMS598/projects/proj2/data_R2.txt', newline='') as file:
        reader = csv.reader(file,delimiter='\t')
        for row in reader:
            data_r2.append(('R2', row))

    with open('/gpfs/projects/AMS598/projects/proj2/data_R3.txt', newline='') as file:
        reader = csv.reader(file,delimiter='\t')
        for row in reader:
            data_r3.append(('R3', row))

    combined_data = data_r1 + data_r2 + data_r3
    return combined_data

def shuffle_data(combined_data):
    random.shuffle(combined_data)
    return combined_data

def mapper(data):
    mapped_data = []
    for key, row in data:
        try:
            if key == 'R1':
                a, b, c = row
                mapped_data.append((a, ('R1', b, c)))
            elif key == 'R2':
                a, d, e = row
                mapped_data.append((a, ('R2', d, e)))
            elif key == 'R3':
                a, f, g = row
                mapped_data.append((a, ('R3', f, g)))
        except ValueError as e:
            print(f"Error unpacking row {row} with key {key}: {e}")
            print("the key values are")
            print(key)
            print(row)
    return mapped_data

def group_by_keys(mapped_data):
    grouped_data = defaultdict(list)
    for key, value in mapped_data:
        grouped_data[key].append(value)
    return list(grouped_data.items())

def distribute_keys(grouped_data, num_reducers):
    scattered_keys = [[] for _ in range(num_reducers)]
    for index, (key, values) in enumerate(grouped_data):
        scattered_keys[index % num_reducers].append((key, values))
    return scattered_keys

def reducer(local_keys):
    final_joined_results = []

    for key, values in local_keys:
        r1, r2, r3 = [], [], []
        for item in values:
            s, x, extra = item
            if s == 'R1':
                r1.append((x, extra))
            elif s == 'R2':
                r2.append((x, extra))
            elif s == 'R3':
                r3.append((x, extra))

        for b_c in r1:
            for d_e in r2:
                for f_g in r3:
                    final_joined_results.append((key,) + b_c + d_e + f_g)

    return final_joined_results

def gather_results(reducer_outputs):
    final_results = []
    for res in reducer_outputs:
        final_results.extend(res)
    return final_results

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        combined_data = read_data()
        shuffled_data = shuffle_data(combined_data)
        
        chunks_per_process = len(shuffled_data) // size
        #scattered_chunks = [shuffled_data[i:i + chunks_per_process] for i in range(0, len(shuffled_data), chunks_per_process)]
        remainder = len(shuffled_data) % size

        scattered_chunks = [shuffled_data[i*chunks_per_process + min(i, remainder):(i+1)*chunks_per_process + min(i+1, remainder)] for i in range(size)]

    else:
        scattered_chunks = None

    local_data = comm.scatter(scattered_chunks, root=0)
    
    local_mapped_data = mapper(local_data)
    
    all_mapped_data = comm.gather(local_mapped_data, root=0)

    if rank == 0:
        flat_mapped_data = [item for sublist in all_mapped_data for item in sublist]
        
        grouped_data = group_by_keys(flat_mapped_data)
        
        scattered_keys = distribute_keys(grouped_data, size)
        
    else:
        scattered_keys = None
        
    local_keys = comm.scatter(scattered_keys, root=0)
    
    result = reducer(local_keys)

    all_results = comm.gather(result, root=0)

    if rank == 0:
        final_table = gather_results(all_results)
        print(len(final_table))
        #for row in final_table:
        #    print(row)
        
        with open('results_2.txt', 'w') as file:
            for row in final_table:
                # Write each row to the file, followed by a newline
                file.write(str(row) + '\n')
        


if __name__ == '__main__':
    main()
