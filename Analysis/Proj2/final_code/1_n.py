from mpi4py import MPI
import csv

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define number of mappers and reducers
num_mappers = min(size, 4)  # You can set this to any number <= size
num_reducers = min(size, 4)  # You can set this to any number <= size

def map_sales_data(data_chunk):
    """Map function to create (customer_id, amount) tuples."""
    mapped_data = []
    for row in data_chunk:
        customer_id = row[0]
        amount = float(row[2])
        mapped_data.append((customer_id, amount))
    return mapped_data

def reduce_sales_data(mapped_data):
    """Reduce function to sum amounts by customer_id."""
    reduced_data = {}
    for customer_id, amount in mapped_data:
        if customer_id in reduced_data:
            reduced_data[customer_id] += amount
        else:
            reduced_data[customer_id] = amount
    return list(reduced_data.items())

if rank == 0:
    # Root process: Read the data and distribute chunks
    with open('/gpfs/projects/AMS598/projects/proj2/sales_data.txt', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        data = list(reader)

    # Calculate chunk size for mappers
    chunk_size = len(data) // num_mappers
    chunks = [data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_mappers)]

    # Handle any remaining data if not evenly divisible
    remainder = len(data) % num_mappers
    if remainder != 0:
        for i in range(remainder):
            chunks[i].append(data[num_mappers * chunk_size + i])

else:
    chunks = None

# Scatter data chunks to mappers
data_chunk = comm.scatter(chunks, root=0)

# Each mapper maps its data chunk
if rank < num_mappers and data_chunk is not None:
    mapped_data = map_sales_data(data_chunk)
else:
    mapped_data = []

# Gather all mapped data at root from mappers only
all_mapped_data = comm.gather(mapped_data, root=0)

if rank == 0:
    # Root process: Combine mapped data from all mappers
    combined_mapped_data = []
    for data in all_mapped_data:
        combined_mapped_data.extend(data)

    # Group by customer_id
    grouped_data = {}
    for customer_id, amount in combined_mapped_data:
        if customer_id not in grouped_data:
            grouped_data[customer_id] = []
        grouped_data[customer_id].append(amount)

    # Prepare keys and split grouped data for reducers
    keys = list(grouped_data.keys())
    chunk_size = len(keys) // num_reducers + (len(keys) % num_reducers > 0)
    keys_chunks = [keys[i * chunk_size:(i + 1) * chunk_size] for i in range(num_reducers)]

    # Split grouped data according to keys_chunks
    grouped_chunks = [{key: grouped_data[key] for key in keys_chunk} for keys_chunk in keys_chunks]

else:
    grouped_chunks = None

# Scatter grouped data to reducers only
grouped_chunk = comm.scatter(grouped_chunks, root=0)

# Each reducer processes its chunk and sums amounts
if rank < num_reducers and grouped_chunk is not None:
    reduced_results = []
    for customer_id, amounts_list in grouped_chunk.items():
        total_amount = sum(amounts_list)
        reduced_results.append((customer_id, total_amount))
else:
    reduced_results = []

# Gather all reduced results at root from reducers only
all_reduced_results = comm.gather(reduced_results, root=0)

if rank == 0:
    # Combine results from all reducers and print final output
    final_results = []
    for results in all_reduced_results:
        final_results.extend(results)
    
    print("Total Amount Spent by Each Customer:")
    for customer_id, total_amount in final_results:
        print(f"Customer ID {customer_id}: ${total_amount:.2f}")

