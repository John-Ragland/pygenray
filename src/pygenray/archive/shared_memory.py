"""
play around with and learn how to use shared memory
"""

import numpy as np
from multiprocessing import Process, shared_memory
import time
import multiprocessing as mp

def worker(args):
    # Unpack the arguments
    shm_name, shape, dtype = args
    
    # Attach to the existing shared memory block
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    
    # Create a NumPy array using the shared memory buffer
    shared_array = np.ndarray(shape=shape, dtype=dtype, buffer=existing_shm.buf)
    
    # Give some time to see the changes from other processes
    time.sleep(2)
    
    result = shared_array[np.random.randint(0, shape[0]), np.random.randint(0, shape[1])]  # Print a random element from the array

    print(result)
    # Cleanup
    existing_shm.close()

    return result

def main():
    # Define the shape and data type for our 2D array
    shape = (1000, 500) 
    dtype = np.float64
    
    # Create the shared memory
    shm = shared_memory.SharedMemory(create=True, size=np.zeros(shape, dtype=dtype).nbytes)
    
    # Create a NumPy array that uses the shared memory
    shared_array = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
    
    # Initialize the array
    shared_array[:] = np.random.rand(*shape)  # Fill with random values

    # Create list of input tuples for worker
    args_list = [(shm.name, shape, dtype) for _ in range(12)]
    
    # Run workers with all required arguments
    with mp.Pool(processes=12) as pool:
        result = pool.map(worker, args_list)
    
    # Clean up the shared memory
    shm.close()
    shm.unlink()  # Free and remove the shared memory block

    return result

if __name__ == "__main__":
    main()