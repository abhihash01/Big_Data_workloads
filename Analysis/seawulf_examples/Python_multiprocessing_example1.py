import multiprocessing
import os

def worker1():
    print("ID process running worker 1: {}".format(os.getpid()))
    
def worker2():
    print("ID process running worker 2: {}".format(os.getpid()))

if __name__ == '__main__':
    print("ID of the main process: {}".format(os.getpid()))
    
    # create processes
    p1 = multiprocessing.Process(target=worker1)
    p2 = multiprocessing.Process(target=worker2)
    
    # starting processes
    p1.start()
    p2.start()
    
    # process IDs
    print("ID of process p1: {}".format(p1.pid))
    print("ID of process p2: {}".format(p2.pid))
          
    p1.join()
    p2.join()



