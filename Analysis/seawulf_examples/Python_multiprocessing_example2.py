import multiprocessing
import time

class Process(multiprocessing.Process):
    def __init__(self, id):
        super(Process, self).__init__()
        self.id = id

    def run(self):
        time.sleep(1)
        print("I'm the process with id: {}".format(self.id))

if __name__ == '__main__':
    p1 = Process(0)
    p1.start()
    p1.join()
    p2 = Process(1)
    p2.start()
    p2.join()
