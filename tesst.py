from multiprocessing import Process, Queue, Value, Lock, Manager
import time
class test():
    def __init__(self, value) -> None:
        self.value = value

    
    def run(self, number):
        self.value+=number
        time.sleep(1)

def f(a:test, num, l, work):

    
    while True:
        time.sleep(1)
        if work:
            l.append([])
            try:
                l[-1] =0
            except:
                pass
        print(work, l)

def get_q(q):
    relay = []
    while True:
        try:
            time.sleep(1)
            print('get ',a.value)

        except:
            print('emty queue')
            break


# a = Value('i', 1)
a = Value('i',0)
m = Manager()
l = m.list()

pro = Process(target=f, args=(test,1 ,l,True))
pro2 = Process(target=f, args=(test,1,l, False))
# pro3 = Process(target=get_q, args=(a,))

pro.start()
pro2.start()
# pro3.start()
pro.join()
pro2.join()
# pro3.join()
# print(type(a.value))

# from dqn.utils import load_pkl
# from dqn.variables import *

# a = load_pkl(ALGEBRA_R_BUFFER)





# b = load_pkl(GEOMETRY_R_BUFFER)
# c = load_pkl(PROBABILITY_R_BUFFER)
# d = load_pkl(ANALYSIS_R_BUFFER)
# e = load_pkl(GRAMMAR_R_BUFFER)
# f = load_pkl(VOCABULARY_R_BUFFER)

# print(a)