# from multiprocessing import Process, Manager, Value, Lock
# import time
# l = Lock()
# manager = Manager()
# def f(a, num):
#     # l.acquire()
#     a.put(num)
#     # l.release()
#     # print(a) 
# def get_q(q):
#     relay = []
#     while True:
#         try:
#             time.sleep(1)
#             print(q)
#             relay.append(q.get())
#             if a.empty():
#                 break
#         except:
#             print('emty queue')
#             break
#     print(relay)

# # a = Value('i', 1)
# a = manager.Queue(maxsize=100)

# pro = Process(target=f, args=(a,2))
# pro2 = Process(target=f, args=(a,3))
# pro3 = Process(target=get_q, args=(a,))

# pro.start()
# pro2.start()
# pro3.start()
# pro.join()
# pro2.join()
# pro3.join()
# # print(type(a.value))

from dqn.utils import load_pkl
from dqn.variables import *

a = load_pkl(ALGEBRA_R_BUFFER)





b = load_pkl(GEOMETRY_R_BUFFER)
c = load_pkl(PROBABILITY_R_BUFFER)
d = load_pkl(ANALYSIS_R_BUFFER)
e = load_pkl(GRAMMAR_R_BUFFER)
f = load_pkl(VOCABULARY_R_BUFFER)

print(a)