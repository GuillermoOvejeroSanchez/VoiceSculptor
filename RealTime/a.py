import sched, time
from itertools import count
import time

#event = sched.scheduler(time.time, time.sleep)

index = 0


def save_data(event): 
    print("Evento")
    #event.enter(1, 1, save_data, (event,))


start_time = time.time()
while (True):
    index += 1
    print(index)
    if((time.time() - start_time) % 1.0 == 0.0):
        print("--------------------------")
    #event.enter(1, 1, save_data, (event,))
    #event.run()
