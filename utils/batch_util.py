from typing import List
import threading

def batch_fn(fn: callable, batch_num:int, *args, **kwargs) -> List:
    results = {}
    threads = []
    # add result
    def add_result(b):
        arg_minibatch = [args[i][b] for i in range(len(args))]
        results[b] = fn(*arg_minibatch, **kwargs)
    # start threads
    for b in range(batch_num):
        t = threading.Thread(target=add_result, args=(b,))
        t.start()
        threads.append(t)
    # join threads
    for t in threads:
        t.join()
    # get results sorted by batch
    results = [results[i] for i in range(batch_num)]
    return results