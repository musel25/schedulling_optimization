import numpy as np, math, itertools, time

proc = {
 'J1':[6,5,3,9,3],
 'J2':[7,6,2,8,2],
 'J3':[5,7,4,7,4],
 'J4':[6,5,3,6,3],
 'J5':[8,6,2,9,2]
}
def makespan(seq):
    m = len(proc[seq[0]])
    comp = np.zeros((len(seq),m))
    for i,j in enumerate(seq):
        for k in range(m):
            a = comp[i-1,k]   if i>0 else 0
            b = comp[i,k-1]   if k>0 else 0
            comp[i,k] = max(a,b)+proc[j][k]
    return comp[-1,-1]

def neh(jobs):
    order = sorted(jobs, key=lambda j:-sum(proc[j]))   # descending Σp
    seq = [order[0]]
    for j in order[1:]:
        best = math.inf
        best_seq = None
        for pos in range(len(seq)+1):
            trial = seq[:pos]+[j]+seq[pos:]
            cmax = makespan(trial)
            if cmax < best:
                best, best_seq = cmax, trial
        seq = best_seq
    return seq, best

t0=time.perf_counter()
seq,cmax = neh(list(proc.keys()))
heur_time = time.perf_counter()-t0
print(f"NEH sequence : {seq}")
print(f"NEH makespan : {cmax}  (CPU {heur_time*1e3:.2f} ms)")


