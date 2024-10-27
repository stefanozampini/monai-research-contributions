import os

cores_per_process = -1
def __init__():
  HT=2
  local_rank = int(os.environ.get('LOCAL_RANK',0))
  processes_per_node = int(os.environ.get('LOCAL_WORLD_SIZE',0))
  cores_per_node = int(os.environ.get('CORES_PER_NODE',os.cpu_count()//HT))
  if processes_per_node > 0:
    global cores_per_process
    cores_per_process = cores_per_node // processes_per_node
    pid = os.getpid()
    #aff0 = os.sched_getaffinity(pid)
    r1 = range(local_rank*cores_per_process, (local_rank + 1)*cores_per_process)
    affinity = set(list(r1))
    #r2 = range(cores_per_node + local_rank*cores_per_process, cores_per_node + (local_rank + 1)*cores_per_process)
    #affinity = set(list(r1) + list(r2))
    #print(pid, local_rank, affinity)
    os.sched_setaffinity(pid, affinity)
    #affp = os.sched_getaffinity(pid)

    proclist=','.join(str(c) for c in affinity)
    #os.environ["KMP_AFFINITY"]=f"verbose,proclist=[{proclist}],explicit"
    os.environ["KMP_AFFINITY"]=f"proclist=[{proclist}],explicit"
    #print(pid, local_rank, os.cpu_count(),os.environ["KMP_AFFINITY"])
    #exit()

__init__()
