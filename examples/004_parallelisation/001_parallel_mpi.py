import numpy as np
from mpi4py import MPI


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def parprint(string):
    if rank == 0:
        print(string)


# Initialise all objects on all processors
# ========================================
a = None
t = None
ring = range(5)
ring_tag = []
for i, r in enumerate(ring):
    if i == 3:
        ring_tag.append('serial')
    else:
        ring_tag.append('parallel')

n_total = 12
n_parts_per_process = n_total/size


# Selective initialisation on processors
# ======================================
parprint("\n--> Each processor initilising part...")
if rank == 0:
    t = np.zeros((size, n_parts_per_process), dtype='float64')

a = np.arange(rank*n_parts_per_process, (rank+1)*n_parts_per_process,
              dtype='float64')
comm.Barrier()
parprint("\n--> Now starting loop...")


# Start loop
# ==========
for tt in range(10):
    parprint("\n\n*** Iteration {:d} at the start of ring...".format(tt))
    lock = ring_tag.index('serial')
    for j, r in enumerate(ring[:lock]):
        print("Running on processor {:d}, part {:s} ".format(rank, a) +
              "through element {:d}".format(j))
    comm.Barrier()


    # Gather and do something
    # =======================
    parprint("\n--> Gathering and doing wakes...")
    comm.Gather(sendbuf=a, recvbuf=t, root=0)
    if rank == 0:
        t += 10
    comm.Scatter(sendbuf=t, recvbuf=a)


    # Print whats on processors
    # =========================
    parprint("\n--> Back to rest of ring...")
    print("Processor {:d} - {:s}".format(rank, a))
    comm.Barrier()
    for j, r in enumerate(ring[lock:]):
        print("Running on processor {:d}, part {:s} through element {:d}".format(
            rank, a, j+lock))
    comm.Barrier()

parprint("\n*** Done!")
