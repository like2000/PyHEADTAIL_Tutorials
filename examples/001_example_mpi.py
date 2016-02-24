from mpi4py import MPI
import sys


comm = MPI.COMM_WORLD
name = comm.Get_name()
rank = comm.Get_rank()
size = comm.Get_size()

sys.stdout.write(
    "Hello, World! I am process %d of %d on %s.\n"
    % (rank, size, name))


if rank == 0:
    data = ['bunch-{:d}'.format(i+1) for i in range(10)]
    comm.send(data, dest=1, tag=11)
elif rank == 1:
    data = comm.recv(source=0, tag=11)
