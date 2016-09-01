from collections import deque
from itertools import chain
from mpi4py import MPI
import numpy as np


class ParallelWakes(object):
    """Wake field applied in parallel fashion - test.

    """
    def __init__(self, slicer, wake_sources_list,
                 n_turns=1, circumference=0,
                 filling_scheme=None, comm=None):

        self.slicer = slicer
        wake_sources_list = np.atleast_1d(wake_sources_list)
        # self.wake_kicks = [source.get_wake_kicks(self.slicer)
        #                    for source in wake_sources_list]

        self.slice_set_deque = deque([], maxlen=n_turns)

        if n_turns > 1 and circumference == 0:
            raise ValueError(
                "Circumference must be provided for multi turn wakes!")
        self.circumference = circumference

        self.filling_scheme = filling_scheme
        self.comm = comm

        self.n_turns = n_turns
        try:
            self.n_bunches = len(list(chain(*filling_scheme)))
        except TypeError:
            self.n_bunches = len(filling_scheme)
        # self.n_bunches = len([a for b in filling_scheme for a in b])
        self.register = None
        if self.comm.rank == 0:
            self.register = np.zeros((self.n_bunches * (2+2*slicer.n_slices)))

    # @profile
    def track(self, beam):
        rank = self.comm.rank
        n_slices = self.slicer.n_slices
        print("Tracking on processor {:d}".format(rank))

        bunches_list = beam.split()
        slice_data = self.get_slice_data(bunches_list, self.slicer)

        '''Gather and scatter operations - check buffer requirements.  Create and
        process buffer locally... check performance compared to buffer as
        member.

        '''
        n_bunches_counts = self.comm.allgather(len(bunches_list))
        n_bunches_offsets = np.cumsum(n_bunches_counts)
        n_bunches_total = sum(n_bunches_counts)

        slice_data_counts = self.comm.allgather(len(slice_data))
        slice_data_offsets = np.insert(
            np.cumsum(slice_data_counts), 0, 0)[:-1]

        register = None
        if self.comm.rank == 0:
            register = np.zeros(
                (n_bunches_total * (2 + 2*n_slices)))

        self.comm.Gatherv(
            [slice_data, len(slice_data), MPI.DOUBLE],
            [register, slice_data_counts, slice_data_offsets, MPI.DOUBLE],
            root=0)

        slices_counts = n_slices * np.array(n_bunches_counts)
        slices_offsets = np.insert(np.cumsum(slices_counts), 0, 0)[:-1]
        kicks = np.zeros((n_bunches_counts[rank], n_slices))
        allkicks = None
        if self.comm.rank == 0:
            for b in bunches_list:
                b.dp += 100
            self.slice_set_deque.appendleft(
                np.reshape(register,
                           (n_bunches_total, 2 + 2*n_slices)))

            allkicks = np.array(
                [b[-n_slices:] for s in self.slice_set_deque
                 for b in s])
            print allkicks[:, :10]
            # print self.register[:, 0]
            # print self.register[:, 1]
            # print self.register[:, 2]
            # print self.register[1, :]

        self.comm.Scatterv(
            [allkicks, slices_counts, slices_offsets, MPI.DOUBLE],
            [kicks, MPI.DOUBLE], root=0)
        print kicks.shape, kicks[0, :10]
        self.comm.Barrier()

    def get_slice_data(self, bunches_list, slicer, moments=None):

        slice_set_list = [b.get_slices(slicer)
                          for b in bunches_list]

        # Buffer with slice data
        n_bunches = len(slice_set_list)
        n_slices = self.slicer.n_slices
        stride = 2 + 2*n_slices
        slice_data_buffer = np.zeros((2 + 2*n_slices)*n_bunches)

        for i, b in enumerate(slice_set_list):
            slice_data_buffer[i*stride] = int(self.comm.rank)  # b.age
            slice_data_buffer[i*stride + 1] = b.beta
            slice_data_buffer[
                i*stride + 2:i*stride + 2 + n_slices] = [
                b.convert_to_time(b.z_centers)][0]
            slice_data_buffer[
                i*stride + 2 + n_slices:i*stride + 2 + 2*n_slices] = [
                b.n_macroparticles_per_slice*getattr(b, moments)
                if moments is not None else
                    b.n_macroparticles_per_slice + int(self.comm.rank)*10][0]

        return slice_data_buffer


# WORKOUT
# =======
import pickle, time
from pprint import pprint


working_sets = {
    'Cable Lat Pulldowns': [(130, 145, 145), 'lb', 10],
    'Hyperextensions': [(10, 10, 10), 'kg', 15],
    'Side Crunches': [(0, 0, 0), 'kg', 20],
    'Hammer Curls': [(16, 16, 16), 'kg', 10],
    'Hanging Leg Raises': [(0, 0, 0), 'kg', 10],
    'Killer Curls': [(55, 55, 55), 'lb', 'x']
    }
back_days = {
    '2016-08-31': working_sets
}

working_sets = {
    'Bench Press': [(60, 60, 60), 'kg', 10],
    'Squats': [(40, 40, 40), 'kg', 10],
    'Butterflies': [(190, 190, 190), 'lb', 10],
    'Hammer Curls': [(16, 16, 16), 'kg', 10],
    'Triceps Press': [(150, 150, 150), 'lb', 10],
    'Cable Curls': [(100, 100, 100), 'lb', 10]}
chest_days = {
    '2016-08-28': working_sets
    }

picklename = 'workout_journal_{:s}.pkl'.format(
    time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
pickle.dump((back_days, chest_days), open(picklename, 'wb'))

test = pickle.load(open(picklename, 'rb'))
pprint(test)
