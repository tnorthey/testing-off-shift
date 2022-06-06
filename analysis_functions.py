import numpy as np

# print basic stats for numpy array
def print_np_stats(dset):
  print('length:  %d' % len(dset))
  print('mean:    %f' % np.mean(dset))
  print('median:  %f' % np.median(dset))
  print('minimum: %f' % np.min(dset))
  print('maximum: %f' % np.max(dset))
  print('st. dev: %f' % np.std(dset))
    
def radial_avg(q,data,nbins):
  """
  T. Northey, May 2022
  radial_avg: radial average of data into nbins radial q-bins.
  nbins must be an integer.
  data and q must be numpy arrays of the same size.
  """
  nq = nbins + 1
  q_max = np.max(q)
  q_rad = np.linspace(0,q_max,nq)
  print('Averaging over %d bins...' % nbins)
  I_rad = np.zeros(nbins)
  counts = np.zeros(nbins, dtype=np.uint32)
  for i in range(0, nbins):
    #print('Iteration %d' % i)
    #print('creating bin in q-range: %f - %f' % (q_rad[i], q_rad[i+1]))
    #print(q >= q_rad[i])
    #print(q < q_rad[i+1])
    condition = np.logical_and(q >= q_rad[i], q < q_rad[i+1])
    tmp = np.where(condition, 1, 0)  # 1s and 0s array of matching the condition
    counts[i] = np.sum(tmp)  # count hits that match the condition
    n = counts[i]  # saves a few ms
    if n > 0:
      I_rad[i] = np.sum(np.multiply(data,tmp), axis=None) / n
    else: continue
  return counts,q_rad[0:nbins],I_rad