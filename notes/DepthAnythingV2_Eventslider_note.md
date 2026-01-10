## Problem
I got weird errors in HPC, where it complained about overflow in line 71 of DSEC/utils/evenslicer.py

## What I did
Cast to int64
```        
time_array_conservative = np.asarray(self.events['t'][t_start_ms_idx:t_end_ms_idx]).astype(np.int64)
```