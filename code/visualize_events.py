import hdf5plugin
import h5py
import matplotlib.pyplot as plt
import os
from DSEC.scripts.utils.eventslicer import EventSlicer

def visualize_events(events, title=None):
    plt.figure(figsize=(10, 6))
    plt.scatter(events['x'], events['y'], c=events['p'], cmap='coolwarm', s=1, alpha=0.7)
    plt.gca().invert_yaxis()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title or 'Event Frame')
    plt.show()

def main():
    h5_path = os.path.join('data', 'train', 'zurich_city_00_a_events_left', 'events.h5')
    with h5py.File(h5_path, 'r') as h5f:
        slicer = EventSlicer(h5f)
        # Get time window
        t_start_us = slicer.get_start_time_us() + 10_000_000
        t_end_us = t_start_us + 10_000
        events = slicer.get_events(t_start_us, t_end_us)
        if events:
            visualize_events(events, title=f'Events from {t_start_us} to {t_end_us} us')
        else:
            print('No events found in the specified time window.')

if __name__ == "__main__":
    main()
