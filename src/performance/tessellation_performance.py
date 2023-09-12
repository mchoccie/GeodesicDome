import sys
import os
import statistics
from timeit import default_timer as timer

PARENT_PATH = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(PARENT_PATH)

from geo_dome.geodesic_dome import GeodesicDome
from archived_versions.list_version import list_icosphere as lv
from archived_versions.set_version import set_icosphere as sv


def main():
    """
    Main function to run performance tests for each tessellation algorithm created
    """

    # opening CSV file to output results to
    fp = open(f"{PARENT_PATH}/results/time/tessellation.csv", "w")
    # adding CSV headers for each column
    fp.write("Frequency,NumbaAve,NumbaMedian,ListAve,ListMedian,SetAve,SetMedian\n")

    # Performance metric variables
    frequency = 10
    iterations = 20
    timeout = 15

    # Timeout flags for each tessellation algorithm
    numba_timeout = False
    list_timeout = False
    set_timeout = False

    GeodesicDome()

    for i in range(frequency + 1):
        times = []

        # running Numba tessellation algorithm and adding results to times list
        if numba_timeout == False:
            iter_times = []
            for _ in range(iterations):
                start = timer()
                GeodesicDome(i)
                end = timer()
                time_elapsed = end - start

                # timeout reached
                if time_elapsed > timeout:
                    numba_timeout = True
                    times.append(0)
                    times.append(0)
                    break
                iter_times.append(end - start)
            if numba_timeout == False:
                # calculating statistics for the iterations
                numba_ave = statistics.mean(iter_times)
                numba_median = statistics.median(iter_times)
                times.append(numba_ave)
                times.append(numba_median)
        else:
            times.append(0)
            times.append(0)

        # running list tessellation algorithm and adding results to times list
        if list_timeout == False:
            iter_times = []
            for _ in range(iterations):
                start = timer()
                lv.create_icosphere(i)
                end = timer()
                time_elapsed = end - start

                # timeout reached
                if time_elapsed > timeout:
                    list_timeout = True
                    times.append(0)
                    times.append(0)
                    break
                iter_times.append(end - start)
            if list_timeout == False:
                # calculating statistics for the iterations
                list_ave = statistics.mean(iter_times)
                list_median = statistics.median(iter_times)
                times.append(list_ave)
                times.append(list_median)
        else:
            times.append(0)
            times.append(0)

        # running set/dict tessellation algorithm and adding results to times list
        if set_timeout == False:
            iter_times = []
            for _ in range(iterations):
                start = timer()
                sv.create_icosphere(i)
                end = timer()
                time_elapsed = end - start

                # timeout reached
                if time_elapsed > timeout:
                    set_timeout = True
                    times.append(0)
                    times.append(0)
                    break
                iter_times.append(end - start)
            if set_timeout == False:
                # calculating statistics for the iterations
                set_ave = statistics.mean(iter_times)
                set_median = statistics.median(iter_times)
                times.append(set_ave)
                times.append(set_median)
        else:
            times.append(0)
            times.append(0)

        # Writing statistics to CSV file
        result = ",".join(map(str, times))
        fp.write(f"{i},{result}\n")
    fp.close()


if __name__ == "__main__":
    main()
