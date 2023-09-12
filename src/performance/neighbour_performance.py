import sys
import os
import statistics
from timeit import default_timer as timer

PARENT_PATH = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(PARENT_PATH)

DEPTH = 99999999

from geo_dome.geodesic_dome import GeodesicDome


def main():
    """
    Main function to run performance tests for each tessellation algorithm created
    """

    # opening CSV file to output results to
    fp = open(f"{PARENT_PATH}/results/time/neighbour_search.csv", "w")
    # adding CSV headers for each column
    fp.write("Frequency,AveTime,MedianTime\n")

    # Performance metric variables
    frequency = 10
    iterations = 20
    timeout = 15

    # Timeout flags for each tessellation algorithm
    search_timeout = False

    gd = GeodesicDome()
    gd.find_neighbours(0, DEPTH)

    for i in range(frequency + 1):
        times = []

        if search_timeout == False:
            search_times = []
            for _ in range(iterations):
                start = timer()
                gd.find_neighbours(0, DEPTH)
                end = timer()
                time_elapsed = end - start
                if time_elapsed > timeout:
                    search_timeout = True
                    times.append(0)
                    times.append(0)
                    break
                search_times.append(time_elapsed)

            if search_timeout == False:
                ave = statistics.mean(search_times)
                median = statistics.median(search_times)
                times.append(ave)
                times.append(median)

        # Writing statistics to CSV file
        result = ",".join(map(str, times))
        fp.write(f"{i},{result}\n")

        gd.tessellate()
    fp.close()


if __name__ == "__main__":
    main()
