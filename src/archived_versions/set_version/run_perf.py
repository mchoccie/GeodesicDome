import sys
import os
import statistics
from timeit import default_timer as timer
from set_icosphere import create_icosphere


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(ROOT_DIR, "../../results/time/set/")

# https://www.geeksforgeeks.org/monitoring-memory-usage-of-a-running-python-program/


def wipe_time_results():
    """Function to remove old performance logs"""

    for filename in os.listdir(path):
        open(path + filename, "w").close()


def write_result(start: float, end: float, freq: int) -> None:
    """Function to write time performance to logs

    Args:
        start (float): start time of iteration
        end (float): end time of iteration
        freq (int): tessellation frequency
    """
    f = open(f"{path}times{freq}.txt", "a")
    f.write(str(end - start) + "\n")
    f.close()


def write_stats(freq: int, ave_msg: str, median_msg: str) -> None:
    """Function to write performance stats to logs

    Args:
        freq (int): tessellation frequency
        ave_msg (str): average time for specified frequency
        median_msg (str): median time for specified frequency
    """
    f = open(f"{path}times{freq}.txt", "a")
    f.write(ave_msg + "\n")
    f.write(median_msg + "\n")
    f.close()


def main():
    max_freq = 0
    max_iter = 0
    try:
        max_freq = int(input("Enter maximum frequency: "))
        max_iter = int(input("Enter number of iterations: "))
    except:
        sys.exit("Invalid input!")

    if max_freq < 0 or max_iter < 0:
        sys.exit("Invalid input!")

    wipe_time_results()
    create_icosphere(0)

    for i in range(max_freq + 1):
        vals = []
        for j in range(max_iter):
            start = timer()
            create_icosphere(i)
            end = timer()
            write_result(start, end, i)
            vals.append(end - start)

        ave = statistics.mean(vals)
        median = statistics.median(vals)
        ave_msg = f"Average time for {i} frequency with {max_iter} repeats: {ave}"
        median_msg = f"Median time for {i} frequency with {max_iter} repeats: {median}"
        write_stats(i, ave_msg, median_msg)


if __name__ == "__main__":
    main()
