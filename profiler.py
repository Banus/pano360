"""Analyze the per-call performances of a piece of code."""
import cProfile
import pstats
import io
from pstats import SortKey


def profile(fun):
    """Profile the given function and return a report."""
    prof = cProfile.Profile()

    prof.enable()
    fun()
    prof.disable()

    strm = io.StringIO()
    stats = pstats.Stats(prof, stream=strm).sort_stats(SortKey.CUMULATIVE)
    stats.print_stats()
    print(strm.getvalue())


if __name__ == '__main__':
    pass
