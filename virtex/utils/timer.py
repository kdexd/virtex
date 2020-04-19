import time
from typing import Optional


class Timer(object):
    r"""
    A simple timer to record time per iteration and ETA of training. ETA is
    estimated by moving window average with fixed window size.

    Parameters
    ----------
    start_from: int, optional (default = 1)
        Iteration from which counting should be started/resumed.
    total_iterations: int, optional (default = None)
        Total number of iterations. ETA will not be tracked (will remain "N/A")
        if this is not provided.
    window_size: int, optional (default = 20)
        Window size for calculating ETA based on average of past few iterations.
    """

    def __init__(
        self,
        start_from: int = 1,
        total_iterations: Optional[int] = None,
        window_size: int = 20,
    ):
        # We decrement by 1 because `current_iter` changes increment during
        # an iteration (for example, will change from 0 -> 1 on iteration 1).
        self.current_iter = start_from - 1
        self.total_iters = total_iterations

        self._start_time = time.time()
        self._times = [0.0] * window_size

    def tic(self) -> None:
        r"""Start recording time: call at the beginning of iteration."""
        self._start_time = time.time()

    def toc(self) -> None:
        r"""Stop recording time: call at the end of iteration."""
        self._times.append(time.time() - self._start_time)
        self._times = self._times[1:]
        self.current_iter += 1

    @property
    def stats(self) -> str:
        r"""Return a single string with current iteration, time and ETA."""
        return (
            f"Iter {self.current_iter} | Time: {self._times[-1]:.3f} sec | "
            f"ETA: {self.eta_hhmm}"
        )

    @property
    def eta_hhmm(self) -> str:
        r"""Return ETA in the form of ``hh mm`` string."""
        if self.total_iters:
            eta_sec = int(self.eta_sec)
            return f"{eta_sec // 3600}h {((eta_sec % 3600) // 60):02d}m"
        else:
            return "N/A"

    @property
    def eta_sec(self) -> float:
        r"""Return ETA in the form of seconds."""
        if self.total_iters:
            avg_time = sum(self._times) / len(self._times)
            return avg_time * (self.total_iters - self.current_iter)
        else:
            return 0.0
