import sys
from contextlib import contextmanager

@contextmanager
def experiment_logger(log_path, also_print=True):
    """
    Redirects stdout and stderr to a log file.
    If also_print=True, keeps printing to console.
    """
    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    with open(log_path, "w", encoding="utf-8") as f:
        if also_print:
            stdout = Tee(sys.stdout, f)
            stderr = Tee(sys.stderr, f)
        else:
            stdout = f
            stderr = f

        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout, stderr

        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

