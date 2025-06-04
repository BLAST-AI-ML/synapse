import sys
import io

class DualStream(io.TextIOBase):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    def close(self):
        for f in self.files:
            f.close()

def setup_output(log_file_path):
    log_file = open(log_file_path, "w")
    dual_stdout = DualStream(sys.stdout, log_file)
    dual_stderr = DualStream(sys.stderr, log_file)
    sys.stdout = dual_stdout
    sys.stderr = dual_stderr
