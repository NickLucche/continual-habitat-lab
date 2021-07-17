import sys, os
def suppress_habitat_logging():
    os.environ["GLOG_minloglevel"] = "3"
    os.environ["MAGNUM_LOG"] = "quiet"

def enable_habitat_logging(log_level: int):
    os.environ["GLOG_minloglevel"] = str(log_level)
    os.environ.pop("MAGNUM_LOG")
