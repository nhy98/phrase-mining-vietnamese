import subprocess
from config_vncorenlp import VNCORENLP_SERVER, VNCORENLP_FILE, HOST, PORT, ANNOTATOR
import argparse
import time

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--background", type=bool, default=False, help="Model name must be siamese or triplet")
parameter = parser.parse_args()

args = {'args': ['java', '-Xmx2g', '-jar', VNCORENLP_SERVER, VNCORENLP_FILE, '-i', HOST, '-p', PORT, '-a', ANNOTATOR]}


if parameter.background:
    args['stdout'] = subprocess.DEVNULL
    args['stderr'] = subprocess.DEVNULL

    process = subprocess.Popen(**args)
    time.sleep(5)
    print(f"Started VNcoreNLP service at http://{HOST}:{PORT}")
    print(f"VnCoreNLP process id: {process.pid}")
else:
    print(f"Started VNcoreNLP service at http://{HOST}:{PORT}")
    subprocess.run(**args)
