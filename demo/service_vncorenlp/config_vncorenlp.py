import os
from vncorenlp import VNCORENLP_SERVER

# Config Host and port to run VnCoreNLP
HOST = "0.0.0.0"
PORT = "5006"

ANNOTATOR = "wseg,pos"
current_path = os.path.dirname(os.path.realpath(__file__))
VNCORENLP_FILE = os.path.join(current_path, r"VnCoreNLP-1.1.1.jar")
