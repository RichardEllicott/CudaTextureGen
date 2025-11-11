"""

🧱 Part 2: Query Client

"""

# client.py
import socket
import pickle

def query_cpu_buffer(host='localhost', port=5000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        data = b''
        while True:
            packet = s.recv(4096)
            if not packet:
                break
            data += packet
        cpu_samples = pickle.loads(data)
        print("Recent CPU samples:", cpu_samples)

query_cpu_buffer()
