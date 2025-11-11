"""

🧱 Part 1: Monitoring Process (Ring Buffer + Socket Server)



"""
# monitor.py
import psutil
import time
import socket
import threading
import pickle
from collections import deque

# Ring buffer to store last 100 samples
cpu_buffer = deque(maxlen=100)
# Lock for thread-safe access
buffer_lock = threading.Lock()


# Background thread to collect CPU usage
def monitor_cpu(interval=1):

    while True:
        cpu = psutil.cpu_percent(interval=interval)
        with buffer_lock:
            cpu_buffer.append(cpu)
            print("cpu: {}".format(cpu))


# Socket server to respond to queries
def socket_server(host='localhost', port=5000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"Server listening on {host}:{port}")
        while True:
            conn, addr = s.accept()
            with conn:
                with buffer_lock:
                    data = pickle.dumps(list(cpu_buffer))
                conn.sendall(data)


# Start both threads
threading.Thread(target=monitor_cpu, daemon=True).start()
threading.Thread(target=socket_server, daemon=True).start()

# Keep main thread alive
while True:
    time.sleep(10)
