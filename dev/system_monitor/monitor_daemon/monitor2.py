"""

test with theta

"""
import psutil
import time
import threading
import math
from collections import deque

cpu_buffer = deque(maxlen=100)
buffer_lock = threading.Lock()
cpu_moving_average = 0.0
last_time = time.time()
theta = 1.0 / 32.0  # decay rate in 1/sec (adjust for responsiveness)

def lerp(a, b, t):
    return a + (b - a) * t

def monitor_cpu():
    global cpu_moving_average, last_time
    while True:
        now = time.time()
        dt = now - last_time
        last_time = now

        cpu = psutil.cpu_percent(interval=None)  # non-blocking
        alpha = 1 - math.exp(-theta * dt)

        with buffer_lock:
            cpu_buffer.append(cpu)
            cpu_moving_average = lerp(cpu_moving_average, cpu, alpha)
            print(f"cpu: {cpu:.1f}%  cpu_ma: {cpu_moving_average:.1f}%")

        time.sleep(0.125)  # sampling interval





threading.Thread(target=monitor_cpu, daemon=True).start()

# Keep main thread alive
while True:
    time.sleep(10)
