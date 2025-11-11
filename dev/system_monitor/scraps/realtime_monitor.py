#!/usr/bin/env python3
"""

pip install psutil
pip install pywin32
pip install wmi


"""


import psutil
import time
import logging

# Setup logging
logging.basicConfig(filename='cpu_monitor.log', level=logging.INFO, format='%(asctime)s - CPU: %(message)s')

def monitor_cpu(interval=1):
    while True:
        cpu = psutil.cpu_percent(interval=interval)
        logging.info(f"{cpu}%")

        print(f"CPU usage: {cpu}%")

        # # Optional: take action if CPU is too high
        # if cpu > 90:
        #     print(f"High CPU usage detected: {cpu}%")




if __name__ == "__main__":
    monitor_cpu()
