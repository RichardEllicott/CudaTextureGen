#!/usr/bin/env python3
"""

pip install psutil
pip install pywin32
pip install wmi


"""


def get_cpu_usage():

    import psutil
    import time

    # Prime the CPU counters
    for proc in psutil.process_iter():
        try:
            proc.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    time.sleep(1)  # Wait a bit to let CPU usage accumulate

    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
        try:
            if proc.info['cpu_percent'] > 0.0:
                print(f"{proc.info['pid']:>6} {proc.info['name']:<25} "
                      f"CPU: {proc.info['cpu_percent']:>5.1f}%  "
                      f"RAM: {proc.info['memory_info'].rss / (1024*1024):.1f} MB")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def list_services():
    import wmi

    c = wmi.WMI()
    for service in c.Win32_Service():
        # print(service)
        print(f"{service.Name}: {service.State} - StartMode: {service.StartMode}")


list_services()