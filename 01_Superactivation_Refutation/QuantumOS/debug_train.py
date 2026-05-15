import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ui.quantum_os import train_loop, shared_state

print("Starting train_loop synchronously...")
train_loop()
print("Final logs:")
for log in shared_state["logs"]:
    print(log)
