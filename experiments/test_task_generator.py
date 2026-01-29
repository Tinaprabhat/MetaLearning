import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tasks.task_generator import generate_task


X1, y1 = generate_task(seed=42)
X2, y2 = generate_task(seed=43)

print("Same task?", (X1 == X2).all())
print("X shape:", X1.shape)
print("y distribution:", y1.mean())
