import numpy as np
import os

if os.path.isfile("./memory"):
    memory = np.load("./memory", allow_pickle=True)
    print("range", np.max(memory) - np.min(memory))
    print("mean", np.mean(memory))
    print("var", np.var(memory))