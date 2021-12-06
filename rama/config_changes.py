import numpy as np

CONFIG_CHANGES = {
    f"mixing_{i}": [{"tasks": {"mixing": [1, i]}}] for i in np.linspace(0, 1, 11)
}
