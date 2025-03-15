import numpy as np


def water_erosion_channels(depth_map, num_channels=5, depth=0.2):
    result = depth_map.astype(np.float32)

    for _ in range(num_channels):
        start = np.random.randint(0, depth_map.shape[1])
        path = np.zeros(depth_map.shape, dtype=bool)
        current = start
        for i in range(depth_map.shape[0]):
            path[i, current] = True
            current += np.random.randint(-1, 2)
            current = np.clip(current, 0, depth_map.shape[1] - 1)

        result[path] -= depth * np.max(depth_map)

    return np.clip(result, 0, np.max(depth_map))
