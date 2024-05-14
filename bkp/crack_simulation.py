import cv2
import numpy as np
import logging


def simulate_cracks(image, num_cracks_range=(2, 5), max_length=120, crack_types=["hairline", "wide", "branching"]):
    if image is None:
        logging.warning("Received None image in simulate_cracks")
        return None  # Or handle the None case differently based on your logic

    # Ensure the image is in grayscale for simplicity
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Number and types of cracks
    num_cracks = np.random.randint(*num_cracks_range)

    for _ in range(num_cracks):
        crack_type = np.random.choice(crack_types)
        if crack_type == "hairline":
            thickness = 2
        elif crack_type == "wide":
            thickness = np.random.randint(3, 8)
        elif crack_type == "branching":
            thickness = np.random.randint(2, 4)
            # Add branching crack logic here
            # Draw additional lines starting from random points on the main crack

        # Start point for the crack
        x_start, y_start = np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0])

        # Simulate crack propagation
        for _ in range(np.random.randint(1, max_length)):
            # Crack step size and angle
            # length = np.random.uniform(20, 100)  # Adjust as needed
            step_length = np.random.randint(3, 9)
            angle = np.random.uniform(0, 2 * np.pi)
            x_end = int(x_start + step_length * np.cos(angle))
            y_end = int(y_start + step_length * np.sin(angle))

            # Draw the crack segment
            cv2.line(image, (x_start, y_start), (x_end, y_end), (0), thickness)

            # Update start point
            x_start, y_start = x_end, y_end

            # Randomly decide if the crack should branch or stop
            if np.random.rand() < 0.1:  # 10% chance to branch or stop
                if np.random.rand() < 0.5:  # 50% of those 10% to branch
                    # Start a new crack segment from the current point
                    x_start, y_start = x_end, y_end
                else:
                    break  # Stop the crack propagation

    return image