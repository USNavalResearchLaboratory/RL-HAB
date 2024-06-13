import numpy as np
import math
import unittest


def calculate_relative_angle(x, y, goal_x, goal_y, heading_x, heading_y):
    # Calculate the current heading based on the heading vector
    heading = np.arctan2(heading_y, heading_x)

    # Calculate the true (inverted) bearing FROM POSITION TO GOAL
    true_bearing = np.arctan2(goal_y-y, goal_x-x)

    # Find the absolute difference between the two angles
    rel_bearing = np.abs(heading - true_bearing)

    # map from [-pi, pi] to [0,pi]
    rel_bearing = abs((rel_bearing + np.pi) % (2 * np.pi) - np.pi)

    # Alternate way to map from [-pi, pi] to [0,pi]
    # If the absolute difference is greater than π, subtract it from 2π
    #if rel_bearing > np.pi:
    #    rel_bearing = 2 * np.pi - rel_bearing

    return rel_bearing


class TestCalculateRelativeAngle(unittest.TestCase):
    def test_case_1(self):
        goal_x, goial_y = 0, 0
        x, y = -3, 3  # Position
        heading_x, heading_y = 1, 0  # Heading east
        expected_result = np.pi/4   # Manually calculated
        result = calculate_relative_angle(x, y, goal_x, goial_y, heading_x, heading_y)
        self.assertAlmostEqual(result, expected_result, places=2)

    def test_case_2(self):
        goal_x, goial_y = 0, 0
        x, y = -3, 3  # Position
        heading_x, heading_y = 0, -1  # Heading south
        expected_result = np.pi/4   # Manually calculated
        result = calculate_relative_angle(x, y, goal_x, goial_y, heading_x, heading_y)
        self.assertAlmostEqual(result, expected_result, places=2)

    def test_case_3(self):
        goal_x, goial_y = 0, 0
        x, y = -3, 3  # Position
        heading_x, heading_y = 0, 1  # Heading north
        expected_result = 3*np.pi/4   # Manually calculated
        result = calculate_relative_angle(x, y, goal_x, goial_y, heading_x, heading_y)
        self.assertAlmostEqual(result, expected_result, places=2)

    def test_case_4(self):
        goal_x, goial_y = 0, 0
        x, y = -3, 3  # Position
        heading_x, heading_y = -1, 0  # Heading west
        expected_result = 3*np.pi/4  # Manually calculated
        result = calculate_relative_angle(x, y, goal_x, goial_y, heading_x, heading_y)
        self.assertAlmostEqual(result, expected_result, places=2)

    def test_case_5(self):
        goal_x, goial_y = 0, 0
        x, y = 0, 1  # Position
        heading_x, heading_y = 0, 1  # Heading north
        expected_result = np.pi  # Manually calculated
        result = calculate_relative_angle(x, y, goal_x, goial_y, heading_x, heading_y)
        self.assertAlmostEqual(result, expected_result, places=2)

    def test_case_6(self):
        goal_x, goial_y = 0, 0
        x, y = 0, 1  # Position
        heading_x, heading_y = 0, -1  # Heading west
        expected_result = 0  # Manually calculated
        result = calculate_relative_angle(x, y, goal_x, goial_y, heading_x, heading_y)
        self.assertAlmostEqual(result, expected_result, places=2)

    def test_case_7(self):
        goal_x, goial_y = 0, 0
        x, y = 0, 1  # Position
        heading_x, heading_y = 1, 0  # Heading west
        expected_result = np.pi/2  # Manually calculated
        result = calculate_relative_angle(x, y, goal_x, goial_y, heading_x, heading_y)
        self.assertAlmostEqual(result, expected_result, places=2)

    def test_case_8(self):
        goal_x, goial_y = 0, 0
        x, y = 0, 1  # Position
        heading_x, heading_y = -1, 0  # Heading west
        expected_result = np.pi/2  # Manually calculated
        result = calculate_relative_angle(x, y, goal_x, goial_y, heading_x, heading_y)
        self.assertAlmostEqual(result, expected_result, places=2)

    def test_case_9(self):
        goal_x, goial_y = 0, 0
        x, y = 0, -1  # Position
        heading_x, heading_y = 0, 1  # Heading North
        expected_result = 0  # Manually calculated
        result = calculate_relative_angle(x, y, goal_x, goial_y, heading_x, heading_y)
        self.assertAlmostEqual(result, expected_result, places=2)

    def test_case_10(self):
        goal_x, goial_y = 250, 250
        x, y = 300, 200  # Position
        heading_x, heading_y = 1, 0  # Heading East
        expected_result = 3*np.pi/4  # Manually calculated
        result = calculate_relative_angle(x, y, goal_x, goial_y, heading_x, heading_y)
        self.assertAlmostEqual(result, expected_result, places=2)


if __name__ == '__main__':
    unittest.main()