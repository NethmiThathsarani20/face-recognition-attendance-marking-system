"""Test runner for all tests in the attendance system.
"""

import os
import sys
import unittest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

# Import test modules
from test_attendance_system import TestAttendanceSystem
from test_cnn_trainer import TestCNNTrainer
from test_face_manager import TestFaceManager
from test_integration import TestIntegration


def run_all_tests():
    """Run all tests in the test suite.
    """
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestFaceManager))
    test_suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestAttendanceSystem),
    )
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestIntegration))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestCNNTrainer))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


def run_specific_test(test_class_name):
    """Run a specific test class.

    Args:
        test_class_name: Name of the test class to run
    """
    test_classes = {
        "face_manager": TestFaceManager,
        "attendance_system": TestAttendanceSystem,
        "integration": TestIntegration,
        "cnn_trainer": TestCNNTrainer,
    }

    if test_class_name in test_classes:
        test_suite = unittest.TestLoader().loadTestsFromTestCase(
            test_classes[test_class_name],
        )
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(test_suite)
        return result.wasSuccessful()
    print(f"Test class '{test_class_name}' not found.")
    print(f"Available test classes: {', '.join(test_classes.keys())}")
    return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        success = run_specific_test(test_name)
    else:
        # Run all tests
        print("Running all tests...")
        success = run_all_tests()

    sys.exit(0 if success else 1)
