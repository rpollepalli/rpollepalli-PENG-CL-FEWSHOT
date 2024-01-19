"""
This file will contain test cases for the automatic evaluation of your
solution in main/lab.py. You should not modify the code in this file. You should
also manually test your solution by running main/app.py.
"""
import unittest

from langchain_core.outputs import LLMResult

from src.utilities.llm_testing_util import llm_connection_check, llm_wakeup
from src.main.lab import llm


class TestLLMResponse(unittest.TestCase):
    """
            This function is a sanity check for the Language Learning Model (LLM) connection.
            It attempts to generate a response from the LLM. If a 'Bad Gateway' error is encountered,
            it initiates the LLM wake-up process. This function is critical for ensuring the LLM is
            operational before running tests and should not be modified without understanding the
            implications.
            Raises:
                Exception: If any error other than 'Bad Gateway' is encountered, it is raised to the caller.
            """

    def test_llm_sanity_check(self):
        try:
            response = llm_connection_check()
            self.assertIsInstance(response, LLMResult)
        except Exception as e:
            if 'Bad Gateway' in str(e):
                llm_wakeup()
                self.fail("LLM is not awake. Please try again in 3-5 minutes.")

    """
    Your prompt should make the LLM correctly generate the response
    "the car is red" for the input "red, car".
    """

    def test_fewshot_1(self):
        result = llm("red, car").lower()
        self.assertIn("the car is red", result)

    """
    Your prompt should make the LLM correctly generate the response
    "the sky is blue" for the input "blue, sky".
    """

    def test_fewshot_2(self):
        result = llm("blue, sky").lower()
        self.assertIn("the sky is blue", result)

    """
    Your prompt should make the LLM correctly generate the response
    "the banana is yellow" for the input "yellow, banana".
    """

    def test_fewshot_3(self):
        result = llm("yellow, banana").lower()
        self.assertIn("the banana is yellow", result)

if __name__ == '__main__':
    unittest.main()
