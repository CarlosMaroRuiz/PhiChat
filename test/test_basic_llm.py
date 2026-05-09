import sys
import os
import unittest
from PhiChat import ChatPhi

class TestBasicLLM(unittest.TestCase):
    def setUp(self):
        self.llm = ChatPhi(model="phi4", temperature=0)

    def test_invoke(self):
        res = self.llm.invoke("Di la palabra 'QUESO' únicamente.")
        self.assertIn("QUESO", res.content.upper())

    def test_identifying_params(self):
        params = self.llm._identifying_params
        self.assertEqual(params["model"], "phi4")

if __name__ == "__main__":
    
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    unittest.main()
