import sys
import os
import unittest
from PhiChat import ChatPhi

class TestBasicLLM(unittest.TestCase):
    def setUp(self):
        model = os.getenv("TEST_MODEL", "phi4")
        self.llm = ChatPhi(model=model, temperature=0)

    def test_invoke(self):
        res = self.llm.invoke("Di la palabra 'QUESO' únicamente.")
        self.assertIn("QUESO", res.content.upper())

    def test_identifying_params(self):
        params = self.llm._identifying_params
        self.assertEqual(params["model"], self.llm.model)

if __name__ == "__main__":
    
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    unittest.main()
