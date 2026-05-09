import sys
import os
import unittest
import asyncio
from PhiChat import ChatPhi
from langchain_core.tools import tool

class TestMCPMock(unittest.TestCase):
    def setUp(self):
        self.llm = ChatPhi(model="phi4", temperature=0)

    def test_async_tool_execution(self):
        @tool
        async def async_calculator(expr: str) -> str:
            """Calcula una expresion matematica."""
            await asyncio.sleep(0.1)
            return str(eval(expr))

        res = self.llm.run_tool_loop(
            messages=[{"role": "user", "content": "¿Cuánto es 2**10?"}],
            tools=[async_calculator],
            verbose=True
        )
        self.assertIn("1024", res.content)

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    unittest.main()
