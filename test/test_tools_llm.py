import sys
import os
import unittest
import asyncio
from PhiChat import ChatPhi, create_tool

class TestToolsLLM(unittest.TestCase):
    def setUp(self):
        model = os.getenv("TEST_MODEL", "phi4")
        self.llm = ChatPhi(model=model, temperature=0)

    def test_bind_tools_invocation(self):
        def get_current_stock(ticker: str) -> str:
            return f"El precio de {ticker} es 150.00"
        
        bound = self.llm.bind_tools([get_current_stock])
        
        messages = [
            {"role": "system", "content": "You are a stock bot. ALWAYS use the get_current_stock tool when asked about stock prices."},
            {"role": "user", "content": "¿A cuánto está la acción de AAPL?"}
        ]
        res = bound.invoke(messages)
        
        self.assertTrue(len(res.tool_calls) > 0)
        self.assertEqual(res.tool_calls[0]["name"], "get_current_stock")
        self.assertIn("AAPL", str(res.tool_calls[0]["args"]))

    def test_run_tool_loop(self):
        def add(a: int, b: int) -> int:
            return a + b
        
        res = self.llm.run_tool_loop(
            messages=[{"role": "human", "content": "¿Cuánto es 123 + 456?"}],
            tools=[create_tool(add)],
            verbose=False
        )
        self.assertIn("579", res.content)

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    unittest.main()
