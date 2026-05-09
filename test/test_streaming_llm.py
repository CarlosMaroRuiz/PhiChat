import sys
import os
import unittest
import asyncio
from PhiChat import ChatPhi

class TestStreamingLLM(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.llm = ChatPhi(model="phi4", temperature=0)

    async def test_astream_text(self):
        chunks = []
        async for chunk in self.llm.astream("Dí hola"):
            chunks.append(chunk)
        
        self.assertTrue(len(chunks) > 0)
        full_text = "".join([c.content for c in chunks])
        self.assertIn("HOLA", full_text.upper())

    async def test_astream_tools_filter(self):
        def get_time():
            return "12:00"
        
        bound = self.llm.bind_tools([get_time])
        chunks = []
        async for chunk in bound.astream("¿Qué hora es?"):
            chunks.append(chunk)
            if chunk.content:
                self.assertNotIn("<|tool_call|>", chunk.content)
        
        has_tool_chunks = any(len(c.tool_call_chunks) > 0 for c in chunks)
        self.assertTrue(has_tool_chunks)

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    unittest.main()
