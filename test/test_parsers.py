import unittest
from langchain_core.messages import AIMessage, SystemMessage
from PhiChat.parsers import parse_phi4_tool_calls, inject_tool_system_message

class TestParsers(unittest.TestCase):
    def test_parse_standard_phi4_tool_call(self):
        content = '<|tool_call|>[{"name": "get_weather", "arguments": {"location": "Madrid"}}]<|/tool_call|>'
        msg = AIMessage(content=content)
        calls = parse_phi4_tool_calls(msg)
        
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "get_weather")
        self.assertEqual(calls[0]["args"]["location"], "Madrid")
        self.assertTrue(calls[0]["id"].startswith("call_"))

    def test_parse_multiple_tool_calls(self):
        content = '<|tool_call|>[{"name": "add", "args": {"a": 1}}, {"name": "sub", "args": {"b": 2}}]<|/tool_call|>'
        msg = AIMessage(content=content)
        calls = parse_phi4_tool_calls(msg)
        
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["name"], "add")
        self.assertEqual(calls[1]["name"], "sub")

    def test_parse_legacy_functools_format(self):
        content = 'functools[{"name": "old_tool", "arguments": {}}]'
        msg = AIMessage(content=content)
        calls = parse_phi4_tool_calls(msg)
        
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "old_tool")

    def test_inject_system_message_new(self):
        messages = [AIMessage(content="Hello")]
        suffix = " [TOOL_HINT]"
        result = inject_tool_system_message(messages, suffix)
        
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], SystemMessage)
        self.assertIn(suffix, result[0].content)

    def test_inject_system_message_existing(self):
        messages = [SystemMessage(content="Rules:"), AIMessage(content="Hi")]
        suffix = " [TOOL_HINT]"
        result = inject_tool_system_message(messages, suffix)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].content, "Rules:" + suffix)

if __name__ == "__main__":
    unittest.main()
