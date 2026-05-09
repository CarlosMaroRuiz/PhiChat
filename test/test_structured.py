import unittest
from pydantic import BaseModel, Field
from PhiChat import ChatPhi
from langchain_core.messages import AIMessage

class Persona(BaseModel):
    nombre: str
    edad: int

class TestStructuredOutput(unittest.TestCase):
    def test_json_normalization_logic(self):
        """
        Prueba la logica de limpieza de contenido residual en ChatPhi._normalize_static.
        Esto asegura que si el modelo escribe texto y luego el JSON, el content quede limpio.
        """
        llm = ChatPhi()
        content = "Aqui tienes el objeto:\n<|tool_call|>[{\"name\": \"test\", \"args\": {}}]<|/tool_call|>\nEspero que te sirva."
        msg = AIMessage(content=content)
        
        # Accedemos al metodo estatico para validar la limpieza
        normalized = llm._normalize_static(msg)
        
        self.assertEqual(normalized.content, "Aqui tienes el objeto:\n\nEspero que te sirva.")
        self.assertEqual(len(normalized.tool_calls), 1)

    def test_key_mapping_robustness(self):
        """
        Prueba que el parser reconozca diferentes variantes de llaves de argumentos.
        """
        from PhiChat.parsers import parse_phi4_tool_calls
        
        # Variantes comunes que los modelos generan
        variants = [
            '{"name": "t1", "arguments": {"a": 1}}',
            '{"name": "t1", "args": {"a": 1}}',
            '{"name": "t1", "parameters": {"a": 1}}',
            '{"type": "t1", "args": {"a": 1}}', # Caso de phi-3.5
        ]
        
        for v in variants:
            msg = AIMessage(content=f"<|tool_call|>[{v}]<|/tool_call|>")
            calls = parse_phi4_tool_calls(msg)
            self.assertEqual(len(calls), 1, f"Fallo con variante: {v}")
            self.assertEqual(calls[0]["name"], "t1")
            self.assertEqual(calls[0]["args"]["a"], 1)

if __name__ == "__main__":
    unittest.main()
