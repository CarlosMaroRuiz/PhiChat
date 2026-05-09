# Changelog

Todas las novedades y cambios notables del proyecto PhiChat se documentarán en este archivo.

El formato se basa en [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), y este proyecto se adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-09

### Añadido
- **Soporte Async Nativo**: Se ha exportado y estabilizado `arun_tool_loop` para flujos de agentes asíncronos.
- **Normalización de Herramientas**: Nueva lógica en `parsers.py` para manejar automáticamente diferentes formatos de argumentos (`args`, `arguments`, `parameters`).
- **Parsing Robustos**: Soporte para parsear argumentos de herramientas que llegan como strings JSON planos (común en Phi-4).

### Cambiado
- **Arquitectura Modular**: Gran refactorización del paquete. El código ahora está organizado en sub-módulos profesionales: `models/`, `tools/`, `utils/` y `constants/`.
- **Mejora de Streaming**: Se migraron los métodos de streaming a implementaciones privadas (`_astream`, `_stream`) para evitar la duplicación de tokens ("eco") en la salida.
- **API Pública**: El `__init__.py` raíz ahora actúa como una fachada limpia, ocultando la complejidad interna de la librería.

### Corregido
- **Token Duplication**: Se eliminó el error que causaba que cada palabra se imprimiera dos veces durante el streaming.
- **AttributeError en Streaming**: Corregido el acceso a propiedades en `ChatGenerationChunk` durante el flujo de datos asíncrono.
- **Detección de Herramientas**: Mejora en los patrones de regex para detectar bloques de herramientas inyectados en medio de la respuesta de texto.

---

## [0.1.7] - 2026-05-08

### Añadido
- Soporte inicial para tool-calling manual con Phi-4.
- Inyección de System Prompt personalizada para forzar formato JSON.
- Implementación de `ChatPhi` como wrapper de Ollama.
