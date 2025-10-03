# Articulos de control de informaicon a implementar a futuro:

- Retrieval-Augmented Generation (RAG): El modelo busca en una base de conocimiento externa y responde solo en función de lo recuperado.

- Table-to-Text Generation: Modelos que generan texto descriptivo a partir de tablas estructuradas.

- Fact-grounded Question Answering: Métodos para que el modelo no alucine, obligándolo a citar o basarse solo en la información de entrada.

- Knowledge-grounded Dialogue Systems: Sistemas conversacionales que usan una base de datos, ontología o diccionario como única fuente de verdad.

Algunos artículos clave que abordan estos temas incluyen:

- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
(Lewis et al., NeurIPS 2020) → Propone RAG, un modelo que combina búsqueda de documentos con generación de respuestas, restringiendo la salida a la información recuperada.

- "TabFact: A Large-scale Dataset for Table-based Fact Verification"
(Chen et al., ICLR 2020) → Se centra en verificar afirmaciones usando exclusivamente la información contenida en tablas.

- "Faithful to the Document or to the World? Mitigating Hallucinations via Effective Grounding"
(Cao et al., ACL 2022) → Aborda cómo hacer que los modelos basen sus respuestas estrictamente en el documento dado.

- "Knowledge-Grounded Conversational Agents"
(Dinan et al., ICLR 2019) → Explora sistemas de diálogo en los que las respuestas deben generarse solo a partir de un conjunto de documentos proporcionados.


## MCP: Model Context Protocol

Model Context Protocol (MCP) es un enfoque para controlar y limitar el contexto que un modelo de lenguaje utiliza al generar respuestas. La idea es definir 
explícitamente qué información puede considerar el modelo, restringiendo su conocimiento a un conjunto específico de datos o documentos. Esto ayuda a reducir la generación de información no verificada o "alucinaciones".

## ¿Qué es MCP?

 - MCP (“Model Context Protocol”) es un protocolo abierto estandarizado introducido por Anthropic en noviembre de 2024.
 - Su objetivo principal es estandarizar la forma en que las aplicaciones con LLMs obtienen contexto externo (datos, herramientas, funciones) de forma segura, organizada y modular. 
 - En vez de que cada LLM/application tenga que escribir integraciones a medida para cada fuente externa (archivos, APIs, bases de datos, etc.), MCP permite definir “servidores MCP” que exponen herramientas, datos o funciones, 
y “clientes MCP” que las llaman, todo usando un protocolo unificado.

## Arquitectura y componentes

1. **MCP Host**
 - La aplicación o entorno que usa al LLM. Puede ser un asistente, una interfaz de chat, etc. 

2. **MCP Client**
 - Componente dentro del host que hace de intermediario con los servidores MCP. Traduce las peticiones, maneja la negociación de capacidades, etc. 

3. **MCP Server**
 - Aquí es donde se encuentran las fuentes externas: archivos, bases de datos, APIs, funciones especializadas, etc. Exponen lo que el cliente pueda usar (lectura de datos, ejecución de funciones, etc.). 

4. **Comunicación / Transporte**
 - El protocolo base usa JSON-RPC 2.0 para estructurar mensajes (requests/responses, notificaciones).
 - Puede usarse sobre distintos medios según la localización del servidor MCP: local (stdin/stdout) o remoto (HTTP, SSE — Server-Sent Events). 

Articulo:
@misc{hou2025modelcontextprotocolmcp,
      title={Model Context Protocol (MCP): Landscape, Security Threats, and Future Research Directions}, 
      author={Xinyi Hou and Yanjie Zhao and Shenao Wang and Haoyu Wang},
      year={2025},
      eprint={2503.23278},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2503.23278}, 
}