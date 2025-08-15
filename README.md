\# KUMULUS AI Consultant



\*\*Version:\*\* 1.0.0

\*\*Status:\*\* Production-Ready, Pending UAT



\## 1. Introduction



The KUMULUS AI Consultant is a production-grade, enterprise-ready AI system designed for advanced urban geospatial analysis. It functions as an agentic consultant, leveraging a suite of specialized tools, a hybrid retrieval-augmented generation (RAG) system, and a core logic layer for explainability and ethical oversight.



This system is built to be robust, scalable, and maintainable, transitioning from a conceptual prototype to a reliable application ready for real-world deployment. This repository contains the complete, production-ready codebase as defined by the \*\*KUMULUS AI Production Blueprint\*\*.



\## 2. System Architecture



The project follows a modern `src` layout to ensure testability and a clear separation of concerns.



\-   \*\*`config/`\*\*: Manages all application configuration via a type-safe `pydantic-settings` model.

\-   \*\*`data/`\*\*: Contains raw and processed data used by the agent, including the knowledge base for the RAG system.

\-   \*\*`logs/`\*\*: Outputs structured, JSON-formatted application logs for monitoring and debugging.

\-   \*\*`src/kumulus\_consultant/`\*\*: The core, installable Python package containing all application logic.

&nbsp;   -   \*\*`tools.py`\*\*: Production-grade, error-hardened tools for geospatial analysis (e.g., NDVI calculation).

&nbsp;   -   \*\*`rag\_system.py`\*\*: A hybrid RAG system combining semantic (FAISS) and keyword (BM25) search for robust context retrieval.

&nbsp;   -   \*\*`logic.py`\*\*: The "Consultant's Logic" layer, providing Explainable AI (XAI) and ethical guardrails.

&nbsp;   -   \*\*`main.py`\*\*: The main application entry point, featuring a Typer-based CLI.

\-   \*\*`tests/`\*\*: A comprehensive suite of unit tests built with `pytest` and `pytest-mock` to ensure code reliability.



\## 3. Setup and Installation



This project uses \*\*Poetry\*\* for deterministic dependency management.



\### Prerequisites



\-   Python 3.11+

\-   Poetry (see \[official installation guide](https://python-poetry.org/docs/#installation))



\### Installation Steps



1\.  \*\*Clone the repository:\*\*

&nbsp;   ```bash

&nbsp;   git clone <your-repository-url>

&nbsp;   cd kumulus-api

&nbsp;   ```



2\.  \*\*Install dependencies:\*\*

&nbsp;   Poetry will create a virtual environment and install all necessary packages from the `poetry.lock` file.

&nbsp;   ```bash

&nbsp;   poetry install

&nbsp;   ```



3\.  \*\*Configure Environment Variables:\*\*

&nbsp;   Create a `.env` file in the project root by copying the example.

&nbsp;   ```bash

&nbsp;   # For PowerShell:

&nbsp;   Copy-Item .env.example .env



&nbsp;   # For bash/zsh:

&nbsp;   cp .env.example .env

&nbsp;   ```

&nbsp;   Now, edit the `.env` file and add your `OPENAI\_API\_KEY`.



&nbsp;   ```env

&nbsp;   # .env

&nbsp;   OPENAI\_API\_KEY="sk-..."

&nbsp;   ```



\## 4. Running the Application



All commands must be run from within the project's virtual environment.



\### Activate the Virtual Environment



```bash

poetry shell

