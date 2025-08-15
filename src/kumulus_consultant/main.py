# src/kumulus_consultant/main.py
import json
import logging
import sys
from typing import List

import structlog
import typer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent # CORRECTED
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # CORRECTED
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

from kumulus_consultant.config.settings import settings
from kumulus_consultant.logic import (
    apply_ethical_guardrail,
    generate_xai_explanation,
    handle_ambiguity,
)
from kumulus_consultant.rag_system import HybridRAG
from kumulus_consultant.tools import analyze_green_space_ndvi, fetch_sentinel2_data

# --- Functions (setup_logging, run_query, etc. are here) ---

def setup_logging(log_level: int = settings.LOG_LEVEL):
    # (This function is unchanged)
    processors = [
        structlog.stdlib.add_log_level, structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"), structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info, structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]
    structlog.configure(
        processors=processors, logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger, cache_logger_on_first_use=True,
    )
    formatter = structlog.stdlib.ProcessorFormatter(processor=structlog.processors.JSONRenderer())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("fiona").setLevel(logging.WARNING)

def initialize_system():
    log = structlog.get_logger("initialization")
    log.info("Initializing KUMULUS AI system...", environment=settings.ENVIRONMENT, llm_provider=settings.LLM_PROVIDER)

    if settings.LLM_PROVIDER == "Google":
        llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=0,
            api_key=settings.GEMINI_API_KEY.get_secret_value()
        )
        log.info("Google GenAI LLM initialized.", model=settings.GEMINI_MODEL)
    elif settings.LLM_PROVIDER == "Ollama":
        llm = ChatOllama(model=settings.OLLAMA_MODEL, temperature=0)
        log.info("ChatOllama LLM initialized.", model=settings.OLLAMA_MODEL)
    else: # OpenAI
        llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY.get_secret_value())
        log.info("ChatOpenAI LLM initialized.")

    rag_system = HybridRAG()
    try:
        docs = rag_system.load_and_process_docs("data/processed")
        if docs:
            rag_system.load_retrievers(docs)
        else:
            rag_system = None
    except Exception as e:
        rag_system = None
        log.error("Failed to initialize RAG system.", error=str(e))

    tools = [analyze_green_space_ndvi, fetch_sentinel2_data]
    log.info("Geospatial tools loaded.", tool_names=[t.name for t in tools])

    # --- NEW, SIMPLER PROMPT FOR A TOOL-CALLING AGENT ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Your goal is to answer the user's question accurately based on the tools provided."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # --- USE THE CORRECT AGENT CONSTRUCTOR FOR GEMINI ---
    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )
    log.info("AgentExecutor created. System initialization complete.")
    
    return agent_executor, rag_system

def run_query(query: str, agent_executor: AgentExecutor, rag_system: HybridRAG | None):
    # (This function is unchanged)
    log = structlog.get_logger("query_handler")
    log.info("Processing new query.", user_query=query)
    try:
        context = ""
        if rag_system:
            context = rag_system.retrieve(query)
        agent_input = {"input": f"User Query: {query}\n\nContext from documents:\n{context}", "chat_history": []} # Added empty chat_history
        agent_response = agent_executor.invoke(agent_input)
        recommendation = agent_response.get("output", handle_ambiguity())
        
        simulated_findings = {}
        # Try to parse tool output if it exists
        if "tool_outputs" in agent_response:
             try:
                 # A simple way to get findings from the first tool call
                 tool_output = agent_response["tool_outputs"][0]
                 simulated_findings = json.loads(tool_output).get("data", {})
             except (IndexError, json.JSONDecodeError):
                 pass # No valid tool output to parse

        simulated_area_info = {"is_historically_underserved": True}
        xai_explanation = generate_xai_explanation(recommendation, simulated_findings)
        ethical_check = apply_ethical_guardrail(recommendation, simulated_area_info)
        
        typer.echo("\n" + "="*50)
        typer.secho("KUMULUS AI Consultant Response", fg=typer.colors.BRIGHT_BLUE, bold=True)
        typer.echo("="*50)
        typer.secho(f"\nRecommendation:", fg=typer.colors.CYAN)
        typer.echo(f"{recommendation}")
        typer.secho(f"\nJustification:", fg=typer.colors.CYAN)
        typer.echo(f"{xai_explanation}")
        typer.secho(f"\nEthical Check:", fg=typer.colors.CYAN)
        typer.secho(f"{ethical_check['reason']}", fg=typer.colors.YELLOW if ethical_check["flagged"] else typer.colors.GREEN)
        typer.echo("="*50)
    except Exception as e:
        log.error("An unexpected error occurred...", error=str(e), exc_info=True)
        typer.secho(f"\nAn unexpected error occurred: {e}", fg=typer.colors.RED)

def main():
    # (This function is unchanged)
    setup_logging()
    agent_executor, rag_system = initialize_system()
    log = structlog.get_logger("chat_session")
    log.info("Starting interactive chat session.")
    typer.echo("Welcome to the KUMULUS AI Consultant. Type 'exit' to quit.")
    while True:
        query = typer.prompt("\n[User]")
        if query.lower() == "exit":
            typer.echo("Exiting consultant. Goodbye!")
            break
        run_query(query, agent_executor, rag_system)

if __name__ == "__main__":
    typer.run(main)