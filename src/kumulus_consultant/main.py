# src/kumulus_consultant/main.py
import dotenv
# Load environment variables from .env file BEFORE any other application imports.
# This ensures that all modules have access to the environment variables when they are initialized.
dotenv.load_dotenv()

import json
import logging
import sys
from typing import List, Any, Dict

import structlog
import typer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

from kumulus_consultant.config.settings import settings
from kumulus_consultant.logic import (
    apply_ethical_guardrail,
    generate_xai_explanation,
    handle_ambiguity,
)
from kumulus_consultant.rag_system import HybridRAG
from kumulus_consultant.tools import (
    analyze_green_space_ndvi,
    fetch_sentinel2_data,
    detect_informal_settlements,
)

# ... (The rest of the file remains exactly the same as the previous version) ...

def setup_logging(log_level: int = settings.LOG_LEVEL):
    """
    Configures structured logging using structlog.
    """
    processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]
    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer()
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)
    # Silence noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("fiona").setLevel(logging.WARNING)

def initialize_system():
    """
    Initializes and assembles all core components of the KUMULUS AI system.
    """
    log = structlog.get_logger("initialization")
    log.info(
        "Initializing KUMULUS AI system...",
        environment=settings.ENVIRONMENT,
        llm_provider=settings.LLM_PROVIDER
    )

    # --- 1. Initialize the LLM based on configuration ---
    if settings.LLM_PROVIDER == "Google":
        llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=0,
            # No need to pass api_key here, LangChain will find GOOGLE_API_KEY from the env
        )
        log.info("Google GenAI LLM initialized.", model=settings.GEMINI_MODEL)
    elif settings.LLM_PROVIDER == "Ollama":
        llm = ChatOllama(model=settings.OLLAMA_MODEL, temperature=0)
        log.info("ChatOllama LLM initialized.", model=settings.OLLAMA_MODEL)
    else: # OpenAI
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=settings.OPENAI_API_KEY.get_secret_value()
        )
        log.info("ChatOpenAI LLM initialized.")

    # --- 2. Set up the RAG System ---
    rag_system = HybridRAG()
    try:
        docs = rag_system.load_and_process_docs("data/processed")
        if docs:
            rag_system.load_retrievers(docs)
            log.info("Hybrid RAG system initialized successfully.")
        else:
            rag_system = None
            log.warning("No documents found for RAG system. Continuing without RAG.")
    except Exception as e:
        rag_system = None
        log.error("Failed to initialize RAG system. It will be disabled.", error=str(e))

    # --- 3. Load All Production-Ready Tools ---
    tools = [
        fetch_sentinel2_data,
        analyze_green_space_ndvi,
        detect_informal_settlements
    ]
    log.info("Geospatial tools loaded.", tool_names=[t.name for t in tools])

    # --- 4. Create the Agent Prompt and Executor ---
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a professional GeoAI consultant named KUMULUS. Your primary goal is to provide actionable, data-driven recommendations based on geospatial analysis. "
            "First, use the provided context to understand the user's query and any relevant background information. "
            "Then, use your available tools in a logical sequence to perform the required analysis. "
            "For example, you must first use 'fetch_sentinel2_data' to get an image before you can use 'analyze_green_space_ndvi' or 'detect_informal_settlements'. "
            "Synthesize the results from your tools into a clear, concise final recommendation for the user. Explain your reasoning step-by-step."
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )
    log.info("AgentExecutor created. System initialization complete.")
    
    return agent_executor, rag_system

def _extract_data_from_steps(intermediate_steps: List[Any]) -> Dict[str, Any]:
    """Parses agent intermediate steps to find key data from tool outputs."""
    findings = {}
    geojson_data = None
    if not intermediate_steps:
        return findings, geojson_data
        
    for action, observation in intermediate_steps:
        try:
            obs_json = json.loads(observation)
            if obs_json.get("status") == "success" and "data" in obs_json:
                data = obs_json["data"]
                if isinstance(data, dict):
                    if data.get("type") == "FeatureCollection":
                        geojson_data = data
                    else:
                        findings.update(data)
        except (json.JSONDecodeError, TypeError):
            continue
    return findings, geojson_data

def get_agent_response(query: str, agent_executor: AgentExecutor, rag_system: HybridRAG | None) -> Dict[str, Any]:
    """
    Core logic function. Invokes agent and processes the response. Returns a dictionary.
    """
    log = structlog.get_logger("agent_handler")
    context = ""
    if rag_system:
        context = rag_system.retrieve(query)
    
    agent_input = {"input": f"User Query: {query}\n\nContext from documents:\n{context}"}
    agent_response = agent_executor.invoke(agent_input)
    
    recommendation = agent_response.get("output", handle_ambiguity())
    
    intermediate_steps = agent_response.get("intermediate_steps", [])
    findings, geojson_data = _extract_data_from_steps(intermediate_steps)
    log.info("Extracted data from tool outputs.", findings=findings, has_geojson=bool(geojson_data))
    
    simulated_area_info = {"is_historically_underserved": True}
    xai_explanation = generate_xai_explanation(recommendation, findings)
    ethical_check = apply_ethical_guardrail(recommendation, simulated_area_info)
    
    return {
        "recommendation": recommendation,
        "justification": xai_explanation,
        "ethical_check": ethical_check,
        "geojson_data": geojson_data,
    }

def run_query(query: str, agent_executor: AgentExecutor, rag_system: HybridRAG | None):
    """
    CLI-specific function. Calls the core logic and prints the response using Typer.
    """
    try:
        response_data = get_agent_response(query, agent_executor, rag_system)
        
        typer.echo("\n" + "="*50)
        typer.secho("KUMULUS AI Consultant Response", fg=typer.colors.BRIGHT_BLUE, bold=True)
        typer.echo("="*50)
        typer.secho(f"\nRecommendation:", fg=typer.colors.CYAN)
        typer.echo(f"{response_data['recommendation']}")
        typer.secho(f"\nJustification:", fg=typer.colors.CYAN)
        typer.echo(f"{response_data['justification']}")
        typer.secho(f"\nEthical Check:", fg=typer.colors.CYAN)
        typer.secho(
            f"{response_data['ethical_check']['reason']}",
            fg=typer.colors.YELLOW if response_data['ethical_check']["flagged"] else typer.colors.GREEN
        )
        if response_data['geojson_data']:
            typer.secho("\nGeospatial data was generated and would be displayed in the UI.", fg=typer.colors.MAGENTA)
        typer.echo("="*50)

    except Exception as e:
        structlog.get_logger("query_handler").error("Unhandled exception.", error=str(e), exc_info=True)
        typer.secho(f"\nAn unexpected error occurred: {e}", fg=typer.colors.RED)

def main():
    """Main entry point for the KUMULUS AI Consultant CLI application."""
    setup_logging()
    agent_executor, rag_system = initialize_system()
    structlog.get_logger("chat_session").info("Starting interactive chat session.")
    typer.echo("Welcome to the KUMULUS AI Consultant. Type 'exit' to quit.")
    while True:
        query = typer.prompt("\n[User]")
        if query.lower() == "exit":
            typer.echo("Exiting consultant. Goodbye!")
            break
        run_query(query, agent_executor, rag_system)

if __name__ == "__main__":
    main()