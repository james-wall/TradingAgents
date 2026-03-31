"""Load agent configurations from agent_configs.yaml."""

from pathlib import Path
import yaml
from tradingagents.default_config import DEFAULT_CONFIG

DEFAULT_CONFIG_FILE = "agent_configs.yaml"

# Analyst processing order (must match cli/main.py ANALYST_ORDER)
ANALYST_ORDER = ["market", "social", "news", "fundamentals"]


def load_agent_configs(config_file: str = DEFAULT_CONFIG_FILE) -> list[dict]:
    """Load and return the list of agent config dicts from YAML."""
    path = Path(config_file)
    if not path.exists():
        raise FileNotFoundError(f"Agent config file not found: {config_file}. Create it or run with --help.")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("agents", [])


def agent_to_graph_config(agent_cfg: dict) -> dict:
    """Convert an agent YAML entry to a TradingAgentsGraph config dict."""
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = agent_cfg.get("llm_provider", "openai")
    config["deep_llm_provider"] = agent_cfg.get("deep_llm_provider")  # None = same as llm_provider
    config["deep_think_llm"] = agent_cfg.get("deep_think_llm", DEFAULT_CONFIG["deep_think_llm"])
    config["quick_think_llm"] = agent_cfg.get("quick_think_llm", DEFAULT_CONFIG["quick_think_llm"])
    config["backend_url"] = agent_cfg.get("backend_url", DEFAULT_CONFIG.get("backend_url"))
    config["deep_backend_url"] = agent_cfg.get("deep_backend_url")
    config["google_thinking_level"] = agent_cfg.get("google_thinking_level")
    config["openai_reasoning_effort"] = agent_cfg.get("openai_reasoning_effort")
    config["max_debate_rounds"] = agent_cfg.get("research_depth", 1)
    config["max_risk_discuss_rounds"] = agent_cfg.get("research_depth", 1)
    return config


def get_analyst_keys(agent_cfg: dict) -> list[str]:
    """Return ordered analyst key list for an agent config."""
    selected = set(agent_cfg.get("analysts", ["market", "social", "news", "fundamentals"]))
    return [a for a in ANALYST_ORDER if a in selected]
