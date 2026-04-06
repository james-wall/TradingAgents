from typing import Optional
import datetime
import typer
from pathlib import Path
from functools import wraps
from rich.console import Console
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.columns import Columns
from rich.markdown import Markdown
from rich.layout import Layout
from rich.text import Text
from rich.table import Table
from collections import deque
import time
from rich.tree import Tree
from rich import box
from rich.align import Align
from rich.rule import Rule

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from cli.models import AnalystType
from cli.utils import *
from cli.announcements import fetch_announcements, display_announcements
from cli.stats_handler import StatsCallbackHandler
from cli.portfolio_aggregator import aggregate_portfolio_recommendations

console = Console()

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,  # Enable shell completion
)


# Create a deque to store recent messages with a maximum length
class MessageBuffer:
    # Fixed teams that always run (not user-selectable)
    FIXED_AGENTS = {
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    # Analyst name mapping
    ANALYST_MAPPING = {
        "market": "Market Analyst",
        "social": "Social Analyst",
        "news": "News Analyst",
        "fundamentals": "Fundamentals Analyst",
    }

    # Report section mapping: section -> (analyst_key for filtering, finalizing_agent)
    # analyst_key: which analyst selection controls this section (None = always included)
    # finalizing_agent: which agent must be "completed" for this report to count as done
    REPORT_SECTIONS = {
        "market_report": ("market", "Market Analyst"),
        "sentiment_report": ("social", "Social Analyst"),
        "news_report": ("news", "News Analyst"),
        "fundamentals_report": ("fundamentals", "Fundamentals Analyst"),
        "investment_plan": (None, "Research Manager"),
        "trader_investment_plan": (None, "Trader"),
        "final_trade_decision": (None, "Portfolio Manager"),
    }

    def __init__(self, max_length=100):
        self.messages = deque(maxlen=max_length)
        self.tool_calls = deque(maxlen=max_length)
        self.current_report = None
        self.final_report = None  # Store the complete final report
        self.agent_status = {}
        self.current_agent = None
        self.report_sections = {}
        self.selected_analysts = []
        self._last_message_id = None

    def init_for_analysis(self, selected_analysts):
        """Initialize agent status and report sections based on selected analysts.

        Args:
            selected_analysts: List of analyst type strings (e.g., ["market", "news"])
        """
        self.selected_analysts = [a.lower() for a in selected_analysts]

        # Build agent_status dynamically
        self.agent_status = {}

        # Add selected analysts
        for analyst_key in self.selected_analysts:
            if analyst_key in self.ANALYST_MAPPING:
                self.agent_status[self.ANALYST_MAPPING[analyst_key]] = "pending"

        # Add fixed teams
        for team_agents in self.FIXED_AGENTS.values():
            for agent in team_agents:
                self.agent_status[agent] = "pending"

        # Build report_sections dynamically
        self.report_sections = {}
        for section, (analyst_key, _) in self.REPORT_SECTIONS.items():
            if analyst_key is None or analyst_key in self.selected_analysts:
                self.report_sections[section] = None

        # Reset other state
        self.current_report = None
        self.final_report = None
        self.current_agent = None
        self.messages.clear()
        self.tool_calls.clear()
        self._last_message_id = None

    def get_completed_reports_count(self):
        """Count reports that are finalized (their finalizing agent is completed).

        A report is considered complete when:
        1. The report section has content (not None), AND
        2. The agent responsible for finalizing that report has status "completed"

        This prevents interim updates (like debate rounds) from counting as completed.
        """
        count = 0
        for section in self.report_sections:
            if section not in self.REPORT_SECTIONS:
                continue
            _, finalizing_agent = self.REPORT_SECTIONS[section]
            # Report is complete if it has content AND its finalizing agent is done
            has_content = self.report_sections.get(section) is not None
            agent_done = self.agent_status.get(finalizing_agent) == "completed"
            if has_content and agent_done:
                count += 1
        return count

    def add_message(self, message_type, content):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((timestamp, message_type, content))

    def add_tool_call(self, tool_name, args):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append((timestamp, tool_name, args))

    def update_agent_status(self, agent, status):
        if agent in self.agent_status:
            self.agent_status[agent] = status
            self.current_agent = agent

    def update_report_section(self, section_name, content):
        if section_name in self.report_sections:
            self.report_sections[section_name] = content
            self._update_current_report()

    def _update_current_report(self):
        # For the panel display, only show the most recently updated section
        latest_section = None
        latest_content = None

        # Find the most recently updated section
        for section, content in self.report_sections.items():
            if content is not None:
                latest_section = section
                latest_content = content
               
        if latest_section and latest_content:
            # Format the current section for display
            section_titles = {
                "market_report": "Market Analysis",
                "sentiment_report": "Social Sentiment",
                "news_report": "News Analysis",
                "fundamentals_report": "Fundamentals Analysis",
                "investment_plan": "Research Team Decision",
                "trader_investment_plan": "Trading Team Plan",
                "final_trade_decision": "Portfolio Management Decision",
            }
            self.current_report = (
                f"### {section_titles[latest_section]}\n{latest_content}"
            )

        # Update the final complete report
        self._update_final_report()

    def _update_final_report(self):
        report_parts = []

        # Analyst Team Reports - use .get() to handle missing sections
        analyst_sections = ["market_report", "sentiment_report", "news_report", "fundamentals_report"]
        if any(self.report_sections.get(section) for section in analyst_sections):
            report_parts.append("## Analyst Team Reports")
            if self.report_sections.get("market_report"):
                report_parts.append(
                    f"### Market Analysis\n{self.report_sections['market_report']}"
                )
            if self.report_sections.get("sentiment_report"):
                report_parts.append(
                    f"### Social Sentiment\n{self.report_sections['sentiment_report']}"
                )
            if self.report_sections.get("news_report"):
                report_parts.append(
                    f"### News Analysis\n{self.report_sections['news_report']}"
                )
            if self.report_sections.get("fundamentals_report"):
                report_parts.append(
                    f"### Fundamentals Analysis\n{self.report_sections['fundamentals_report']}"
                )

        # Research Team Reports
        if self.report_sections.get("investment_plan"):
            report_parts.append("## Research Team Decision")
            report_parts.append(f"{self.report_sections['investment_plan']}")

        # Trading Team Reports
        if self.report_sections.get("trader_investment_plan"):
            report_parts.append("## Trading Team Plan")
            report_parts.append(f"{self.report_sections['trader_investment_plan']}")

        # Portfolio Management Decision
        if self.report_sections.get("final_trade_decision"):
            report_parts.append("## Portfolio Management Decision")
            report_parts.append(f"{self.report_sections['final_trade_decision']}")

        self.final_report = "\n\n".join(report_parts) if report_parts else None


message_buffer = MessageBuffer()


def create_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    layout["main"].split_column(
        Layout(name="upper", ratio=3), Layout(name="analysis", ratio=5)
    )
    layout["upper"].split_row(
        Layout(name="progress", ratio=2), Layout(name="messages", ratio=3)
    )
    return layout


def format_tokens(n):
    """Format token count for display."""
    if n >= 1000:
        return f"{n/1000:.1f}k"
    return str(n)


def update_display(layout, spinner_text=None, stats_handler=None, start_time=None):
    # Header with welcome message
    layout["header"].update(
        Panel(
            "[bold green]Welcome to TradingAgents CLI[/bold green]\n"
            "[dim]© [Tauric Research](https://github.com/TauricResearch)[/dim]",
            title="Welcome to TradingAgents",
            border_style="green",
            padding=(1, 2),
            expand=True,
        )
    )

    # Progress panel showing agent status
    progress_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        box=box.SIMPLE_HEAD,  # Use simple header with horizontal lines
        title=None,  # Remove the redundant Progress title
        padding=(0, 2),  # Add horizontal padding
        expand=True,  # Make table expand to fill available space
    )
    progress_table.add_column("Team", style="cyan", justify="center", width=20)
    progress_table.add_column("Agent", style="green", justify="center", width=20)
    progress_table.add_column("Status", style="yellow", justify="center", width=20)

    # Group agents by team - filter to only include agents in agent_status
    all_teams = {
        "Analyst Team": [
            "Market Analyst",
            "Social Analyst",
            "News Analyst",
            "Fundamentals Analyst",
        ],
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    # Filter teams to only include agents that are in agent_status
    teams = {}
    for team, agents in all_teams.items():
        active_agents = [a for a in agents if a in message_buffer.agent_status]
        if active_agents:
            teams[team] = active_agents

    for team, agents in teams.items():
        # Add first agent with team name
        first_agent = agents[0]
        status = message_buffer.agent_status.get(first_agent, "pending")
        if status == "in_progress":
            spinner = Spinner(
                "dots", text="[blue]in_progress[/blue]", style="bold cyan"
            )
            status_cell = spinner
        else:
            status_color = {
                "pending": "yellow",
                "completed": "green",
                "error": "red",
            }.get(status, "white")
            status_cell = f"[{status_color}]{status}[/{status_color}]"
        progress_table.add_row(team, first_agent, status_cell)

        # Add remaining agents in team
        for agent in agents[1:]:
            status = message_buffer.agent_status.get(agent, "pending")
            if status == "in_progress":
                spinner = Spinner(
                    "dots", text="[blue]in_progress[/blue]", style="bold cyan"
                )
                status_cell = spinner
            else:
                status_color = {
                    "pending": "yellow",
                    "completed": "green",
                    "error": "red",
                }.get(status, "white")
                status_cell = f"[{status_color}]{status}[/{status_color}]"
            progress_table.add_row("", agent, status_cell)

        # Add horizontal line after each team
        progress_table.add_row("─" * 20, "─" * 20, "─" * 20, style="dim")

    layout["progress"].update(
        Panel(progress_table, title="Progress", border_style="cyan", padding=(1, 2))
    )

    # Messages panel showing recent messages and tool calls
    messages_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        expand=True,  # Make table expand to fill available space
        box=box.MINIMAL,  # Use minimal box style for a lighter look
        show_lines=True,  # Keep horizontal lines
        padding=(0, 1),  # Add some padding between columns
    )
    messages_table.add_column("Time", style="cyan", width=8, justify="center")
    messages_table.add_column("Type", style="green", width=10, justify="center")
    messages_table.add_column(
        "Content", style="white", no_wrap=False, ratio=1
    )  # Make content column expand

    # Combine tool calls and messages
    all_messages = []

    # Add tool calls
    for timestamp, tool_name, args in message_buffer.tool_calls:
        formatted_args = format_tool_args(args)
        all_messages.append((timestamp, "Tool", f"{tool_name}: {formatted_args}"))

    # Add regular messages
    for timestamp, msg_type, content in message_buffer.messages:
        content_str = str(content) if content else ""
        if len(content_str) > 200:
            content_str = content_str[:197] + "..."
        all_messages.append((timestamp, msg_type, content_str))

    # Sort by timestamp descending (newest first)
    all_messages.sort(key=lambda x: x[0], reverse=True)

    # Calculate how many messages we can show based on available space
    max_messages = 12

    # Get the first N messages (newest ones)
    recent_messages = all_messages[:max_messages]

    # Add messages to table (already in newest-first order)
    for timestamp, msg_type, content in recent_messages:
        # Format content with word wrapping
        wrapped_content = Text(content, overflow="fold")
        messages_table.add_row(timestamp, msg_type, wrapped_content)

    layout["messages"].update(
        Panel(
            messages_table,
            title="Messages & Tools",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Analysis panel showing current report
    if message_buffer.current_report:
        layout["analysis"].update(
            Panel(
                Markdown(message_buffer.current_report),
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        layout["analysis"].update(
            Panel(
                "[italic]Waiting for analysis report...[/italic]",
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )

    # Footer with statistics
    # Agent progress - derived from agent_status dict
    agents_completed = sum(
        1 for status in message_buffer.agent_status.values() if status == "completed"
    )
    agents_total = len(message_buffer.agent_status)

    # Report progress - based on agent completion (not just content existence)
    reports_completed = message_buffer.get_completed_reports_count()
    reports_total = len(message_buffer.report_sections)

    # Build stats parts
    stats_parts = [f"Agents: {agents_completed}/{agents_total}"]

    # LLM and tool stats from callback handler
    if stats_handler:
        stats = stats_handler.get_stats()
        stats_parts.append(f"LLM: {stats['llm_calls']}")
        stats_parts.append(f"Tools: {stats['tool_calls']}")

        # Token display with graceful fallback
        if stats["tokens_in"] > 0 or stats["tokens_out"] > 0:
            tokens_str = f"Tokens: {format_tokens(stats['tokens_in'])}\u2191 {format_tokens(stats['tokens_out'])}\u2193"
        else:
            tokens_str = "Tokens: --"
        stats_parts.append(tokens_str)

    stats_parts.append(f"Reports: {reports_completed}/{reports_total}")

    # Elapsed time
    if start_time:
        elapsed = time.time() - start_time
        elapsed_str = f"\u23f1 {int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        stats_parts.append(elapsed_str)

    stats_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    stats_table.add_column("Stats", justify="center")
    stats_table.add_row(" | ".join(stats_parts))

    layout["footer"].update(Panel(stats_table, border_style="grey50"))


def get_user_selections():
    """Get all user selections before starting the analysis display."""
    # Display ASCII art welcome message
    with open("./cli/static/welcome.txt", "r", encoding="utf-8") as f:
        welcome_ascii = f.read()

    # Create welcome box content
    welcome_content = f"{welcome_ascii}\n"
    welcome_content += "[bold green]TradingAgents: Multi-Agents LLM Financial Trading Framework - CLI[/bold green]\n\n"
    welcome_content += "[bold]Workflow Steps:[/bold]\n"
    welcome_content += "I. Analyst Team → II. Research Team → III. Trader → IV. Risk Management → V. Portfolio Management\n\n"
    welcome_content += (
        "[dim]Built by [Tauric Research](https://github.com/TauricResearch)[/dim]"
    )

    # Create and center the welcome box
    welcome_box = Panel(
        welcome_content,
        border_style="green",
        padding=(1, 2),
        title="Welcome to TradingAgents",
        subtitle="Multi-Agents LLM Financial Trading Framework",
    )
    console.print(Align.center(welcome_box))
    console.print()
    console.print()  # Add vertical space before announcements

    # Fetch and display announcements (silent on failure)
    announcements = fetch_announcements()
    display_announcements(console, announcements)

    # Create a boxed questionnaire for each step
    def create_question_box(title, prompt, default=None):
        box_content = f"[bold]{title}[/bold]\n"
        box_content += f"[dim]{prompt}[/dim]"
        if default:
            box_content += f"\n[dim]Default: {default}[/dim]"
        return Panel(box_content, border_style="blue", padding=(1, 2))

    # Step 1: Ticker symbol
    console.print(
        create_question_box(
            "Step 1: Ticker Symbol", "Enter the ticker symbol to analyze", "SPY"
        )
    )
    selected_ticker = get_ticker()

    # Step 2: Analysis date
    default_date = datetime.datetime.now().strftime("%Y-%m-%d")
    console.print(
        create_question_box(
            "Step 2: Analysis Date",
            "Enter the analysis date (YYYY-MM-DD)",
            default_date,
        )
    )
    analysis_date = get_analysis_date()

    # Step 3: Select analysts
    console.print(
        create_question_box(
            "Step 3: Analysts Team", "Select your LLM analyst agents for the analysis"
        )
    )
    selected_analysts = select_analysts()
    console.print(
        f"[green]Selected analysts:[/green] {', '.join(analyst.value for analyst in selected_analysts)}"
    )

    # Step 4: Research depth
    console.print(
        create_question_box(
            "Step 4: Research Depth", "Select your research depth level"
        )
    )
    selected_research_depth = select_research_depth()

    # Step 5: OpenAI backend
    console.print(
        create_question_box(
            "Step 5: OpenAI backend", "Select which service to talk to"
        )
    )
    selected_llm_provider, backend_url = select_llm_provider()
    
    # Step 6: Thinking agents
    console.print(
        create_question_box(
            "Step 6: Thinking Agents", "Select your thinking agents for analysis"
        )
    )
    selected_shallow_thinker = select_shallow_thinking_agent(selected_llm_provider)
    selected_deep_thinker = select_deep_thinking_agent(selected_llm_provider)

    # Step 7: Provider-specific thinking configuration
    thinking_level = None
    reasoning_effort = None

    provider_lower = selected_llm_provider.lower()
    if provider_lower == "google":
        console.print(
            create_question_box(
                "Step 7: Thinking Mode",
                "Configure Gemini thinking mode"
            )
        )
        thinking_level = ask_gemini_thinking_config()
    elif provider_lower == "openai":
        console.print(
            create_question_box(
                "Step 7: Reasoning Effort",
                "Configure OpenAI reasoning effort level"
            )
        )
        reasoning_effort = ask_openai_reasoning_effort()

    return {
        "ticker": selected_ticker,
        "analysis_date": analysis_date,
        "analysts": selected_analysts,
        "research_depth": selected_research_depth,
        "llm_provider": selected_llm_provider.lower(),
        "backend_url": backend_url,
        "shallow_thinker": selected_shallow_thinker,
        "deep_thinker": selected_deep_thinker,
        "google_thinking_level": thinking_level,
        "openai_reasoning_effort": reasoning_effort,
    }


def get_ticker():
    """Get ticker symbol from user input."""
    return typer.prompt("", default="SPY")


def get_analysis_date():
    """Get the analysis date from user input."""
    while True:
        date_str = typer.prompt(
            "", default=datetime.datetime.now().strftime("%Y-%m-%d")
        )
        try:
            # Validate date format and ensure it's not in the future
            analysis_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            if analysis_date.date() > datetime.datetime.now().date():
                console.print("[red]Error: Analysis date cannot be in the future[/red]")
                continue
            return date_str
        except ValueError:
            console.print(
                "[red]Error: Invalid date format. Please use YYYY-MM-DD[/red]"
            )


def save_report_to_disk(final_state, ticker: str, save_path: Path):
    """Save complete analysis report to disk with organized subfolders."""
    save_path.mkdir(parents=True, exist_ok=True)
    sections = []

    # 1. Analysts
    analysts_dir = save_path / "1_analysts"
    analyst_parts = []
    if final_state.get("market_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "market.md").write_text(final_state["market_report"])
        analyst_parts.append(("Market Analyst", final_state["market_report"]))
    if final_state.get("sentiment_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "sentiment.md").write_text(final_state["sentiment_report"])
        analyst_parts.append(("Social Analyst", final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "news.md").write_text(final_state["news_report"])
        analyst_parts.append(("News Analyst", final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "fundamentals.md").write_text(final_state["fundamentals_report"])
        analyst_parts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
    if analyst_parts:
        content = "\n\n".join(f"### {name}\n{text}" for name, text in analyst_parts)
        sections.append(f"## I. Analyst Team Reports\n\n{content}")

    # 2. Research
    if final_state.get("investment_debate_state"):
        research_dir = save_path / "2_research"
        debate = final_state["investment_debate_state"]
        research_parts = []
        if debate.get("bull_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bull.md").write_text(debate["bull_history"])
            research_parts.append(("Bull Researcher", debate["bull_history"]))
        if debate.get("bear_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bear.md").write_text(debate["bear_history"])
            research_parts.append(("Bear Researcher", debate["bear_history"]))
        if debate.get("judge_decision"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "manager.md").write_text(debate["judge_decision"])
            research_parts.append(("Research Manager", debate["judge_decision"]))
        if research_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in research_parts)
            sections.append(f"## II. Research Team Decision\n\n{content}")

    # 3. Trading
    if final_state.get("trader_investment_plan"):
        trading_dir = save_path / "3_trading"
        trading_dir.mkdir(exist_ok=True)
        (trading_dir / "trader.md").write_text(final_state["trader_investment_plan"])
        sections.append(f"## III. Trading Team Plan\n\n### Trader\n{final_state['trader_investment_plan']}")

    # 4. Risk Management
    if final_state.get("risk_debate_state"):
        risk_dir = save_path / "4_risk"
        risk = final_state["risk_debate_state"]
        risk_parts = []
        if risk.get("aggressive_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "aggressive.md").write_text(risk["aggressive_history"])
            risk_parts.append(("Aggressive Analyst", risk["aggressive_history"]))
        if risk.get("conservative_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "conservative.md").write_text(risk["conservative_history"])
            risk_parts.append(("Conservative Analyst", risk["conservative_history"]))
        if risk.get("neutral_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "neutral.md").write_text(risk["neutral_history"])
            risk_parts.append(("Neutral Analyst", risk["neutral_history"]))
        if risk_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in risk_parts)
            sections.append(f"## IV. Risk Management Team Decision\n\n{content}")

        # 5. Portfolio Manager
        if risk.get("judge_decision"):
            portfolio_dir = save_path / "5_portfolio"
            portfolio_dir.mkdir(exist_ok=True)
            (portfolio_dir / "decision.md").write_text(risk["judge_decision"])
            sections.append(f"## V. Portfolio Manager Decision\n\n### Portfolio Manager\n{risk['judge_decision']}")

    # Write consolidated report
    header = f"# Trading Analysis Report: {ticker}\n\nGenerated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    (save_path / "complete_report.md").write_text(header + "\n\n".join(sections))
    return save_path / "complete_report.md"


def display_complete_report(final_state):
    """Display the complete analysis report sequentially (avoids truncation)."""
    console.print()
    console.print(Rule("Complete Analysis Report", style="bold green"))

    # I. Analyst Team Reports
    analysts = []
    if final_state.get("market_report"):
        analysts.append(("Market Analyst", final_state["market_report"]))
    if final_state.get("sentiment_report"):
        analysts.append(("Social Analyst", final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts.append(("News Analyst", final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
    if analysts:
        console.print(Panel("[bold]I. Analyst Team Reports[/bold]", border_style="cyan"))
        for title, content in analysts:
            console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    # II. Research Team Reports
    if final_state.get("investment_debate_state"):
        debate = final_state["investment_debate_state"]
        research = []
        if debate.get("bull_history"):
            research.append(("Bull Researcher", debate["bull_history"]))
        if debate.get("bear_history"):
            research.append(("Bear Researcher", debate["bear_history"]))
        if debate.get("judge_decision"):
            research.append(("Research Manager", debate["judge_decision"]))
        if research:
            console.print(Panel("[bold]II. Research Team Decision[/bold]", border_style="magenta"))
            for title, content in research:
                console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    # III. Trading Team
    if final_state.get("trader_investment_plan"):
        console.print(Panel("[bold]III. Trading Team Plan[/bold]", border_style="yellow"))
        console.print(Panel(Markdown(final_state["trader_investment_plan"]), title="Trader", border_style="blue", padding=(1, 2)))

    # IV. Risk Management Team
    if final_state.get("risk_debate_state"):
        risk = final_state["risk_debate_state"]
        risk_reports = []
        if risk.get("aggressive_history"):
            risk_reports.append(("Aggressive Analyst", risk["aggressive_history"]))
        if risk.get("conservative_history"):
            risk_reports.append(("Conservative Analyst", risk["conservative_history"]))
        if risk.get("neutral_history"):
            risk_reports.append(("Neutral Analyst", risk["neutral_history"]))
        if risk_reports:
            console.print(Panel("[bold]IV. Risk Management Team Decision[/bold]", border_style="red"))
            for title, content in risk_reports:
                console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

        # V. Portfolio Manager Decision
        if risk.get("judge_decision"):
            console.print(Panel("[bold]V. Portfolio Manager Decision[/bold]", border_style="green"))
            console.print(Panel(Markdown(risk["judge_decision"]), title="Portfolio Manager", border_style="blue", padding=(1, 2)))


def update_research_team_status(status):
    """Update status for research team members (not Trader)."""
    research_team = ["Bull Researcher", "Bear Researcher", "Research Manager"]
    for agent in research_team:
        message_buffer.update_agent_status(agent, status)


# Ordered list of analysts for status transitions
ANALYST_ORDER = ["market", "social", "news", "fundamentals"]
ANALYST_AGENT_NAMES = {
    "market": "Market Analyst",
    "social": "Social Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}
ANALYST_REPORT_MAP = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}


def update_analyst_statuses(message_buffer, chunk):
    """Update all analyst statuses based on current report state.

    Logic:
    - Analysts with reports = completed
    - First analyst without report = in_progress
    - Remaining analysts without reports = pending
    - When all analysts done, set Bull Researcher to in_progress
    """
    selected = message_buffer.selected_analysts
    found_active = False

    for analyst_key in ANALYST_ORDER:
        if analyst_key not in selected:
            continue

        agent_name = ANALYST_AGENT_NAMES[analyst_key]
        report_key = ANALYST_REPORT_MAP[analyst_key]
        has_report = bool(chunk.get(report_key))

        if has_report:
            message_buffer.update_agent_status(agent_name, "completed")
            message_buffer.update_report_section(report_key, chunk[report_key])
        elif not found_active:
            message_buffer.update_agent_status(agent_name, "in_progress")
            found_active = True
        else:
            message_buffer.update_agent_status(agent_name, "pending")

    # When all analysts complete, transition research team to in_progress
    if not found_active and selected:
        if message_buffer.agent_status.get("Bull Researcher") == "pending":
            message_buffer.update_agent_status("Bull Researcher", "in_progress")

def extract_content_string(content):
    """Extract string content from various message formats.
    Returns None if no meaningful text content is found.
    """
    import ast

    def is_empty(val):
        """Check if value is empty using Python's truthiness."""
        if val is None or val == '':
            return True
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return True
            try:
                return not bool(ast.literal_eval(s))
            except (ValueError, SyntaxError):
                return False  # Can't parse = real text
        return not bool(val)

    if is_empty(content):
        return None

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        text = content.get('text', '')
        return text.strip() if not is_empty(text) else None

    if isinstance(content, list):
        text_parts = [
            item.get('text', '').strip() if isinstance(item, dict) and item.get('type') == 'text'
            else (item.strip() if isinstance(item, str) else '')
            for item in content
        ]
        result = ' '.join(t for t in text_parts if t and not is_empty(t))
        return result if result else None

    return str(content).strip() if not is_empty(content) else None


def classify_message_type(message) -> tuple[str, str | None]:
    """Classify LangChain message into display type and extract content.

    Returns:
        (type, content) - type is one of: User, Agent, Data, Control
                        - content is extracted string or None
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    content = extract_content_string(getattr(message, 'content', None))

    if isinstance(message, HumanMessage):
        if content and content.strip() == "Continue":
            return ("Control", content)
        return ("User", content)

    if isinstance(message, ToolMessage):
        return ("Data", content)

    if isinstance(message, AIMessage):
        return ("Agent", content)

    # Fallback for unknown types
    return ("System", content)


def format_tool_args(args, max_length=80) -> str:
    """Format tool arguments for terminal display."""
    result = str(args)
    if len(result) > max_length:
        return result[:max_length - 3] + "..."
    return result

def run_analysis():
    # First get all user selections
    selections = get_user_selections()

    # Create config with selected research depth
    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = selections["research_depth"]
    config["max_risk_discuss_rounds"] = selections["research_depth"]
    config["quick_think_llm"] = selections["shallow_thinker"]
    config["deep_think_llm"] = selections["deep_thinker"]
    config["backend_url"] = selections["backend_url"]
    config["llm_provider"] = selections["llm_provider"].lower()
    # Provider-specific thinking configuration
    config["google_thinking_level"] = selections.get("google_thinking_level")
    config["openai_reasoning_effort"] = selections.get("openai_reasoning_effort")

    # Create stats callback handler for tracking LLM/tool calls
    stats_handler = StatsCallbackHandler()

    # Normalize analyst selection to predefined order (selection is a 'set', order is fixed)
    selected_set = {analyst.value for analyst in selections["analysts"]}
    selected_analyst_keys = [a for a in ANALYST_ORDER if a in selected_set]

    # Initialize the graph with callbacks bound to LLMs
    graph = TradingAgentsGraph(
        selected_analyst_keys,
        config=config,
        debug=True,
        callbacks=[stats_handler],
    )

    # Initialize message buffer with selected analysts
    message_buffer.init_for_analysis(selected_analyst_keys)

    # Track start time for elapsed display
    start_time = time.time()

    # Create result directory
    results_dir = Path(config["results_dir"]) / selections["ticker"] / selections["analysis_date"]
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "message_tool.log"
    log_file.touch(exist_ok=True)

    def save_message_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, message_type, content = obj.messages[-1]
            content = content.replace("\n", " ")  # Replace newlines with spaces
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [{message_type}] {content}\n")
        return wrapper
    
    def save_tool_call_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, tool_name, args = obj.tool_calls[-1]
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [Tool Call] {tool_name}({args_str})\n")
        return wrapper

    def save_report_section_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(section_name, content):
            func(section_name, content)
            if section_name in obj.report_sections and obj.report_sections[section_name] is not None:
                content = obj.report_sections[section_name]
                if content:
                    file_name = f"{section_name}.md"
                    with open(report_dir / file_name, "w", encoding="utf-8") as f:
                        f.write(content)
        return wrapper

    message_buffer.add_message = save_message_decorator(message_buffer, "add_message")
    message_buffer.add_tool_call = save_tool_call_decorator(message_buffer, "add_tool_call")
    message_buffer.update_report_section = save_report_section_decorator(message_buffer, "update_report_section")

    # Now start the display layout
    layout = create_layout()

    with Live(layout, refresh_per_second=4) as live:
        # Initial display
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Add initial messages
        message_buffer.add_message("System", f"Selected ticker: {selections['ticker']}")
        message_buffer.add_message(
            "System", f"Analysis date: {selections['analysis_date']}"
        )
        message_buffer.add_message(
            "System",
            f"Selected analysts: {', '.join(analyst.value for analyst in selections['analysts'])}",
        )
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Update agent status to in_progress for the first analyst
        first_analyst = f"{selections['analysts'][0].value.capitalize()} Analyst"
        message_buffer.update_agent_status(first_analyst, "in_progress")
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Create spinner text
        spinner_text = (
            f"Analyzing {selections['ticker']} on {selections['analysis_date']}..."
        )
        update_display(layout, spinner_text, stats_handler=stats_handler, start_time=start_time)

        # Initialize state and get graph args with callbacks
        init_agent_state = graph.propagator.create_initial_state(
            selections["ticker"], selections["analysis_date"]
        )
        # Pass callbacks to graph config for tool execution tracking
        # (LLM tracking is handled separately via LLM constructor)
        args = graph.propagator.get_graph_args(callbacks=[stats_handler])

        # Stream the analysis
        trace = []
        for chunk in graph.graph.stream(init_agent_state, **args):
            # Process messages if present (skip duplicates via message ID)
            if len(chunk["messages"]) > 0:
                last_message = chunk["messages"][-1]
                msg_id = getattr(last_message, "id", None)

                if msg_id != message_buffer._last_message_id:
                    message_buffer._last_message_id = msg_id

                    # Add message to buffer
                    msg_type, content = classify_message_type(last_message)
                    if content and content.strip():
                        message_buffer.add_message(msg_type, content)

                    # Handle tool calls
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        for tool_call in last_message.tool_calls:
                            if isinstance(tool_call, dict):
                                message_buffer.add_tool_call(
                                    tool_call["name"], tool_call["args"]
                                )
                            else:
                                message_buffer.add_tool_call(tool_call.name, tool_call.args)

            # Update analyst statuses based on report state (runs on every chunk)
            update_analyst_statuses(message_buffer, chunk)

            # Research Team - Handle Investment Debate State
            if chunk.get("investment_debate_state"):
                debate_state = chunk["investment_debate_state"]
                bull_hist = debate_state.get("bull_history", "").strip()
                bear_hist = debate_state.get("bear_history", "").strip()
                judge = debate_state.get("judge_decision", "").strip()

                # Only update status when there's actual content
                if bull_hist or bear_hist:
                    update_research_team_status("in_progress")
                if bull_hist:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Bull Researcher Analysis\n{bull_hist}"
                    )
                if bear_hist:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Bear Researcher Analysis\n{bear_hist}"
                    )
                if judge:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Research Manager Decision\n{judge}"
                    )
                    update_research_team_status("completed")
                    message_buffer.update_agent_status("Trader", "in_progress")

            # Trading Team
            if chunk.get("trader_investment_plan"):
                message_buffer.update_report_section(
                    "trader_investment_plan", chunk["trader_investment_plan"]
                )
                if message_buffer.agent_status.get("Trader") != "completed":
                    message_buffer.update_agent_status("Trader", "completed")
                    message_buffer.update_agent_status("Aggressive Analyst", "in_progress")

            # Risk Management Team - Handle Risk Debate State
            if chunk.get("risk_debate_state"):
                risk_state = chunk["risk_debate_state"]
                agg_hist = risk_state.get("aggressive_history", "").strip()
                con_hist = risk_state.get("conservative_history", "").strip()
                neu_hist = risk_state.get("neutral_history", "").strip()
                judge = risk_state.get("judge_decision", "").strip()

                if agg_hist:
                    if message_buffer.agent_status.get("Aggressive Analyst") != "completed":
                        message_buffer.update_agent_status("Aggressive Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Aggressive Analyst Analysis\n{agg_hist}"
                    )
                if con_hist:
                    if message_buffer.agent_status.get("Conservative Analyst") != "completed":
                        message_buffer.update_agent_status("Conservative Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Conservative Analyst Analysis\n{con_hist}"
                    )
                if neu_hist:
                    if message_buffer.agent_status.get("Neutral Analyst") != "completed":
                        message_buffer.update_agent_status("Neutral Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Neutral Analyst Analysis\n{neu_hist}"
                    )
                if judge:
                    if message_buffer.agent_status.get("Portfolio Manager") != "completed":
                        message_buffer.update_agent_status("Portfolio Manager", "in_progress")
                        message_buffer.update_report_section(
                            "final_trade_decision", f"### Portfolio Manager Decision\n{judge}"
                        )
                        message_buffer.update_agent_status("Aggressive Analyst", "completed")
                        message_buffer.update_agent_status("Conservative Analyst", "completed")
                        message_buffer.update_agent_status("Neutral Analyst", "completed")
                        message_buffer.update_agent_status("Portfolio Manager", "completed")

            # Update the display
            update_display(layout, stats_handler=stats_handler, start_time=start_time)

            trace.append(chunk)

        # Get final state and decision
        final_state = trace[-1]
        decision = graph.process_signal(final_state["final_trade_decision"])

        # Update all agent statuses to completed
        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "completed")

        message_buffer.add_message(
            "System", f"Completed analysis for {selections['analysis_date']}"
        )

        # Update final report sections
        for section in message_buffer.report_sections.keys():
            if section in final_state:
                message_buffer.update_report_section(section, final_state[section])

        update_display(layout, stats_handler=stats_handler, start_time=start_time)

    # Post-analysis prompts (outside Live context for clean interaction)
    console.print("\n[bold cyan]Analysis Complete![/bold cyan]\n")

    # Prompt to save report
    save_choice = typer.prompt("Save report?", default="Y").strip().upper()
    if save_choice in ("Y", "YES", ""):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = Path.cwd() / "reports" / f"{selections['ticker']}_{timestamp}"
        save_path_str = typer.prompt(
            "Save path (press Enter for default)",
            default=str(default_path)
        ).strip()
        save_path = Path(save_path_str)
        try:
            report_file = save_report_to_disk(final_state, selections["ticker"], save_path)
            console.print(f"\n[green]✓ Report saved to:[/green] {save_path.resolve()}")
            console.print(f"  [dim]Complete report:[/dim] {report_file.name}")
        except Exception as e:
            console.print(f"[red]Error saving report: {e}[/red]")

    # Prompt to display full report
    display_choice = typer.prompt("\nDisplay full report on screen?", default="Y").strip().upper()
    if display_choice in ("Y", "YES", ""):
        display_complete_report(final_state)


@app.command()
def analyze():
    run_analysis()


def load_tickers_from_file(file_path: str) -> list[str]:
    """Read tickers from a text file, one per line. Ignores blank lines and #-comments."""
    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)
    tickers = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()  # strip inline comments
        if line:
            tickers.append(line.upper())
    if not tickers:
        console.print(f"[red]Error: No tickers found in {file_path}[/red]")
        raise typer.Exit(1)
    return tickers


def get_portfolio_selections(tickers: list[str]) -> dict:
    """Collect analysis configuration (no ticker prompt — tickers come from file)."""
    console.print(
        Panel(
            f"[bold cyan]Tickers loaded:[/bold cyan] {', '.join(tickers)}\n"
            f"[dim]{len(tickers)} stock(s) will be analyzed sequentially.[/dim]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()

    def create_question_box(title, prompt, default=None):
        box_content = f"[bold]{title}[/bold]\n[dim]{prompt}[/dim]"
        if default:
            box_content += f"\n[dim]Default: {default}[/dim]"
        return Panel(box_content, border_style="blue", padding=(1, 2))

    # Analysis date
    default_date = datetime.datetime.now().strftime("%Y-%m-%d")
    console.print(
        create_question_box("Analysis Date", "Enter the analysis date (YYYY-MM-DD)", default_date)
    )
    analysis_date = get_analysis_date()

    # Analysts
    console.print(create_question_box("Analysts Team", "Select your LLM analyst agents"))
    selected_analysts = select_analysts()
    console.print(
        f"[green]Selected analysts:[/green] {', '.join(a.value for a in selected_analysts)}"
    )

    # Research depth
    console.print(create_question_box("Research Depth", "Select your research depth level"))
    selected_research_depth = select_research_depth()

    # LLM provider
    console.print(create_question_box("LLM Backend", "Select which service to talk to"))
    selected_llm_provider, backend_url = select_llm_provider()

    # Models
    console.print(create_question_box("Thinking Agents", "Select your thinking agents"))
    selected_shallow_thinker = select_shallow_thinking_agent(selected_llm_provider)
    selected_deep_thinker = select_deep_thinking_agent(selected_llm_provider)

    # Provider-specific config
    thinking_level = None
    reasoning_effort = None
    provider_lower = selected_llm_provider.lower()
    if provider_lower == "google":
        console.print(create_question_box("Thinking Mode", "Configure Gemini thinking mode"))
        thinking_level = ask_gemini_thinking_config()
    elif provider_lower == "openai":
        console.print(create_question_box("Reasoning Effort", "Configure OpenAI reasoning effort"))
        reasoning_effort = ask_openai_reasoning_effort()

    return {
        "analysis_date": analysis_date,
        "analysts": selected_analysts,
        "research_depth": selected_research_depth,
        "llm_provider": selected_llm_provider.lower(),
        "backend_url": backend_url,
        "shallow_thinker": selected_shallow_thinker,
        "deep_thinker": selected_deep_thinker,
        "google_thinking_level": thinking_level,
        "openai_reasoning_effort": reasoning_effort,
    }


@app.command(name="analyze-portfolio")
def analyze_portfolio(
    file: Optional[str] = typer.Option(
        None, "--file", "-f", help="Path to .txt file with tickers (one per line)"
    )
):
    """Analyze a portfolio of stocks from a tickers file and generate weighted daily recommendations."""
    # Resolve tickers file
    if file is None:
        file = typer.prompt("Path to tickers file (.txt, one ticker per line)")
    tickers = load_tickers_from_file(file)

    # Display welcome header
    console.print(
        Panel(
            "[bold green]TradingAgents — Portfolio Analysis Mode[/bold green]\n"
            "[dim]Analyzes multiple tickers and synthesizes a weighted daily action plan.[/dim]",
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print()

    # Gather configuration
    selections = get_portfolio_selections(tickers)

    # Build graph config
    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = selections["research_depth"]
    config["max_risk_discuss_rounds"] = selections["research_depth"]
    config["quick_think_llm"] = selections["shallow_thinker"]
    config["deep_think_llm"] = selections["deep_thinker"]
    config["backend_url"] = selections["backend_url"]
    config["llm_provider"] = selections["llm_provider"]
    config["google_thinking_level"] = selections.get("google_thinking_level")
    config["openai_reasoning_effort"] = selections.get("openai_reasoning_effort")

    selected_set = {a.value for a in selections["analysts"]}
    selected_analyst_keys = [a for a in ANALYST_ORDER if a in selected_set]

    # Initialise graph once and reuse across all tickers
    console.print("[dim]Initialising analysis graph...[/dim]")
    graph = TradingAgentsGraph(
        selected_analyst_keys,
        config=config,
        debug=False,
    )

    ticker_results = []
    n = len(tickers)

    for idx, ticker in enumerate(tickers, 1):
        console.print()
        console.print(
            Panel(
                f"[bold cyan]Analyzing {ticker}[/bold cyan]  ({idx}/{n})\n"
                f"[dim]Date: {selections['analysis_date']}[/dim]",
                border_style="cyan",
                padding=(0, 2),
            )
        )

        with console.status(
            f"[bold green]Running agents for {ticker}...[/bold green]", spinner="dots"
        ):
            try:
                final_state, signal = graph.propagate(ticker, selections["analysis_date"])
            except Exception as e:
                console.print(f"[red]  Error analyzing {ticker}: {e}[/red]")
                ticker_results.append(
                    {
                        "ticker": ticker,
                        "signal": "ERROR",
                        "final_trade_decision": f"Analysis failed: {e}",
                    }
                )
                continue

        console.print(
            f"  [bold]Result:[/bold] "
            + (
                f"[green]{signal}[/green]"
                if signal == "BUY"
                else f"[red]{signal}[/red]"
                if signal == "SELL"
                else f"[yellow]{signal}[/yellow]"
            )
        )

        ticker_results.append(
            {
                "ticker": ticker,
                "signal": signal,
                "final_trade_decision": final_state.get("final_trade_decision", ""),
            }
        )

        # Optionally save individual report
        results_dir = (
            Path(config["results_dir"])
            / "portfolio"
            / selections["analysis_date"]
            / ticker
        )
        try:
            save_report_to_disk(final_state, ticker, results_dir)
        except Exception:
            pass  # Don't block portfolio analysis if individual save fails

    # Show per-ticker summary table
    console.print()
    console.print(Rule("Individual Results", style="bold cyan"))
    summary_table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box.SIMPLE_HEAD,
        padding=(0, 2),
    )
    summary_table.add_column("Ticker", style="cyan", justify="center")
    summary_table.add_column("Signal", justify="center")
    for r in ticker_results:
        sig = r["signal"]
        if sig == "BUY":
            sig_str = "[green]BUY[/green]"
        elif sig == "SELL":
            sig_str = "[red]SELL[/red]"
        elif sig == "HOLD":
            sig_str = "[yellow]HOLD[/yellow]"
        else:
            sig_str = f"[dim]{sig}[/dim]"
        summary_table.add_row(r["ticker"], sig_str)
    console.print(summary_table)

    # Filter out errored tickers before aggregation
    valid_results = [r for r in ticker_results if r["signal"] != "ERROR"]
    if not valid_results:
        console.print("[red]No successful analyses to aggregate.[/red]")
        raise typer.Exit(1)

    # Portfolio aggregation step
    console.print()
    console.print(Rule("Portfolio Aggregation", style="bold green"))
    with console.status("[bold green]Synthesizing portfolio action plan...[/bold green]", spinner="dots"):
        portfolio_plan = aggregate_portfolio_recommendations(
            graph.quick_thinking_llm,
            valid_results,
            selections["analysis_date"],
        )

    # Display the portfolio plan
    console.print()
    console.print(Panel(Markdown(portfolio_plan), title="Portfolio Action Plan", border_style="green", padding=(1, 2)))

    # Save portfolio plan
    save_choice = typer.prompt("\nSave portfolio report?", default="Y").strip().upper()
    if save_choice in ("Y", "YES", ""):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = Path.cwd() / "reports" / f"portfolio_{selections['analysis_date']}_{timestamp}"
        save_path_str = typer.prompt(
            "Save path (press Enter for default)", default=str(default_path)
        ).strip()
        save_path = Path(save_path_str)
        try:
            save_path.mkdir(parents=True, exist_ok=True)
            # Save portfolio plan
            (save_path / "portfolio_plan.md").write_text(
                f"# Portfolio Action Plan\n\nGenerated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                + portfolio_plan,
                encoding="utf-8",
            )
            # Save individual reports
            for r in ticker_results:
                if r["signal"] != "ERROR":
                    ticker_dir = save_path / r["ticker"]
                    ticker_dir.mkdir(exist_ok=True)
                    (ticker_dir / "final_decision.md").write_text(
                        f"# {r['ticker']} — {r['signal']}\n\n{r['final_trade_decision']}",
                        encoding="utf-8",
                    )
            console.print(f"\n[green]✓ Portfolio report saved to:[/green] {save_path.resolve()}")
            console.print(f"  [dim]Main report:[/dim] portfolio_plan.md")
        except Exception as e:
            console.print(f"[red]Error saving report: {e}[/red]")


@app.command(name="build-watchlist")
def build_watchlist(
    source: str = typer.Option("tickers.txt", "--source", "-s", help="Source tickers file (one ticker per line)"),
    output: str = typer.Option("watchlist.txt", "--output", "-o", help="Output watchlist file"),
    days: int = typer.Option(5, "--days", "-d", help="Number of business days ahead to look for earnings"),
    strategy: str = typer.Option("pre-earnings", "--strategy", help="Strategy: pre-earnings, earnings-reversal, fomc-fade, max-pain-friday"),
    trade_date: Optional[str] = typer.Option(None, "--trade-date", help="Reference date YYYY-MM-DD (default: today)"),
):
    """
    Build a focused watchlist for a given trading strategy.

    Strategies:
      pre-earnings       — tickers with earnings in the next N business days (default)
      earnings-reversal  — tickers that just reported and moved >5%
      fomc-fade          — broad-market ETFs after a big FOMC-day move
      max-pain-friday    — tickers far from options max pain heading into Friday expiry
    """
    import yfinance as yf
    import pandas as pd
    import numpy as np

    if strategy == "pre-earnings":
        _build_pre_earnings_watchlist(source, output, days)
    elif strategy == "earnings-reversal":
        _build_earnings_reversal_watchlist_cmd(source, output, trade_date)
    elif strategy == "fomc-fade":
        _build_fomc_fade_watchlist_cmd(output, trade_date)
    elif strategy == "max-pain-friday":
        _build_max_pain_friday_watchlist_cmd(source, output, trade_date)
    else:
        console.print(f"[red]Unknown strategy: {strategy}[/red]")
        console.print("[dim]Available: pre-earnings, earnings-reversal, fomc-fade, max-pain-friday[/dim]")
        raise typer.Exit(1)


def _build_pre_earnings_watchlist(source: str, output: str, days: int):
    """Original pre-earnings watchlist builder."""
    import yfinance as yf
    import pandas as pd

    tickers = load_tickers_from_file(source)

    console.print(
        Panel(
            f"[bold cyan]Earnings Watchlist Builder[/bold cyan]\n"
            f"[dim]Scanning {len(tickers)} tickers for earnings within the next {days} business days.[/dim]\n"
            f"[dim]Source: {source}  →  Output: {output}[/dim]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()

    today = pd.Timestamp.today().normalize()
    bdays = pd.bdate_range(start=today, periods=days + 1)
    window_end = bdays[-1]

    earnings_tickers = []
    errors = []
    skipped = 0

    with console.status("[bold green]Checking earnings dates...[/bold green]", spinner="dots") as status:
        for i, ticker in enumerate(tickers, 1):
            status.update(f"[bold green]Checking {ticker} ({i}/{len(tickers)})...[/bold green]")
            try:
                t = yf.Ticker(ticker)
                earnings_date = _get_next_earnings_date(t)
                if earnings_date is None:
                    skipped += 1
                    continue
                ed = pd.Timestamp(earnings_date).normalize()
                if today <= ed <= window_end:
                    earnings_tickers.append((ticker, ed.strftime("%Y-%m-%d")))
            except Exception as e:
                errors.append((ticker, str(e)))

    console.print()

    if earnings_tickers:
        result_table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE_HEAD, padding=(0, 2))
        result_table.add_column("Ticker", style="cyan", justify="center")
        result_table.add_column("Earnings Date", style="green", justify="center")
        for ticker, date in sorted(earnings_tickers, key=lambda x: x[1]):
            result_table.add_row(ticker, date)
        console.print(Panel(result_table, title=f"Tickers with Earnings in Next {days} Business Days", border_style="green"))
    else:
        console.print(f"[yellow]No tickers found with earnings in the next {days} business days.[/yellow]")

    console.print(f"\n[dim]Scanned: {len(tickers)} | Found: {len(earnings_tickers)} | No data: {skipped} | Errors: {len(errors)}[/dim]")

    if earnings_tickers:
        output_path = Path(output)
        lines = [
            f"# Earnings Watchlist",
            f"# Strategy: pre-earnings",
            f"# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Source: {source}",
            f"# Window: next {days} business days from {today.strftime('%Y-%m-%d')}",
            f"# Tickers: {len(earnings_tickers)}",
            "",
        ]
        for ticker, date in sorted(earnings_tickers, key=lambda x: x[1]):
            lines.append(f"{ticker}    # Earnings: {date}")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        console.print(f"\n[green]Watchlist saved to:[/green] {output_path.resolve()}")
    else:
        console.print("[dim]No output file written (no matches found).[/dim]")


def _build_earnings_reversal_watchlist_cmd(source: str, output: str, trade_date: Optional[str]):
    """Build watchlist of post-earnings overreactions."""
    from cli.strategies import build_earnings_reversal_watchlist

    tickers = load_tickers_from_file(source)
    console.print(
        Panel(
            f"[bold cyan]Earnings Reversal Watchlist[/bold cyan]\n"
            f"[dim]Scanning {len(tickers)} tickers for recent earnings with >5% moves.[/dim]",
            border_style="cyan", padding=(1, 2),
        )
    )

    with console.status("[bold green]Scanning for post-earnings overreactions...[/bold green]", spinner="dots"):
        results = build_earnings_reversal_watchlist(tickers, trade_date)

    if results:
        result_table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE_HEAD, padding=(0, 2))
        result_table.add_column("Ticker", style="cyan", justify="center")
        result_table.add_column("Earnings", style="green", justify="center")
        result_table.add_column("Move %", justify="right")
        result_table.add_column("Post-Close", justify="right")
        for ticker, edate, move, close in results:
            color = "red" if move < 0 else "green"
            result_table.add_row(ticker, edate, f"[{color}]{move:+.1f}%[/{color}]", f"${close:.2f}")
        console.print(Panel(result_table, title="Post-Earnings Overreactions", border_style="green"))

        output_path = Path(output)
        lines = [
            f"# Earnings Reversal Watchlist",
            f"# Strategy: earnings-reversal",
            f"# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Tickers: {len(results)}",
            "",
        ]
        for ticker, edate, move, close in results:
            lines.append(f"{ticker}    # Earnings: {edate} | Move: {move:+.1f}% | Close: ${close:.2f}")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        console.print(f"\n[green]Watchlist saved to:[/green] {output_path.resolve()}")
    else:
        console.print("[yellow]No post-earnings overreactions found.[/yellow]")
        console.print("[dim]No output file written.[/dim]")


def _build_fomc_fade_watchlist_cmd(output: str, trade_date: Optional[str]):
    """Build watchlist of ETFs after a big FOMC-day move."""
    from cli.strategies import build_fomc_fade_watchlist

    console.print(
        Panel(
            f"[bold cyan]FOMC Fade Watchlist[/bold cyan]\n"
            f"[dim]Checking if today is in an FOMC window and if markets moved.[/dim]",
            border_style="cyan", padding=(1, 2),
        )
    )

    with console.status("[bold green]Checking FOMC calendar and market moves...[/bold green]", spinner="dots"):
        results = build_fomc_fade_watchlist(trade_date)

    if results:
        result_table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE_HEAD, padding=(0, 2))
        result_table.add_column("ETF", style="cyan", justify="center")
        result_table.add_column("Move %", justify="right")
        result_table.add_column("FOMC Date", style="green", justify="center")
        for ticker, move, fdate in results:
            color = "red" if move < 0 else "green"
            result_table.add_row(ticker, f"[{color}]{move:+.2f}%[/{color}]", fdate)
        console.print(Panel(result_table, title="FOMC Fade Candidates", border_style="green"))

        output_path = Path(output)
        lines = [
            f"# FOMC Fade Watchlist",
            f"# Strategy: fomc-fade",
            f"# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Tickers: {len(results)}",
            "",
        ]
        for ticker, move, fdate in results:
            lines.append(f"{ticker}    # FOMC: {fdate} | Move: {move:+.2f}%")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        console.print(f"\n[green]Watchlist saved to:[/green] {output_path.resolve()}")
    else:
        console.print("[yellow]Not in an FOMC window or no significant moves detected.[/yellow]")
        console.print("[dim]No output file written.[/dim]")


def _build_max_pain_friday_watchlist_cmd(source: str, output: str, trade_date: Optional[str]):
    """Build watchlist of stocks far from max pain heading into Friday."""
    from cli.strategies import build_max_pain_friday_watchlist

    tickers = load_tickers_from_file(source)
    console.print(
        Panel(
            f"[bold cyan]Max Pain Friday Watchlist[/bold cyan]\n"
            f"[dim]Scanning {len(tickers)} tickers for options max pain divergence.[/dim]",
            border_style="cyan", padding=(1, 2),
        )
    )

    with console.status("[bold green]Calculating max pain levels...[/bold green]", spinner="dots"):
        results = build_max_pain_friday_watchlist(tickers, trade_date)

    if not results:
        import datetime as _dt
        today = _dt.date.fromisoformat(trade_date) if trade_date else _dt.date.today()
        if today.weekday() not in (2, 3):
            console.print(f"[yellow]Today is {today.strftime('%A')} — max pain strategy only runs on Wed/Thu.[/yellow]")
        else:
            console.print("[yellow]No tickers found with significant max pain divergence.[/yellow]")
        console.print("[dim]No output file written.[/dim]")
        return

    result_table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE_HEAD, padding=(0, 2))
    result_table.add_column("Ticker", style="cyan", justify="center")
    result_table.add_column("Price", justify="right")
    result_table.add_column("Max Pain", justify="right")
    result_table.add_column("Distance %", justify="right")
    for ticker, price, mp, dist in results:
        color = "red" if dist > 0 else "green"
        result_table.add_row(ticker, f"${price:.2f}", f"${mp:.2f}", f"[{color}]{dist:+.1f}%[/{color}]")
    console.print(Panel(result_table, title="Max Pain Friday Candidates", border_style="green"))

    output_path = Path(output)
    lines = [
        f"# Max Pain Friday Watchlist",
        f"# Strategy: max-pain-friday",
        f"# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Tickers: {len(results)}",
        "",
    ]
    for ticker, price, mp, dist in results:
        lines.append(f"{ticker}    # Price: ${price:.2f} | Max Pain: ${mp:.2f} | Distance: {dist:+.1f}%")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"\n[green]Watchlist saved to:[/green] {output_path.resolve()}")


def _get_next_earnings_date(ticker_obj):
    """
    Try multiple yfinance APIs to find the next upcoming earnings date.
    Returns a pandas Timestamp or None if not found.
    """
    import pandas as pd

    today = pd.Timestamp.today().normalize()

    # Method 1: ticker.calendar (dict with 'Earnings Date' key)
    try:
        cal = ticker_obj.calendar
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date")
            if dates:
                if not isinstance(dates, (list, tuple)):
                    dates = [dates]
                future = [pd.Timestamp(d) for d in dates if pd.Timestamp(d).normalize() >= today]
                if future:
                    return min(future)
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            if "Earnings Date" in cal.index:
                val = cal.loc["Earnings Date"].iloc[0] if hasattr(cal.loc["Earnings Date"], "iloc") else cal.loc["Earnings Date"]
                ts = pd.Timestamp(val)
                if ts.normalize() >= today:
                    return ts
    except Exception:
        pass

    # Method 2: ticker.earnings_dates DataFrame (future dates have NaN EPS)
    try:
        ed = ticker_obj.earnings_dates
        if ed is not None and not ed.empty:
            ed.index = pd.to_datetime(ed.index)
            future = ed[ed.index.normalize() >= today]
            if not future.empty:
                return future.index.min()
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Shared helper: run portfolio analysis programmatically (no interactive UI)
# ---------------------------------------------------------------------------

def run_portfolio_analysis_for_agent(tickers, graph_config, analyst_keys, analysis_date, strategy=None):
    """
    Core portfolio analysis loop — usable by both interactive CLI and paper trading.
    Returns (ticker_results, portfolio_plan_text, quick_thinking_llm).
    """
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    graph = TradingAgentsGraph(analyst_keys, config=graph_config, debug=False)
    ticker_results = []

    for ticker in tickers:
        try:
            final_state, signal = graph.propagate(ticker, analysis_date)
            ticker_results.append({
                "ticker": ticker,
                "signal": signal,
                "final_trade_decision": final_state.get("final_trade_decision", ""),
            })
        except Exception as e:
            ticker_results.append({
                "ticker": ticker,
                "signal": "ERROR",
                "final_trade_decision": str(e),
            })

    valid = [r for r in ticker_results if r["signal"] != "ERROR"]
    portfolio_plan = None
    if valid:
        portfolio_plan = aggregate_portfolio_recommendations(
            graph.quick_thinking_llm, valid, analysis_date, strategy=strategy
        )

    return ticker_results, portfolio_plan, graph.quick_thinking_llm


# ---------------------------------------------------------------------------
# init-paper-trading
# ---------------------------------------------------------------------------

@app.command(name="init-paper-trading")
def init_paper_trading(
    config_file: str = typer.Option("agent_configs.yaml", "--config", "-c", help="Path to agent_configs.yaml"),
):
    """
    Initialise the paper trading database from agent_configs.yaml.
    Creates paper_trading.db and registers all configured agents.
    Safe to re-run — existing agents are updated, balances are preserved.
    """
    from paper_trading.database import init_db, register_agent, get_all_agents
    from paper_trading.config import load_agent_configs

    try:
        agents = load_agent_configs(config_file)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    init_db()

    for agent_cfg in agents:
        name = agent_cfg["name"]
        starting_balance = agent_cfg.get("starting_balance", 100000)
        register_agent(name, agent_cfg, starting_balance)
        console.print(f"  [green]✓[/green] Registered agent: [cyan]{name}[/cyan]  (${starting_balance:,.0f})")

    all_agents = get_all_agents()
    console.print(f"\n[bold green]Paper trading initialised.[/bold green] {len(all_agents)} agent(s) ready.")
    console.print("[dim]Run:[/dim] tradingagents paper-trade --file watchlist.txt --date YYYY-MM-DD")


# ---------------------------------------------------------------------------
# paper-trade
# ---------------------------------------------------------------------------

@app.command(name="paper-trade")
def paper_trade(
    file: str = typer.Option(..., "--file", "-f", help="Watchlist .txt file to analyse"),
    date: Optional[str] = typer.Option(None, "--date", "-d", help="Trade date YYYY-MM-DD (default: today)"),
    config_file: str = typer.Option("agent_configs.yaml", "--config", "-c", help="Agent configs YAML"),
    agents: Optional[str] = typer.Option(None, "--agents", "-a", help="Comma-separated agent names to run (default: all)"),
    strategy: Optional[str] = typer.Option(None, "--strategy", help="Strategy context: pre-earnings, earnings-reversal, fomc-fade, max-pain-friday"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Analyse and parse but do not write trades to DB"),
):
    """
    Run all paper trading agents against a watchlist for a given date.
    Executes trades at market-open prices and records daily snapshots.

    Use --agents to run a subset:
      tradingagents paper-trade -f watchlist.txt --agents claude-full-deep,gpt-earnings-fast

    Use --strategy to apply strategy-specific decision making:
      tradingagents paper-trade -f watchlist.txt --strategy earnings-reversal
    """
    from paper_trading.database import init_db, get_agent, get_cash, get_positions
    from paper_trading.account import Account
    from paper_trading.execution import get_open_prices, get_current_prices as gcp, execute_paper_trades
    from paper_trading.config import load_agent_configs, agent_to_graph_config, get_analyst_keys
    from cli.portfolio_aggregator import parse_portfolio_plan as ppp

    trade_date = date or datetime.datetime.now().strftime("%Y-%m-%d")

    tickers = load_tickers_from_file(file)
    if not tickers:
        console.print("[yellow]Watchlist is empty — nothing to trade.[/yellow]")
        raise typer.Exit(0)

    try:
        agent_cfgs = load_agent_configs(config_file)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    # Filter to requested agents if --agents is provided
    if agents:
        requested = {name.strip() for name in agents.split(",")}
        all_names = {cfg["name"] for cfg in agent_cfgs}
        unknown = requested - all_names
        if unknown:
            console.print(f"[yellow]Warning: Unknown agent(s) skipped: {', '.join(sorted(unknown))}[/yellow]")
            console.print(f"[dim]Available: {', '.join(sorted(all_names))}[/dim]")
        agent_cfgs = [cfg for cfg in agent_cfgs if cfg["name"] in requested]
        if not agent_cfgs:
            console.print("[red]No matching agents found. Exiting.[/red]")
            raise typer.Exit(1)

    init_db()

    console.print(
        Panel(
            f"[bold cyan]Paper Trade — {trade_date}[/bold cyan]\n"
            f"[dim]Tickers: {', '.join(tickers)}\n"
            f"Agents: {len(agent_cfgs)}  |  Dry run: {dry_run}[/dim]",
            border_style="cyan", padding=(1, 2),
        )
    )

    # Fetch closing prices once (shared across agents)
    console.print("\n[dim]Fetching closing prices...[/dim]")
    prices = get_open_prices(tickers, trade_date)
    missing = [t for t in tickers if t not in prices]
    if missing:
        console.print(f"[yellow]Warning: Could not fetch prices for: {', '.join(missing)}. Those tickers will be skipped.[/yellow]")
    if not prices:
        console.print("[red]No prices fetched at all — check yfinance connectivity. Skipping all agents.[/red]")
        raise typer.Exit(1)

    # Also get prices for any existing positions not in today's watchlist
    all_agent_tickers = set(tickers)
    for agent_cfg in agent_cfgs:
        name = agent_cfg["name"]
        if get_agent(name):
            for t in get_positions(name):
                all_agent_tickers.add(t)
    extra_prices = gcp(list(all_agent_tickers - set(tickers)))
    all_prices = {**extra_prices, **prices}  # open prices take precedence

    for agent_cfg in agent_cfgs:
        name = agent_cfg["name"]
        console.print(f"\n{'─'*60}")
        console.print(f"[bold cyan]Agent: {name}[/bold cyan]")

        if not get_agent(name):
            console.print(f"  [yellow]Not initialised — run init-paper-trading first. Skipping.[/yellow]")
            continue

        graph_config = agent_to_graph_config(agent_cfg)
        analyst_keys = get_analyst_keys(agent_cfg)

        # Run analysis
        with console.status(f"[green]Running analysis for {name}...[/green]", spinner="dots"):
            ticker_results, portfolio_plan, llm = run_portfolio_analysis_for_agent(
                tickers, graph_config, analyst_keys, trade_date, strategy=strategy
            )

        # Log per-ticker results so failures are visible
        errors = [r for r in ticker_results if r["signal"] == "ERROR"]
        if errors:
            for r in errors:
                console.print(f"  [red]ERROR on {r['ticker']}: {r['final_trade_decision'][:120]}[/red]")

        if not portfolio_plan:
            if not [r for r in ticker_results if r["signal"] != "ERROR"]:
                console.print("  [red]All tickers failed analysis — no portfolio plan possible.[/red]")
            else:
                console.print("  [yellow]No portfolio plan generated — skipping trades.[/yellow]")
            continue

        # Parse plan into structured actions
        with console.status("[green]Parsing trade actions...[/green]", spinner="dots"):
            actions = ppp(llm, portfolio_plan)

        if not actions:
            console.print("  [yellow]Could not parse trade actions from plan. Raw plan (first 200 chars):[/yellow]")
            console.print(f"  [dim]{portfolio_plan[:200]}[/dim]")
            continue

        console.print(f"  Parsed [bold]{len(actions)}[/bold] trade action(s).")

        if not dry_run:
            account = Account(name)
            trade_results = execute_paper_trades(account, actions, trade_date, prices)

            # Mark-to-market snapshot (use all_prices for existing positions)
            account.take_snapshot(trade_date, all_prices)

            # Display trade results
            trade_table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE_HEAD, padding=(0, 2))
            trade_table.add_column("Ticker", style="cyan", justify="center")
            trade_table.add_column("Action", justify="center")
            trade_table.add_column("Shares", justify="right")
            trade_table.add_column("Price", justify="right")
            trade_table.add_column("Total", justify="right")
            trade_table.add_column("Status", justify="center")

            for r in trade_results:
                action_str = r["action"]
                color = "green" if action_str == "BUY" else "red" if action_str == "SELL" else "yellow"
                shares_str = str(r.get("shares", "—"))
                price_str = f"${r['price']:.2f}" if "price" in r else "—"
                total_str = f"${r['total']:,.2f}" if "total" in r else "—"
                status_color = "green" if r["status"] == "OK" else "red" if r["status"] == "FAILED" else "dim"
                trade_table.add_row(
                    r["ticker"],
                    f"[{color}]{action_str}[/{color}]",
                    shares_str, price_str, total_str,
                    f"[{status_color}]{r['status']}[/{status_color}]",
                )
            console.print(trade_table)

            snap = account.take_snapshot.__func__ if False else None
            from paper_trading.database import get_latest_snapshot
            snap = get_latest_snapshot(name)
            if snap:
                daily = f"{snap['daily_return_pct']:+.2f}%" if snap.get("daily_return_pct") is not None else "—"
                console.print(
                    f"  Portfolio: [bold]${snap['portfolio_value']:,.2f}[/bold]  "
                    f"Cash: ${snap['cash']:,.2f}  "
                    f"Day: {daily}  "
                    f"Total: [bold]{snap['cumulative_return_pct']:+.2f}%[/bold]"
                )
        else:
            console.print("  [dim](dry run — no trades written)[/dim]")
            for a in actions:
                console.print(f"    {a['ticker']:6s}  {a['action']:4s}  conviction={a.get('conviction','?')}")

    console.print(f"\n[bold green]Paper trade run complete for {trade_date}.[/bold green]")
    console.print("[dim]Run:[/dim] tradingagents leaderboard")


# ---------------------------------------------------------------------------
# leaderboard
# ---------------------------------------------------------------------------

@app.command(name="leaderboard")
def leaderboard():
    """Show all paper trading agents ranked by cumulative return."""
    from paper_trading.database import init_db, get_leaderboard

    init_db()
    rows = get_leaderboard()

    if not rows:
        console.print("[yellow]No agents found. Run init-paper-trading first.[/yellow]")
        raise typer.Exit(0)

    console.print()
    console.print(Rule("Paper Trading Leaderboard", style="bold green"))

    lb_table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE_HEAD, padding=(0, 2))
    lb_table.add_column("#", justify="center", width=4)
    lb_table.add_column("Agent", style="cyan")
    lb_table.add_column("Portfolio Value", justify="right")
    lb_table.add_column("Cash", justify="right")
    lb_table.add_column("Positions", justify="right")
    lb_table.add_column("Day Return", justify="right")
    lb_table.add_column("Total Return", justify="right")
    lb_table.add_column("Last Updated", justify="center")

    for i, row in enumerate(rows, 1):
        cumulative = row.get("cumulative_return_pct") or 0.0
        daily = row.get("daily_return_pct")
        cum_color = "green" if cumulative >= 0 else "red"
        day_color = "green" if (daily or 0) >= 0 else "red"
        lb_table.add_row(
            str(i),
            row["name"],
            f"${row['portfolio_value']:,.2f}",
            f"${row['cash']:,.2f}",
            f"${row['positions_value']:,.2f}",
            f"[{day_color}]{daily:+.2f}%[/{day_color}]" if daily is not None else "—",
            f"[{cum_color}]{cumulative:+.2f}%[/{cum_color}]",
            row.get("last_updated") or "never",
        )

    console.print(lb_table)


# ---------------------------------------------------------------------------
# agent-history
# ---------------------------------------------------------------------------

@app.command(name="agent-history")
def agent_history(
    name: str = typer.Argument(..., help="Agent name to inspect"),
):
    """Show full trade history and daily P&L for a paper trading agent."""
    from paper_trading.database import init_db, get_agent, get_all_trades, get_all_snapshots, get_positions, get_cash

    init_db()
    agent = get_agent(name)
    if not agent:
        console.print(f"[red]Agent '{name}' not found.[/red]")
        raise typer.Exit(1)

    import json as _json
    cfg = _json.loads(agent["config"])

    console.print()
    console.print(Rule(f"Agent: {name}", style="bold cyan"))
    console.print(f"  Provider: [cyan]{cfg.get('llm_provider')}[/cyan]  "
                  f"Deep: {cfg.get('deep_think_llm')}  "
                  f"Quick: {cfg.get('quick_think_llm')}")
    console.print(f"  Analysts: {', '.join(cfg.get('analysts', []))}  "
                  f"Thinking: {cfg.get('google_thinking_level') or 'off'}")
    console.print(f"  Starting balance: ${agent['starting_balance']:,.2f}")

    # Current state
    cash = get_cash(name)
    positions = get_positions(name)
    console.print(f"  Cash on hand: ${cash:,.2f}")
    if positions:
        pos_str = ", ".join(f"{t} x{p['shares']:.0f}" for t, p in positions.items())
        console.print(f"  Open positions: {pos_str}")
    console.print()

    # Daily snapshots
    snapshots = get_all_snapshots(name)
    if snapshots:
        console.print(Rule("Daily P&L", style="dim"))
        snap_table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE_HEAD, padding=(0, 2))
        snap_table.add_column("Date", justify="center")
        snap_table.add_column("Portfolio Value", justify="right")
        snap_table.add_column("Cash", justify="right")
        snap_table.add_column("Positions", justify="right")
        snap_table.add_column("Day Return", justify="right")
        snap_table.add_column("Cumulative", justify="right")
        for s in snapshots:
            daily = s.get("daily_return_pct")
            cum = s.get("cumulative_return_pct", 0)
            day_color = "green" if (daily or 0) >= 0 else "red"
            cum_color = "green" if cum >= 0 else "red"
            snap_table.add_row(
                s["date"],
                f"${s['portfolio_value']:,.2f}",
                f"${s['cash']:,.2f}",
                f"${s['positions_value']:,.2f}",
                f"[{day_color}]{daily:+.2f}%[/{day_color}]" if daily is not None else "—",
                f"[{cum_color}]{cum:+.2f}%[/{cum_color}]",
            )
        console.print(snap_table)
        console.print()

    # Trade log
    trades = get_all_trades(name)
    if trades:
        console.print(Rule("Trade History", style="dim"))
        trade_table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE_HEAD, padding=(0, 2))
        trade_table.add_column("Date", justify="center")
        trade_table.add_column("Ticker", justify="center", style="cyan")
        trade_table.add_column("Action", justify="center")
        trade_table.add_column("Shares", justify="right")
        trade_table.add_column("Price", justify="right")
        trade_table.add_column("Total", justify="right")
        trade_table.add_column("Conviction", justify="right")
        for t in trades:
            color = "green" if t["action"] == "BUY" else "red" if t["action"] == "SELL" else "yellow"
            conv = t.get("conviction_weight")
            trade_table.add_row(
                t["date"], t["ticker"],
                f"[{color}]{t['action']}[/{color}]",
                f"{t['shares']:.0f}",
                f"${t['price']:.2f}",
                f"${t['total_value']:,.2f}",
                f"{conv:.0%}" if conv is not None else "—",
            )
        console.print(trade_table)
    else:
        console.print("[dim]No trades recorded yet.[/dim]")


if __name__ == "__main__":
    app()
