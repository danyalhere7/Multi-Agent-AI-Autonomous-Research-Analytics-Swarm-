from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    task: str # Original objective
    plan: List[str] # List of sub-tasks
    current_step: int # Current sub-task index
    research_data: List[Dict[str, Any]] # Findings with URLs and metrics
    analysis_results: str # Analytical insights
    metrics_json: Dict[str, Any] # Structured JSON for Plotly/Metrics
    citations: List[Dict[str, str]] # List of {title, url}
    draft_report: str # Final report content
    critic_gauge: float # Score out of 10
    feedback: str # Critic's improvement instructions
    portfolio_data: List[Dict[str, Any]] # Archive of past reports
    iteration: int # Current loop count
    max_iterations: int # Loop limit
