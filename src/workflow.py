from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.agents import planner_node, research_node, analyst_node, writer_node, critic_node

def should_continue(state: AgentState):
    """Routing logic after Critics evaluation."""
    score = state.get("critique_score", 0)
    iterations = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 1) # Reduce max iterations for speed
    
    if score >= 8 or iterations >= max_iterations:
        print(f"\n[Router] Workflow complete. Final Score: {score}/10.")
        return END
    else:
        print(f"\n[Router] Report rejected (score {score}/10). Resetting for improvement.")
        return "Planner"

def get_workflow():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("Planner", planner_node)
    workflow.add_node("Researcher", research_node)
    workflow.add_node("Analyst", analyst_node)
    workflow.add_node("Writer", writer_node)
    workflow.add_node("Critic", critic_node)

    # Define Linear Edges for maximum speed
    workflow.set_entry_point("Planner")
    
    workflow.add_edge("Planner", "Researcher")
    workflow.add_edge("Researcher", "Analyst")
    workflow.add_edge("Analyst", "Writer")
    workflow.add_edge("Writer", "Critic")

    # Final Conditional Edge (Keep one feedback loop but limit it)
    workflow.add_conditional_edges(
        "Critic",
        should_continue,
        {
            "Planner": "Planner",
            END: END
        }
    )

    return workflow.compile()
