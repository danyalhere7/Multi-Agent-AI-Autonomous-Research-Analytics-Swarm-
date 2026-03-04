import os
import sys

# Add the project root to the python path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.workflow import get_workflow
from src.state import AgentState

def print_step(event):
    for node, state in event.items():
        if node == "StepIncrement":
            continue
            
        print(f"\n{'='*50}")
        print(f"Node Executed: {node}")
        print(f"{'='*50}")
        
        if node == "Planner":
            print(f"Extracted Objective: {state.get('task', 'N/A')}")
            print("Current Plan:")
            for i, p in enumerate(state.get("plan", [])):
                print(f"  {i+1}. {p}")
        
        if node == "Researcher":
            data = state.get("research_data", [])
            if data:
                last = data[-1]
                print(f"Findings for: {last.get('task')}")
                content = last.get('content', '')
                print(f"Content Snippet: {content[:300]}...")
                print(f"Citations Found: {len(state.get('citations', []))}")

        if node == "Analyst":
            results = state.get("analysis_results", "")
            print(f"Analysis Results Snippet: {results[-300:]}")
            
        if node == "Writer":
            print(f"Draft Report Generated! Length: {len(state.get('draft_report', ''))} characters.")
            
        if node == "Critic":
            print(f"Critic Score: {state.get('critique_score', 0)}/10")
            print(f"Critic Feedback: {state.get('feedback', 'No feedback provided.')}")

def main():
    print("Initializing Multi-Agent System Upgrade...")
    workflow = get_workflow()
    
    # Initialize State
    initial_state: AgentState = {
        "task": "Analyze AI job market trends in 2025 and generate a structured report.",
        "plan": [],
        "current_step": 0,
        "research_data": [],
        "analysis_results": "",
        "metrics": {},
        "citations": [],
        "draft_report": "",
        "critique_score": 0,
        "feedback": "",
        "iteration": 0,
        "max_iterations": 3
    }

    print(f"\nStarting Task: '{initial_state['task']}'")
    print("Streaming events...\n")

    # Run the graph
    current_state = initial_state
    try:
        for event in workflow.stream(initial_state):
            print_step(event)
            # Merge the event delta into our current_state
            for node, next_state in event.items():
                current_state.update(next_state)
    except Exception as e:
        print(f"\n[Error] Execution failed: {e}")

    print("\n\n" + "="*50)
    print("FINAL REPORT")
    print("="*50 + "\n")
    final_report = current_state.get("draft_report", "No report generated.")
    print(final_report)
    
    if current_state.get("citations"):
        print("\n" + "="*50)
        print("SOURCES & CITATIONS")
        print("="*50)
        for url in current_state["citations"]:
            print(f"- {url}")

if __name__ == "__main__":
    main()
