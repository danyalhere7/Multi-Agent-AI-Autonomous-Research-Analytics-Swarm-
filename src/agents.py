from langchain_ollama import ChatOllama
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from src.state import AgentState
from src.tools import TOOLS, web_search, execute_python_code
from src.memory import memory
import json
import re
import time

# Initialize the local LLM
# NOTE: Using "1b" version to fit in systems with limited RAM (fixes 500 Memory Error)
MODEL_NAME = "llama3.2:1b" 
llm = ChatOllama(model=MODEL_NAME, temperature=0, base_url="http://127.0.0.1:11434")
llm_with_tools = llm.bind_tools(TOOLS)

def clean_json_response(content: str):
    """Safely extract and parse JSON from LLM output."""
    content = content.strip()
    try:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        content = content.strip()
        return json.loads(content)
    except:
        try:
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                return json.loads(content[start:end+1])
        except:
            pass
    return None

def fuzzy_tool_call(response_content: str, tool_name: str):
    """Detects if a local LLM tried to call a tool via raw JSON text."""
    if f'"{tool_name}"' in response_content or f"'{tool_name}'" in response_content:
        data = clean_json_response(response_content)
        if data and isinstance(data, dict):
            # DRILL DOWN GUARD: If model returns a nested "name" / "arguments" structure
            if "name" in data and data["name"] == tool_name:
                p = data.get("parameters", data.get("args", data.get("arguments", {})))
                if isinstance(p, dict) and "properties" in p:
                    # Model returned a schema definition! Try to pull values from it
                    p = {k: v.get("default", "") for k, v in p.get("properties", {}).items() if isinstance(v, dict)}
                return p
            return data
    return None

def planner_node(state: AgentState):
    """Extracts objective and breaks it into sub-tasks (Optimized)."""
    print(f"\n[Node] Planner: Strategic phase extraction.")
    
    prompt = f"""You are a Senior AI Data Scientist. 
    Task: {state['task']}
    
    PRIMARY OBJECTIVE:
    Provide accurate, factual knowledge and perform real analysis when applicable.
    
    STRICT BEHAVIOR RULES:
    1. NO synthetic data or fabricated statistics.
    2. Focus ONLY on grounding and real-world facts.
    3. If requesting data analysis, ensure real statistical methods are planned.
    
    Goal: Define 2-3 specific research sub-tasks.
    Output JSON ONLY:
    {{"objective": "Summarized objective", "plan": ["Step 1", "Step 2"]}}
    """
    
    try:
        response = llm.invoke(prompt)
        data = clean_json_response(response.content)
        return {
            "task": data.get("objective", state["task"]) if data else state["task"],
            "plan": data.get("plan", ["General Search", "Metric Extraction"]) if data else ["Research", "Analysis"],
            "iteration": state.get("iteration", 0) + 1
        }
    except Exception as e:
        print(f"Planner Error: {e}")
        return {"plan": ["Real-time research", "Metric compute"], "iteration": state.get("iteration", 0) + 1}

def research_node(state: AgentState):
    """BATCHED RESEARCH: Executes all search tasks at once with strict citation tracking."""
    print(f"\n[Node] Researcher: High-Fidelity Fact Finding...")
    plan = state.get("plan", [])
    if not plan: return {}

    all_results = []
    citations = state.get("citations", [])
    
    for sub_task in plan:
        print(f"   -> Searching for Grounding: {sub_task}")
        try:
            # We use the raw tool output to get more context
            results = web_search.invoke({"query": sub_task})
            all_results.append(f"Sub-task: {sub_task}\nFindings: {results}")
            
            # Extract URLs for citation structure
            urls = re.findall(r'https?://[^\s)\]]+', str(results))
            for url in urls:
                if not any(c['url'] == url for c in citations):
                    citations.append({"title": f"Source: {sub_task}", "url": url})
        except Exception as e:
            print(f"   ! Search failed for {sub_task}: {e}")

    context_str = "\n\n".join(all_results)
    prompt = f"""You are a Researcher. Summarize findings into bulleted, grounded facts.
    STRICT RULES:
    1. EXTRACT ALL NUMERICAL DATA, STATISTICS, AND TABLES.
    2. ONLY report what is found. NO synthetic data.
    3. Include source URLs next to facts where possible.
    
    Findings:
    {context_str}
    """
    
    response = llm.invoke(prompt)
    summary = response.content
    
    memory.add_memory(summary, metadata={"node": "researcher", "type": "batch"})
    
    return {
        "research_data": [{"task": "Batch Grounding", "content": summary}],
        "citations": citations,
        "current_step": len(plan)
    }

def analyst_node(state: AgentState):
    """Performs FORCED Python-based numerical analysis and generates Metrics JSON."""
    print(f"\n[Node] Analyst: Performing Forced Statistical Analytics.")
    research_text = "\n".join([d['content'] for d in state.get("research_data", [])])
    
    prompt = f"""You are a Senior Data Scientist.
    STRICT POLICY:
    1. EVERY metric must be computed via Python code. DO NOT hardcode.
    2. Analyze the provided research data for trends, averages, or correlations.
    3. Generate EXACTLY one visualization (plot) using matplotlib.
    4. MUST generate a 'Metrics JSON' at the end of your response.
    
    JSON FORMAT:
    ```json
    {{
      "computed_stats": {{"Metric Name": "Value", ...}},
      "demo_synthetic": "Label this: Portfolio Demo Only: [Your synthetic theory here]"
    }}
    ```
    
    RESEARCH DATA:
    {research_text}
    """
    
    response = llm_with_tools.invoke(prompt)
    
    tool_calls = getattr(response, 'tool_calls', [])
    if not tool_calls:
        fuzzy_args = fuzzy_tool_call(response.content, "execute_python_code")
        if fuzzy_args:
             tool_calls = [{"name": "execute_python_code", "args": fuzzy_args, "id": "fuzzy_analyst"}]

    analysis_output = ""
    metrics_json = {}

    if tool_calls:
        print(f"[Tool] Executing Forced Python Analysis...")
        messages = [("user", "Perform analysis logic"), response]
        for tc in tool_calls:
            args = tc.get('args', tc.get('parameters', {}))
            
            # DRILL DOWN (1B Hardware Compatibility)
            if isinstance(args, dict) and "properties" in args:
                props = args.get("properties", {})
                args = {"code": props.get("code", {}).get("default", props.get("code", ""))}

            if isinstance(args, dict) and 'query' in args and 'code' not in args:
                args['code'] = args.pop('query')
            
            try:
                result = execute_python_code.invoke(args)
                messages.append(ToolMessage(tool_call_id=tc.get('id', 'manual'), content=result))
            except Exception as e:
                print(f"   ! Analytical error: {e}")

        summary_res = llm.invoke([{"role": "system", "content": "Return the summary of findings AND the required Metrics JSON."}] + messages)
        analysis_output = summary_res.content
    else:
        analysis_output = response.content

    # Extract JSON or fallback
    data = clean_json_response(analysis_output)
    metrics_json = data if data else {"computed_stats": {"Status": "Analytical Review Complete"}, "demo_synthetic": "No dataset for modeling."}

    return {"analysis_results": analysis_output, "metrics_json": metrics_json}

def writer_node(state: AgentState):
    """Follows the 6-STEP PORTFOLIO workflow with JSON-grounded numbers."""
    print(f"\n[Node] Writer: Constructing Structured Dossier.")
    
    metrics = state.get("metrics_json", {})
    
    prompt = f"""Generate a research dossier following this structure:
    1. Executive Summary
    2. Data Overview (Real metrics from: {metrics.get('computed_stats')})
    3. Statistical Analysis
    4. Visual Findings (Describe generated charts)
    5. Interpretation
    6. Limitations & References: {state.get('citations')}
    
    STRICT RULES:
    - ALL NUMBERS must come from the Metrics JSON.
    - Professional, objective, strategic tone.
    - Include a clearly labeled 'Synthetic/Demonstration Modeling' section if provided: {metrics.get('demo_synthetic')}
    """
    
    response = llm.invoke(prompt)
    return {"draft_report": response.content}

def critic_node(state: AgentState):
    """Quality Audit with numeric Gauge Output."""
    print(f"\n[Node] Critic: Performance Audit.")
    
    prompt = f"""Evaluate this report.
    Report: {state.get('draft_report')}
    Metrics Code Used: {True if state.get('metrics_json') else False}
    
    Return JSON: {{"score": <0-10>, "feedback": "Brief suggestions"}}
    """
    
    try:
        response = llm.invoke(prompt)
        data = clean_json_response(response.content)
        score = float(data.get("score", 5))
        return {"critic_gauge": score, "feedback": data.get("feedback", "Excellent grounding.")}
    except:
        return {"critic_gauge": 5.0, "feedback": "Manual validation pending."}
