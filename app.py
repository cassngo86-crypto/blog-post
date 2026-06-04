import streamlit as st
import os
import time
import warnings
import litellm
from crewai import Agent, Task, Crew, LLM
# 1. IMPORT THE SEARCH TOOL
from crewai_tools import SerperDevTool

# Disable warnings
warnings.filterwarnings("ignore")

# Global Monkey Patch for LiteLLM to prevent structural rejections
original_completion = litellm.completion

def safe_completion(*args, **kwargs):
    if "extra_body" in kwargs and isinstance(kwargs["extra_body"], dict):
        kwargs["extra_body"].pop("cache_breakpoint", None)
        if not kwargs["extra_body"]:
            kwargs.pop("extra_body")
            
    if "messages" in kwargs and isinstance(kwargs["messages"], list):
        for msg in kwargs["messages"]:
            if isinstance(msg, dict):
                msg.pop("cache_breakpoint", None)
                
    kwargs.pop("cache_breakpoint", None)
    return original_completion(*args, **kwargs)

litellm.completion = safe_completion

# --- STREAMLIT UI & CONFIGURATIONS ---
st.set_page_config(page_title="Advanced AI Content Crew", page_icon="🚀", layout="centered")

st.title("🚀 Advanced Multi-Agent Content Crew")
st.write("An enhanced system utilizing live web search, parameter tuning, and intelligent caching.")

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("🔑 API Credentials")
groq_api_key = os.environ.get("GROQ_API_KEY") or st.sidebar.text_input("Groq API Key", type="password")
serper_api_key = os.environ.get("SERPER_API_KEY") or st.sidebar.text_input("Serper API Key (For Live Search)", type="password")

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Generation Settings (Enhancement 2)")

# Dynamic Pacing Control
tone = st.sidebar.selectbox(
    "Select Writing Tone",
    ["Professional", "Casual/Conversational", "Technical & Academic", "SEO-Optimized Marketing"]
)

temperature = st.sidebar.slider(
    "LLM Creativity (Temperature)",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1,
    help="Lower values are more deterministic and factual; higher values are more creative."
)

if not groq_api_key:
    st.info("🔑 Please enter your Groq API Key in the sidebar to begin.")
    st.stop()


# --- CACHED CREW EXECUTION FUNCTION (Enhancement 3) ---
# This decorator ensures identical topics + settings don't waste your Groq tokens.
@st.cache_data(show_spinner=False)
def run_cached_crew(topic, tone_setting, temp_setting, _groq_key, _serper_key):
    
    # Instantiate Search Tool if key is present
    search_tool = None
    if _serper_key:
        os.environ["SERPER_API_KEY"] = _serper_key
        search_tool = SerperDevTool()

    # Define LLMs with user-configured temperature
    fast_llm = LLM(
        model="groq/llama-3.1-8b-instant",
        api_key=_groq_key,
        temperature=temp_setting
    )

    smart_llm = LLM(
        model="groq/llama-3.3-70b-versatile",
        api_key=_groq_key,
        temperature=temp_setting
    )

    # --- AGENTS (Dynamically tracking the chosen 'tone') ---
    planner = Agent(
        role="Content Research Planner",
        goal=f"Plan factually accurate content architecture on {{topic}} using a {tone_setting} approach.",
        backstory="You extract core structural points, note high-profile sub-concepts, and search for live, accurate data points.",
        allow_delegation=False,
        verbose=True,
        llm=fast_llm,
        tools=[search_tool] if search_tool else [] # Injects Live Search into the Planning phase
    )

    writer = Agent(
        role="Content Writer",
        goal=f"Write insightful articles about the topic: {{topic}} matching a {tone_setting} tone perfectly.",
        backstory=f"You write premium, multi-paragraph copy. You maintain a strictly {tone_setting} narrative style.",
        allow_delegation=False,
        verbose=True,
        llm=smart_llm
    )

    editor = Agent(
        role="Editor",
        goal="Edit a given blog post to align with professional guidelines and formatting rules.",
        backstory=f"You adjust syntax, pacing, and verify the post strictly embodies a {tone_setting} delivery.",
        allow_delegation=False,
        verbose=True,
        llm=smart_llm
    )

    linker = Agent(
        role="Digital Link Optimizer",
        goal="Read a completed text draft and format references into standard Markdown links dynamically.",
        backstory="You convert high-level organizations, cities, or applications mentioned into valid hyperlinks.",
        allow_delegation=False,
        verbose=True,
        llm=fast_llm
    )

    # --- TASKS ---
    plan_description = "1. Identify trends and recent facts on {topic}.\n2. Create a clean article outline."
    if search_tool:
        plan_description += "\n3. Use your search tool to gather the latest real-time news or data on this topic."

    plan = Task(
        description=plan_description,
        expected_output="An outline document with accompanying recent resource data notes.",
        agent=planner,
    )

    write = Task(
        description=f"1. Convert the content plan into a blog post.\n2. Structure with markdown headers in a {tone_setting} voice.",
        expected_output="A comprehensive markdown blog post where each section contains 2 or 3 detailed paragraphs.",
        agent=writer,
    )

    edit = Task(
        description="Review and refine the blog post written by the writer for structural balance and grammar.",
        expected_output="A polished version of the blog post in markdown format.",
        agent=editor
    )

    enrich_links = Task(
        description="Locate official bodies or industry-standard platforms mentioned and add markdown links without altering core text text.",
        expected_output="The complete markdown blog post containing navigation hyperlinks.",
        agent=linker
    )

    # Assemble Crew with strict rate-limit protection
    crew = Crew(
        agents=[planner, writer, editor, linker],
        tasks=[plan, write, edit, enrich_links],
        verbose=True,
        max_rpm=1  
    )

    # Kickoff the execution
    result = crew.kickoff(inputs={"topic": topic})
    return result.raw if hasattr(result, 'raw') else str(result)


# --- MAIN INTERFACE ---
topic = st.text_input("What topic would you like the agents to handle today?", placeholder="e.g., Current Travel Trends Singapore to Tokyo")

if st.button("Launch Crew Execution", type="primary"):
    if not topic.strip():
        st.warning("Please provide a valid topic.")
    else:
        status_box = st.empty()
        
        # We try 3 times to manage potential API cooldown limits
        max_retries = 3
        
        for attempt in range(max_retries):
            status_box.info(f"🕵️‍♂️ Fetching content for '{topic}' using a {tone} tone... (Checking Cache / Pacing Agents)")
            try:
                # Call the cached helper function
                final_text = run_cached_crew(topic, tone, temperature, groq_api_key, serper_api_key)
                
                status_box.empty()
                st.success("🎉 Article ready!")
                st.subheader("📝 Generated Blog Post")
                st.markdown(final_text)
                
                st.download_button(
                    label="📥 Download Enriched Article",
                    data=final_text,
                    file_name=f"{topic.lower().replace(' ', '_')}_advanced.md",
                    mime="text/markdown"
                )
                break
                
            except Exception as e:
                error_msg = str(e)
                if "rate_limit_exceeded" in error_msg.lower() or "ratelimiterror" in error_msg.lower():
                    wait_time = 15.0
                    if "try again in" in error_msg:
                        try:
                            parts = error_msg.split("try again in ")
                            wait_time = float(parts[1].split("s")[0]) + 1.5
                        except:
                            pass
                    status_box.warning(f"⏳ Dynamic Rate-limit pause triggered. Resuming in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    status_box.error(f"Execution Error: {e}")
                    break
        else:
            status_box.error("❌ Failed to clear the server queue. Please try again in 1 minute.")