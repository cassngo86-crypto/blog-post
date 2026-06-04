import streamlit as st
import os
import time
import warnings
import litellm
from crewai import Agent, Task, Crew, LLM
# 1. IMPORT THE SEARCH TOOL
from crewai_tools import SerperDevTool
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List

# Disable warnings
warnings.filterwarnings("ignore")


# Define the rigid structure your blog post must fulfill
class StructuredArticle(BaseModel):
    title: str = Field(description="The catchy, SEO-optimized title of the article.")
    introduction: str = Field(description="The introductory section. Must include an embedded blockquote callout for a core metric or trend.")
    comparative_table_markdown: str = Field(description="A fully formatted Markdown table comparing 3+ core entities (e.g., flight options, software tools, or architectural patterns) with descriptive context hyperlinks embedded inside the table cells.")
    body_sections_markdown: str = Field(description="The main editorial content sections separated by clean markdown H3 headers. Each section must have 2-3 detailed paragraphs containing descriptive, context-anchored links.")
    conclusion: str = Field(description="Summarizing wrap-up section and a clear call to action.")
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

   
    # --- TASKS WITH ANCHORED SEARCH CONTROLS & GROQ GUARDRAILS ---
    current_year = datetime.now().strftime("%Y")

    plan_description = (
        f"1. Identify trends, structural shifts, and key authoritative entities on {{topic}}.\n"
        f"2. Create a clean article outline highlighting where comparative tables or metric callouts would maximize readability.\n"
        f"3. CRITICAL SEARCH RULE: When searching for data on {{topic}}, target the **current year ({current_year})**.\n"
        f"   Always prioritize official documentation, primary source sites, and corporate/government announcements from {current_year}.\n"
        f"4. GROQ TOOL USE RULE: You must execute exactly ONE search query at a time. Do not attempt to call multiple functions or merge "
        f"text notes directly into your tool invocation blocks. Wait for the response before executing another search."
    )

    plan = Task(
        description=plan_description,
        expected_output=f"An authoritative outline document with verified data notes from {current_year}.",
        agent=planner,
        cache=False # <-- CRITICAL: Prevents Groq from choking on cached multi-tool formatting structures
    )

    write = Task(
        description=(
            f"1. Convert the content plan into a premium article on {{topic}} in a {tone_setting} voice.\n"
            f"2. VISUAL READABILITY RULES:\n"
            f"   - Break long prose blocks by introducing a Markdown Table if comparing 2 or more entities (e.g., flight options, software tools, or patterns).\n"
            f"   - Use Markdown Blockquotes (`>`) to emphasize critical takeaways, warnings, or expert metrics.\n"
            f"   - Ensure each core section header contains exactly 2 to 3 well-paced paragraphs."
        ),
        expected_output="A visually engaging, magazine-ready blog post in markdown format with integrated tables and blockquotes.",
        agent=writer,
    )

    edit = Task(
        description=(
            "Review and refine the blog post. Ensure it meets journalistic standards, balances tone consistency, "
            "and verifies that markdown tables or blockquotes are syntactically flawless."
        ),
        expected_output="A polished version of the blog post in markdown format.",
        agent=editor
    )

    enrich_links = Task(
        description=(
            "Locate official bodies, software repositories, tools, or major platforms mentioned in the text. "
            "Convert these references into clickable Markdown hyperlinks. "
            "SEO ANCHOR RULE: Do not link single words like '[Skyscanner]'. Instead, bind the link to context-aware descriptive "
            "action phrases (e.g., '[compare flight routes via Skyscanner](https://www.skyscanner.com)' or '[review the official LangChain Framework](url)'). "
            "Keep the core markdown tables, blockquotes, and text completely unaltered."
        ),
        expected_output="The final markdown blog post containing highly descriptive, context-anchored navigation hyperlinks.",
        agent=linker,
        output_pydantic=StructuredArticle
    )

    # Assemble Crew with strict rate-limit protection
    crew = Crew(
        agents=[planner, writer, editor, linker],
        tasks=[plan, write, edit, enrich_links],
        verbose=True,
        max_rpm=1  
    )

    # Kickoff the execution
    # Execute the Crew
    result = crew.kickoff(inputs={"topic": topic})
    
    # Extract our strongly-typed Pydantic structure
    article_data = result.pydantic
    
    # Assemble the pieces into a stunning, professional markdown layout
    assembled_markdown = f"""# {article_data.title}

### Introduction
{article_data.introduction}

### Comparative Overview & Key Resources
{article_data.comparative_table_markdown}

{article_data.body_sections_markdown}

### Conclusion
{article_data.conclusion}
"""
    return assembled_markdown


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
                # Call the cached helper function which now returns our beautifully structured layout
                final_text = run_cached_crew(topic, tone, temperature, groq_api_key, serper_api_key)
                
                status_box.empty()
                st.success("🎉 Visually optimized article ready!")
                st.subheader("📝 Generated Blog Post")
                
                # Render beautifully to screen with tables and blockquotes
                st.markdown(final_text)
                
                st.download_button(
                    label="📥 Download Production-Grade Article",
                    data=final_text,
                    file_name=f"{topic.lower().replace(' ', '_')}_production.md",
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