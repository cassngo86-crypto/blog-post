import streamlit as st
import os
import time
import warnings
import litellm
from crewai import Agent, Task, Crew, LLM

# Disable warnings
warnings.filterwarnings("ignore")

# 1. Global Monkey Patch for LiteLLM to prevent structural rejections
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

# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Agent Content Creator", page_icon="✍️", layout="centered")

st.title("✍️ Multi-Agent Content Generation Crew")
st.write("Input a topic below to watch your specialized AI agents plan, write, and edit articles consecutively.")

# Secure API Key Setup
groq_api_key = os.environ.get("GROQ_API_KEY") or st.sidebar.text_input("Enter Groq API Key", type="password")

if not groq_api_key:
    st.info("🔑 Please enter your Groq API Key in the sidebar or set the environment variable to begin.")
    st.stop()

# --- MODEL SELECTION ---
# We use Llama 3.1 8B for discovery and structural linking to conserve token headroom
fast_llm = LLM(
    model="groq/llama-3.1-8b-instant",
    api_key=groq_api_key,
    temperature=0.2
)

# We reserve the heavy-duty 70B model ONLY for premium writing and polishing
smart_llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=groq_api_key,
    temperature=0.7
)

topic = st.text_input("What topic would you like the agents to write about?", placeholder="e.g., Quantum Computing, Stock Market Basics")

if st.button("Launch Crew Execution", type="primary"):
    if not topic.strip():
        st.warning("Please provide a valid topic.")
    else:
        # Create clear status placeholders for the user during cool-off delays
        status_box = st.empty()
        
        # Track attempts for consecutive generation protection
        max_retries = 3
        retry_delay = 5  # Base delay in seconds
        
        for attempt in range(max_retries):
            status_box.info(f"🚀 Initializing agents for '{topic}' (Attempt {attempt + 1}/{max_retries})...")
            
            try:
                # --- AGENTS ---
                planner = Agent(
                    role="Content planner",
                    goal="Plan engaging and factually accurate content architecture on {topic}",
                    backstory="You extract core structural points and note high-profile sub-concepts.",
                    allow_delegation=False,
                    verbose=True,
                    llm=fast_llm
                )

                writer = Agent(
                    role="Content Writer",
                    goal="Write insightful opinion pieces about the topic: {topic}.",
                    backstory="You write multi-paragraph editorial copy.",
                    allow_delegation=False,
                    verbose=True,
                    llm=smart_llm
                )

                editor = Agent(
                    role="Editor",
                    goal="Edit a given blog post to align with professional guidelines.",
                    backstory="You adjust syntax and correct technical pacing bugs from drafts.",
                    allow_delegation=False,
                    verbose=True,
                    llm=smart_llm
                )

                linker = Agent(
                    role="Digital Link Optimizer",
                    goal="Read a completed text draft and format references into standard Markdown links dynamically.",
                    backstory="You convert high-level references into standard Markdown hyperlinks cleanly.",
                    allow_delegation=False,
                    verbose=True,
                    llm=fast_llm
                )

                # --- TASKS ---
                plan = Task(
                    description="1. Identify trends on {topic}.\n2. Create a clean article outline.",
                    expected_output="An outline document.",
                    agent=planner,
                )

                write = Task(
                    description="1. Convert the content plan into a blog post.\n2. Structure with markdown headers.",
                    expected_output="A blog post in markdown format where each section has 2 or 3 paragraphs.",
                    agent=writer,
                )

                edit = Task(
                    description="Review and refine the blog post written by the writer.",
                    expected_output="A polished version of the blog post in markdown format.",
                    agent=editor
                )

                enrich_links = Task(
                    description="Locate official bodies or industry-standard platforms and add markdown links without altering core text.",
                    expected_output="The complete markdown blog post containing navigation hyperlinks.",
                    agent=linker
                )

                # --- CREW ASSEMBLY WITH MAX REQUEST PACING ---
                crew = Crew(
                    agents=[planner, writer, editor, linker],
                    tasks=[plan, write, edit, enrich_links],
                    verbose=True,
                    max_rpm=1  # <-- Hard limit: Ensures actions stagger across a broader minute window
                )

                # Execute
                result = crew.kickoff(inputs={"topic": topic})
                final_text = result.raw if hasattr(result, 'raw') else str(result)
                
                # Success display
                status_box.empty()
                st.success("🎉 Article generated successfully without exceeding limits!")
                st.subheader("📝 Finished Blog Post")
                st.markdown(final_text)
                
                st.download_button(
                    label="📥 Download Article",
                    data=final_text,
                    file_name=f"{topic.lower().replace(' ', '_')}_article.md",
                    mime="text/markdown"
                )
                break # Exit the retry loop upon successful execution
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a Rate Limit / Token threshold error
                if "rate_limit_exceeded" in error_msg.lower() or "ratelimiterror" in error_msg.lower():
                    # Check if Groq told us exactly how long to wait via their API response
                    # If not specified, we exponentially back off (5s, 15s, etc.)
                    wait_time = retry_delay * (attempt + 1) * 2
                    if "try again in" in error_msg:
                        try:
                            # Try parsing the exact seconds out of Groq's error text (e.g., "try again in 3.27s")
                            parts = error_msg.split("try again in ")
                            wait_time = float(parts[1].split("s")[0]) + 1.5 # Add a tiny padding buffer
                        except:
                            pass
                            
                    status_box.warning(f"⏳ Groq rate limit threshold reached. Cooled down triggered: Waiting {wait_time:.2f} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    # If it's a different code error, stop execution immediately to debug
                    status_box.error(f"Execution Error: {e}")
                    break
        else:
            status_box.error("❌ High traffic limit: Failed to execute after maximum cooling attempts. Please wait 1 minute before requesting another topic.")