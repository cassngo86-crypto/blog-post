import streamlit as st
import os
import warnings
import litellm
from crewai import Agent, Task, Crew, LLM

# Disable warnings
warnings.filterwarnings("ignore")

# 1. Store the original LiteLLM completion function
original_completion = litellm.completion

# 2. Define a clean wrapper to intercept and strip out 'cache_breakpoint'
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

# 3. Apply the global monkey patch
litellm.completion = safe_completion

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="AI Agent Content Creator", page_icon="✍️", layout="centered")

st.title("✍️ Multi-Agent Dynamic Content Crew")
st.write("Input any topic. Specialized agents will plan, write, edit, and dynamically add contextual resource links.")

# 4. Handle API Key Securely
groq_api_key = os.environ.get("GROQ_API_KEY") or st.sidebar.text_input("Enter Groq API Key", type="password")

if not groq_api_key:
    st.info("🔑 Please enter your Groq API Key in the sidebar or set the environment variable to begin.")
    st.stop()

# --- TARGETED DUAL MODEL INSTANTIATION ---
# Lightweight, high-token-allowance model for the upfront tracking & structural link formatting tasks
fast_llm = LLM(
    model="groq/llama-3.1-8b-instant",
    api_key=groq_api_key,
    temperature=0.2
)

# High-intelligence model reserved strictly for pure writing and editing tasks
smart_llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=groq_api_key,
    temperature=0.7
)

# 5. User Input
topic = st.text_input("What topic would you like the agents to write about?", placeholder="e.g., Data Architecture Strategies, Travelling to Tokyo")

# 6. Execution Trigger
if st.button("Launch Crew Execution", type="primary"):
    if not topic.strip():
        st.warning("Please provide a valid topic.")
    else:
        with st.spinner(f"🕵️‍♂️ Agents are managing tokens and designing your linked article for '{topic}'..."):
            try:
                # --- AGENTS ---
                planner = Agent(
                    role="Content planner",
                    goal="Plan engaging and factually accurate content architecture on {topic}",
                    backstory=(
                        "You're organizing an article on {topic}. "
                        "You extract core structural points and note high-profile organizations or authoritative agencies related to the field."
                    ),
                    allow_delegation=False,
                    verbose=True,
                    llm=fast_llm # Uses faster model to preserve token capacity
                )

                writer = Agent(
                    role="Content Writer",
                    goal="Write insightful and factually accurate opinion pieces about the topic: {topic}.",
                    backstory="You're writing a premium, multi-paragraph opinion or guide piece about {topic}.",
                    allow_delegation=False,
                    verbose=True,
                    llm=smart_llm # Reserved for the heavy writing assignment
                )

                editor = Agent(
                    role="Editor",
                    goal="Edit a given blog post to align with professional formatting guidelines.",
                    backstory="You adjust syntax, tone, and correct mechanical bugs from drafts.",
                    allow_delegation=False,
                    verbose=True,
                    llm=smart_llm # Reserved for styling polish
                )

                linker = Agent(
                    role="Digital Link Optimizer",
                    goal="Read a completed text draft and format references to primary brands, official agencies, and key platforms into standard Markdown links dynamically.",
                    backstory=(
                        "You possess deep domain awareness. You map well-known tools, official documentation bodies, "
                        "or primary corporations mentioned in text directly into valid Markdown hyperlinks without changing text structure."
                    ),
                    allow_delegation=False,
                    verbose=True,
                    llm=fast_llm # Uses the fast model to avoid rate limiting on formatting loops
                )

                # --- TASKS ---
                plan = Task(
                    description=(
                        "1. Identify trends and key players on {topic}.\n"
                        "2. Create a clean article outline with clear structural requirements."
                    ),
                    expected_output="A clean content plan document with an outline and key domain terms.",
                    agent=planner,
                )

                write = Task(
                    description=(
                        "1. Convert the content plan into an engaging article on {topic}.\n"
                        "2. Structure it cleanly using markdown headers."
                    ),
                    expected_output="A structured blog post in markdown format where each section has 2 or 3 paragraphs.",
                    agent=writer,
                )

                edit = Task(
                    description="Review and refine the blog post. Ensure it meets journalistic standards and tone.",
                    expected_output="A polished draft of the blog post in markdown format.",
                    agent=editor
                )

                enrich_links = Task(
                    description=(
                        "Carefully read the edited text. Locate any major official platforms, governing departments, dominant corporate entities, "
                        "or industry-standard applications referenced. Seamlessly convert those exact references into clickable Markdown links "
                        "(e.g., if it mentions Python documentation, convert it to [Python Documentation](https://www.python.org)). "
                        "Return the complete, final article with these links embedded."
                    ),
                    expected_output="The complete, finalized markdown blog post containing automated contextual navigation hyperlinks.",
                    agent=linker
                )

                # --- ASSEMBLE CREW WITH RATE RECOVERY ---
                crew = Crew(
                    agents=[planner, writer, editor, linker],
                    tasks=[plan, write, edit, enrich_links],
                    verbose=True,
                    max_rpm=2 # Enforces a deliberate multi-second pause to let the Groq token bucket replenish
                )

                # Run Crew
                result = crew.kickoff(inputs={"topic": topic})
                final_text = result.raw if hasattr(result, 'raw') else str(result)
                
                # --- SAFETY FALLBACK DISPLAY ---
                st.success("🎉 Process complete!")
                st.subheader("📝 Finished Blog Post with Dynamic Links")
                st.markdown(final_text)
                
                st.download_button(
                    label="📥 Download Enriched Article",
                    data=final_text,
                    file_name=f"{topic.lower().replace(' ', '_')}_linked_article.md",
                    mime="text/markdown"
                )

            except Exception as e:
                st.error(f"An error occurred during execution: {e}")