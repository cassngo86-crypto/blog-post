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
        if not kwargs["extra_body"]:  # Clean up if empty
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

st.title("✍️ Multi-Agent Content Generation Crew")
st.write("Input a topic below to watch a crew of specialized AI agents plan, write, and edit a blog post for you.")

# 4. Handle API Key Securely
# Looks for environmental variables (for production) or sidebar input (for quick testing)
groq_api_key = os.environ.get("GROQ_API_KEY") or st.sidebar.text_input("Enter Groq API Key", type="password")

if not groq_api_key:
    st.info("🔑 Please enter your Groq API Key in the sidebar or set the environment variable to begin.")
    st.stop()

# Instantiate the LLM
groq_llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=groq_api_key,
    temperature=0.7
)

# 5. User Input
topic = st.text_input("What topic would you like the agents to write about?", placeholder="e.g., Artificial Intelligence, Quantum Computing")

# 6. Execution Trigger
if st.button("Launch Crew Execution", type="primary"):
    if not topic.strip():
        st.warning("Please provide a valid topic.")
    else:
        # Spinner visualizes execution status to user
        with st.spinner(f"🕵️‍♂️ Agents are working hard on planning, writing, and editing your post about '{topic}'... Please wait."):
            try:
                # Define Agents
                planner = Agent(
                    role="Content planner",
                    goal="Plan engaging and factually accurate content on {topic}",
                    backstory=(
                        "You're working on a blog article about the topic: {topic}. "
                        "You collect information that helps the audience learn something and make informed decisions."
                    ),
                    allow_delegation=False,
                    verbose=True,
                    llm=groq_llm
                )

                writer = Agent(
                    role="Content Writer",
                    goal="Write insightful and factually accurate opinion pieces about the topic: {topic}.",
                    backstory=(
                        "You're working on writing a new opinion piece about the topic: {topic}."
                    ),
                    allow_delegation=False,
                    verbose=True,
                    llm=groq_llm
                )

                editor = Agent(
                    role="Editor",
                    goal="Edit a given blog post to align with the writing style of the organization.",
                    backstory=(
                        "You are an editor who receives the blog post from the Content Writer."
                    ),
                    allow_delegation=False,
                    verbose=True,
                    llm=groq_llm
                )

                # Define Tasks
                plan = Task(
                    description=(
                        "1. Prioritize the latest trends, key players, and noteworthy news on {topic}.\n" 
                        "2. Identify target audience, considering their interests and pain points.\n"
                        "3. Develop a detailed content outline including an introduction, key points, and a call to action.\n" 
                        "4. Include SEO keywords and relevant data or sources."
                    ),
                    expected_output="A comprehensive content plan document with an outline, audience analysis, SEO keywords, and resources.",
                    agent=planner,
                )

                write = Task(
                    description=(
                        "1. Use the content plan to create a compelling blog post on {topic}.\n"
                        "2. Incorporate SEO keywords naturally.\n"
                        "3. Ensure sections and subtitles are named in an engaging manner.\n"
                        "4. Structure the post with an engaging introduction, insightful body, and summarizing conclusion.\n"
                        "5. Proofread for grammatical errors and alignment with the brand's voice."
                    ),
                    expected_output="A well-written blog post in markdown format, ready for publication, where each section has 2 or 3 paragraphs.",
                    agent=writer,
                )

                edit = Task(
                    description="Review and refine the blog post written by the writer. Ensure it meets journalistic standards and tone.",
                    expected_output="A polished, final version of the blog post in markdown format.",
                    agent=editor
                )

                # Assemble Crew
                crew = Crew(
                    agents=[planner, writer, editor],
                    tasks=[plan, write, edit],
                    verbose=True
                )

                # Kickoff execution
                result = crew.kickoff(inputs={"topic": topic})
                
                # Retrieve raw string output compatibility from CrewOutput object
                final_text = result.raw if hasattr(result, 'raw') else str(result)
                
                # Display Results
                st.success("🎉 Crew tasks completed successfully!")
                
                st.subheader("📝 Finished Blog Post")
                st.markdown(final_text)
                
                # Download Button for the file
                st.download_button(
                    label="📥 Download Article as Markdown",
                    data=final_text,
                    file_name=f"{topic.lower().replace(' ', '_')}_article.md",
                    mime="text/markdown"
                )

            except Exception as e:
                st.error(f"An error occurred during execution: {e}")
