import streamlit as st
from agent import create_content
from agent import GeminiLLM  # for lightweight additional agent
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="AI Content Generator Suite",
    layout="wide",
)

# ---------------------------------------------
# HEADER
# ---------------------------------------------
st.title("✨ AI Content Generator Suite")
st.markdown("""
Welcome to your **Multi-Agent AI Content Studio** powered by:
- **CrewAI** (Research → Writing → Editing → SEO)
- **Google Gemini API**
- **Tavily Web Search**
""")

tabs = st.tabs(["🧠 CrewAI Multi-Agent Generator", "⚡ Additional Content Generator Agent"])


# ================================================================
# 🔥 TAB 1 — FULL CREWAI MULTI-AGENT CONTENT GENERATOR
# ================================================================
with tabs[0]:

    st.header("🧠 Multi-Agent Content Generator (CrewAI)")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Enter Topic")
        topic = st.text_area("Topic", placeholder="e.g. 'AI Tools for Students in 2025'")

    with col2:
        st.subheader("Settings")
        content_type = st.selectbox("Content Type", ["Blog", "YouTube Script", "Newsletter", "Product Description"])
        length = st.selectbox("Length", ["Short", "Medium", "Long"])
        tone = st.selectbox("Tone", ["Informal", "Professional", "Conversational", "Friendly"])
        use_research = st.checkbox("Use Tavily Research", value=True)

        generate_btn = st.button("🚀 Generate Content")

    if generate_btn:
        if not topic.strip():
            st.error("❌ Please enter a valid topic.")
        else:
            with st.spinner("Running CrewAI Agents..."):
                result = create_content(topic, content_type, length, tone, use_research)

            st.success("✅ CrewAI Pipeline Complete!")

            # Final Output — render as normal (markdown/plain text)
            st.markdown("## 📄 Final Output")
            final_out = result.get("final_output", "") or "(no final output)"
            # Use markdown to render headings/links; will appear as normal text if plain
            try:
                st.markdown(final_out)
            except Exception:
                st.text(final_out)

            # Pipeline Agent Outputs
            st.markdown("---")
            st.markdown("## 🧩 Agent Outputs")
            pipeline = result.get("pipeline", {})
            if pipeline:
                for task_name, output in pipeline.items():
                    st.markdown(f"### 🔸 **{task_name}**")
                    out_text = output or ""
                    try:
                        st.markdown(out_text)
                    except Exception:
                        st.text(out_text)
            else:
                st.info("No pipeline details available.")



# ================================================================
# ⚡ TAB 2 — ADDITIONAL FAST CONTENT GENERATOR AGENT
# ================================================================
with tabs[1]:

    st.header("⚡ Additional Quick Content Generator Agent")

    st.markdown("""
This tool uses the **Gemini LLM only** (not CrewAI).  
It is optimized for fast generation of:
- Instagram captions  
- YouTube titles  
- Tweet/X posts  
- Short hooks  
- LinkedIn micro-posts  
- Summaries  
""")

    colA, colB = st.columns([1.3, 1])

    with colA:
        quick_topic = st.text_area("Enter Topic / Keyword", placeholder="e.g. 'Future of AI'", key="quick_topic_box")

    with colB:
        quick_type = st.selectbox(
            "Select Content Type",
            [
                "Instagram Caption",
                "YouTube Title Ideas",
                "Short Tweet Post",
                "LinkedIn Mini Post",
                "Viral Hooks",
                "1-Paragraph Summary"
            ]
        )

        quick_btn = st.button("⚡ Generate Quick Content")

    if quick_btn:
        if not quick_topic.strip():
            st.error("Please enter text for generation.")
        else:
            st.info("Generating...")

            llm = GeminiLLM()

            # Build prompt for fast agent
            quick_prompt = f"Generate {quick_type} for the topic: {quick_topic}. Provide 5 options."

            output = llm.run(quick_prompt)

            st.success("Generated Successfully!")
            st.markdown("### 📤 Output")
            # Render as markdown/plain text in normal font
            try:
                st.markdown(output)
            except Exception:
                st.text(output)
