import streamlit as st
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
import os

# Load environment variables (make sure .env contains GROQ_API_KEY)
load_dotenv()

# Setup Agents
web_search_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    name="Web Search Agent",
    description="Search the web for latest financial and company news.",
    instructions=[
        "Always include sources in your response.",
        "Provide only the most relevant and recent information.",
    ],
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    name="Financial Analyst Agent",
    description="Provide detailed stock information using tables.",
    instructions=[
        "Use tables to summarize analyst recommendations, stock fundamentals, and technical indicators.",
        "Use bullet points when summarizing company news.",
    ],
    tools=[YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        stock_fundamentals=True,
        company_news=True,
        technical_indicators=True,
        historical_prices=True,
    )],
    show_tool_calls=True,
    markdown=True,
)

smart_investor_ai = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    name="Smart Investor AI",
    description="Aggregates real-time current stock insights and financial news through collaboration.",
    instructions=[
        "Delegate financial queries to Financial Analyst Agent.",
        "Use Web Search Agent when web-based info is needed.",
        "Always include sources and tables where necessary.",
    ],
    team=[finance_agent, web_search_agent],
    markdown=True,
    debug_mode=False,  # Turn off debug in production UI
)

# Streamlit UI
st.set_page_config(page_title="Smart Investor AI", layout="centered")
st.title("📊 Smart Investor AI")
st.markdown(
    "Ask for stock analysis, financial news, or analyst recommendations.<br>"
    "(e.g., _'Summarize analyst recommendations and share the latest news of Apple'_)", 
    unsafe_allow_html=True
)
# Input form
user_input = st.text_input("🔍 Enter your financial query:", "")

# Run button
if st.button("Run Analysis") and user_input.strip():
    with st.spinner("Analyzing your query..."):
        try:
            response = smart_investor_ai.run(user_input)
            result = response.content

            # Display the output
            st.markdown(result)

            # Download button
            
            st.download_button(
                label="📥 Download Report",
                data=result,
                file_name="smart_investor_report.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"❌ Error: {e}")
