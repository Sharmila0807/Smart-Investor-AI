from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Provide the latest news", "Always include sources"],
    show_tools_calls=True,
    markdown=True,
)
finance_agent = Agent(
    name="Financial AI Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        stock_fundamentals=True,
        company_news=True,
        technical_indicators=True,
        historical_prices=True,
    )],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    team=[web_search_agent, finance_agent],
    instructions=["Always include sources", "Use tables to display the data"],
    markdown=True,
    debug_mode=True,
)

multi_ai_agent.print_response("Summarize analyst recommendations and share the latest news for Cognizant Technology Solution", stream=True)