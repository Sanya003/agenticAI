# Import necessary libraries and modules
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os
from groq import Client

# Load environment variables from .env
load_dotenv()

# Initialize the Groq client
client = Client(api_key=os.getenv("GROQ_API_KEY"))

# Create a web search agent with DuckDuckGo integration
web_search_agent=Agent(
    name="Web Search Agent", 
    role="Search the web for information.",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True
)

# Create a financial agent with Yahoo Finance tools
finance_agent=Agent(
    name="Finance AI Agent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[
        YFinanceTools(
            stock_price=True, 
            analyst_recommendations=True, 
            stock_fundamentals=True, 
            company_news=True
        )
    ],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True
)

# Combine the web search agent and the financial agent into a multi-agent system
multi_ai_agent=Agent(
    team=[web_search_agent, finance_agent],
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True
)

# Use the combined multi-agent system to generate a response
multi_ai_agent.print_response(
    "Summarize analyst recommendation and share the latest news for NVDA",
    stream=True
)
