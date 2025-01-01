# Import necessary libraries and modules
import openai
from phi.agent import Agent
import phi.api
from phi.model.openai import OpenAIChat
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os
import phi
from phi.playground import Playground, serve_playground_app

# Load environment variables from a .env file
load_dotenv()

# Set the PHI API key from environment variables
phi.api=os.getenv("PHI_API_KEY")

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

# Create a PHI Playground app with the defined agents
app=Playground(agents=[finance_agent, web_search_agent]).get_app()

# Run the Playground app
if __name__=="__main__":
    serve_playground_app("playground:app", reload=True)