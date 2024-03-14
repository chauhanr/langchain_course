from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv
load_dotenv()
from tools.sql import run_query_tool
import os

openai_api_version = "2024-02-15-preview"
deployment_name = os.getenv("DEPLOYMENT_NAME")

def run_agents(chat: AzureChatOpenAI, query: str):
    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template("{input}"), 
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
    ) 

    tools = [run_query_tool]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(
        llm=chat, 
        prompt=react_prompt, 
        tools=tools
    )
    
    agent_executor=AgentExecutor(
        agent=agent, 
        verbose=True, 
        tools=tools
    )
    result = agent_executor.invoke(input={"input": prompt.format(input=query)})
    return result["output"]


if __name__ == "__main__":
    chat = AzureChatOpenAI(
        openai_api_version= openai_api_version, 
        deployment_name=deployment_name
    )

    res = run_agents(chat, "How many pending orders are present in my database.")
    print(res)
    