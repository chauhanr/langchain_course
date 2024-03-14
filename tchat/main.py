from dotenv import load_dotenv
load_dotenv()
import os
from langchain.prompts import MessagesPlaceholder,HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.chains import LLMChain
#from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from langchain.memory import ConversationSummaryMemory

openai_api_version = "2024-02-15-preview"
deployment_name = os.getenv("DEPLOYMENT_NAME")

def chat(content: str, memory: ConversationSummaryMemory, chat: AzureChatOpenAI) -> str:
    prompt = ChatPromptTemplate(
        input_variables=["content", "messages"], 
        messages=[
            MessagesPlaceholder(variable_name="messages"),
            HumanMessagePromptTemplate.from_template("{content}")
        ]
    )

    chain = LLMChain(
        verbose=True,
        prompt=prompt,
        llm=chat, 
        memory=memory 
    )
    result = chain.invoke(input={"content": content})

    return result["text"]

if __name__ == "__main__":
    llm_chat = AzureChatOpenAI(
        openai_api_version=openai_api_version,
        deployment_name=deployment_name
    )    
    memory = ConversationSummaryMemory(
        #chat_memory=FileChatMessageHistory("chat_memory.json"),
        memory_key="messages",
        return_messages=True, 
        llm = llm_chat)

    while True: 
        content = input("> ") # this function will wait for the user to put in details and press enter. 
        replay = chat(content, memory, llm_chat)
        print(f"Bot: {replay}")
