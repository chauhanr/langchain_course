from dotenv import load_dotenv
load_dotenv()
import os
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

import argparse


openai_version = "2024-02-15-preview"
deployment_name = os.environ.get('DEPLOYMENT_NAME')
# the azure openai key needs to be set as env variable AZURE_OPENAI_API_KEY
# set the AZURE_OPENAI_ENDPOINT as well for the AzureOpenChatAPI to work

def run_llm_task(language: str, task: str):
   llm = AzureChatOpenAI(
      deployment_name=deployment_name,
      openai_api_version=openai_version
   )   
   # create a prompt template 
   code_prompt = PromptTemplate(
      template="Write a very short {language} function that will {task}",
      input_variables=["language","task"]
   )
   test_prompt = PromptTemplate(
      template="Write a test case for the {language} and code:\n {code}",
      input_variables=["language","code"]
   )

   code_chain = LLMChain(
        prompt=code_prompt,
        llm=llm, 
        output_key="code"
   )
   test_chain = LLMChain(
        prompt=test_prompt,
        llm=llm, 
        output_key="test"
   )

   chain =  SequentialChain(
        chains=[code_chain, test_chain], 
        input_variables=["language","task"],
        output_variables=["code","test"]
   )
   
   result = chain.invoke(input={"language":language, "task":task})
   print(f"code: \n {result['code']}\n tests: \n {result['test']}")


if __name__ == '__main__':
    # capture parser. 
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", help="language to use", default="python")
    parser.add_argument("--task", help="task to perform", default="write a function to get factorial of a number")
    args = parser.parse_args()
    
    run_llm_task(args.language, args.task)