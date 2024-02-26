# This is a sample Python script.
from typing import Tuple

from langchain.chains import LLMChain #Sequences of calls
from langchain_openai import ChatOpenAI #interface to interact with LLMs
from langchain_core.prompts import PromptTemplate # wrapper class for our inputs to the LLM
import openai
from third_party.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from output_parsers import (
    person_intel_parser,
    PersonIntel
)
from langchain.output_parsers import PydanticOutputParser
import os

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
OPENAI_API_KEY = "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"
openai.api_key = "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"
os.environ["SERPAPI_API_KEY"] = "22957c946deeee4f2061460d1832bc082ed1168ae6d665b8fbc632681eb1f21d"
os.environ["OPENAI_API_KEY"] = "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"


def ice_break(name:str) -> Tuple[PersonIntel,str]:
    information = """Elon Reeve Musk (/ˈiːlɒn/; EE-lon; born June 28, 1971) is a businessman and investor. 
        He is the founder, chairman, CEO, and CTO of SpaceX; angel investor, CEO, product architect, and former chairman of Tesla, Inc.; owner, 
        chairman, and CTO of X Corp.; founder of the Boring Company and xAI; co-founder of Neuralink and OpenAI; and president of the Musk Foundation. 
        He is the second wealthiest person in the world, with an estimated net worth of US$232 billion as of December 2023, 
        according to the Bloomberg Billionaires Index, and $182.6  billion according to Forbes, primarily from his ownership stakes in Tesla and SpaceX."""

    summary_template = """
        given the information {information} about a person I want you to create
        1. a short summary
        2. two interesting facts about them
        \n{format_instructions}
        """

    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template,
                                             partial_variables={
                                                 "format_instructions": person_intel_parser.get_format_instructions()})

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    result = chain.run(information=information)

    print(result)

    # repeat this with our linkedIn data instead

    linkedin_data = scrape_linkedin_profile(linkedin_profile_url="https://www.linkedin.com/in/sritad/")

    linkedin_profile_url = linkedin_lookup_agent(name=name) #lookup function that uses the agent + tool
    # try again but clear the data
    linkedin_data2 = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)

    print(chain.run(information=linkedin_data))
    result2 = chain.run(information=linkedin_data2)
    print(result2)
    return person_intel_parser.parse(result2),  linkedin_data2.get("profile_pic_url")
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print("Hello LangChain!")
    final_result = ice_break(name = "Srikanth Tadisetty")

#create a prompt



    """So far we just used a simple web call to get information, but let's use agents part of langchain now"""


