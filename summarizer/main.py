import os
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
import textwrap



if __name__ == '__main__':
    llm = OpenAI(temperature=0)
    os.environ["OPENAI_API_KEY"] = ""

    # load the doc
    with open('/content/how_to_win_friends.txt') as f:
        how_to_win_friends = f.read()
    texts = text_splitter.split_text(how_to_win_friends)

    docs = [Document(page_content=t) for t in texts[:4]]

    chain = load_summarize_chain(llm, chain_type="map_reduce")

    output_summary = chain.run(docs)
    wrapped_text = textwrap.fill(output_summary, width=100)
    print(wrapped_text)
    # for summarizing each part
    chain.llm_chain.prompt.template
    # for combining the parts
    chain.combine_document_chain.llm_chain.prompt.template

    chain = load_summarize_chain(llm,
                                 chain_type="map_reduce",
                                 verbose=True
                                 )

    output_summary = chain.run(docs)
    wrapped_text = textwrap.fill(output_summary,
                                 width=100,
                                 break_long_words=False,
                                 replace_whitespace=False)
    print(wrapped_text)

    chain = load_summarize_chain(llm, chain_type="stuff")

    prompt_template = """Write a concise bullet point summary of the following:


    {text}


    CONSCISE SUMMARY IN BULLET POINTS:"""

    BULLET_POINT_PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm,
                                 chain_type="stuff",
                                 prompt=BULLET_POINT_PROMPT)

    output_summary = chain.run(docs)

    wrapped_text = textwrap.fill(output_summary,
                                 width=100,
                                 break_long_words=False,
                                 replace_whitespace=False)
    print(wrapped_text)

    chain = load_summarize_chain(llm,
                                 chain_type="map_reduce",
                                 map_prompt=BULLET_POINT_PROMPT,
                                 combine_prompt=BULLET_POINT_PROMPT)

    # chain.llm_chain.prompt= BULLET_POINT_PROMPT
    # chain.combine_document_chain.llm_chain.prompt= BULLET_POINT_PROMPT

    output_summary = chain.run(docs)
    wrapped_text = textwrap.fill(output_summary,
                                 width=100,
                                 break_long_words=False,
                                 replace_whitespace=False)
    print(wrapped_text)

    # with a custom prompt
    prompt_template = """Write a concise summary of the following:


    {text}


    CONSCISE SUMMARY IN BULLET POINTS:"""

    PROMPT = PromptTemplate(template=prompt_template,
                            input_variables=["text"])

    # with intermediate steps
    chain = load_summarize_chain(OpenAI(temperature=0),
                                 chain_type="map_reduce",
                                 return_intermediate_steps=True,
                                 map_prompt=PROMPT,
                                 combine_prompt=PROMPT)

    output_summary = chain({"input_documents": docs}, return_only_outputs=True)
    wrapped_text = textwrap.fill(output_summary['output_text'],
                                 width=100,
                                 break_long_words=False,
                                 replace_whitespace=False)
    print(wrapped_text)

    wrapped_text = textwrap.fill(output_summary['intermediate_steps'][2],
                                 width=100,
                                 break_long_words=False,
                                 replace_whitespace=False)
    print(wrapped_text)

    """This method involves an initial prompt on the first chunk of data, generating some output. For the 
    remaining documents,that output is passed in, along with the next document, asking the LLM 
    to refine the output based on the new document."""

    chain = load_summarize_chain(llm, chain_type="refine")

    output_summary = chain.run(docs)
    wrapped_text = textwrap.fill(output_summary, width=100)
    print(wrapped_text)

    prompt_template = """Write a concise summary of the following extracting the key information:


    {text}


    CONCISE SUMMARY:"""
    PROMPT = PromptTemplate(template=prompt_template,
                            input_variables=["text"])

    refine_template = (
        "Your job is to produce a final summary\n"
        "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original summary"
        "If the context isn't useful, return the original summary."
    )
    refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )
    chain = load_summarize_chain(OpenAI(temperature=0),
                                 chain_type="refine",
                                 return_intermediate_steps=True,
                                 question_prompt=PROMPT,
                                 refine_prompt=refine_prompt)

    output_summary = chain({"input_documents": docs}, return_only_outputs=True)
    wrapped_text = textwrap.fill(output_summary['output_text'],
                                 width=100,
                                 break_long_words=False,
                                 replace_whitespace=False)
    print(wrapped_text)

    wrapped_text = textwrap.fill(output_summary['intermediate_steps'][0],
                                 width=100,
                                 break_long_words=False,
                                 replace_whitespace=False)
    print(wrapped_text)
