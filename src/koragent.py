import os
import json
import nltk
import openai
from typing import List, Optional
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number
import pandas as pd
from pydantic import BaseModel, Field, validator
from kor import extract_from_documents, from_pydantic, create_extraction_chain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.callbacks import get_openai_callback

def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

async def call_extraction_to_json(schema, year, month, day, saved_patent_names, index=8, logging=True, model_name = 'gpt-3.5-turbo'):
    """
    Load a specified patent file, perform a document extraction based on the provided schema, and save the results in a JSON format.

    This function uses the provided schema to create an extraction chain which is then applied to a document loaded from a 
    specified patent file (determined by year, month, day, and file name). The result of the extraction is converted to a JSON
    object, which is then written to a file. A patent identifier is manually assigned to the output JSON object.

    Parameters:
        schema (dict): The schema to use for creating the extraction chain.
        year (int): The year part of the data folder name.
        month (int): The month part of the data folder name.
        day (int): The day part of the data folder name.
        saved_patent_names (list): A list of strings containing the names of saved patent text files.
        index (int, optional): The index of the saved patent text file to process. Default is 8.
        logging (bool, optional): If True, print logs to the console. Default is True.

    Returns:
        tuple: A tuple containing two elements:
            - documents_raw (str): The raw document content loaded from the specified patent file.
            - output (str): A JSON string representing the output from the document extraction process.

    Note:
        The output is also written to a file in the 'output' directory with the same name as the input file and a '.json' extension.
    """

    llm = ChatOpenAI(model_name=model_name)

    if logging:
        print("Starting the extraction process...")

    chain = create_extraction_chain(llm, schema)

    file_path = os.path.join(
        os.getcwd(),
        "data",
        "ipa" + str(year)[2:] + f"{month:02d}" + f"{day:02d}",
        saved_patent_names[index],
    )

    if logging:
        print(f"Loading documents from: {file_path}")

    loader = TextLoader(file_path)
    documents_raw = loader.load()
    documents = split_docs(documents_raw)

    if logging:
        print("Running extraction chain...")

    with get_openai_callback() as cb:
        output = await extract_from_documents(chain, documents, max_concurrency=5, use_uid=False, return_exceptions=True)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Successful Requests: {cb.successful_requests}")
        print(f"Total Cost (USD): ${cb.total_cost}")

    if logging:
        print(output[0])

    # Convert output to dictionary
    output_dict = output[0]

    # Manually assign the Patent Identifier
    output_dict["Patent Identifier"] = saved_patent_names[index].split("-")[0]

    # Check if the directory 'output' exists, if not create it
    if not os.path.exists("output"):
        os.makedirs("output")

    if logging:
        print("Writing the output to a file...")

    # Write the output to a file in the 'output' directory
    with open(f"output/{saved_patent_names[index]}_{model_name}.json", "w", encoding="utf-8") as json_file:
        json.dump(output_dict, json_file, indent=4, ensure_ascii=False)

    if logging:
        print("Call to 'call_extraction_to_json' completed.")

    return documents_raw, output