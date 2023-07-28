import os
import json
import nltk
import openai
from langchain.document_loaders import UnstructuredXMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Move variables and functions that don't need to be in the main function outside
nltk.download("punkt")

from nltk import word_tokenize, sent_tokenize


openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise Exception("OPENAI_KEY not found in environment variables")

embeddings = OpenAIEmbeddings()

model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name)


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


def call_QA_to_json(prompt, year, month, day, count=8):
    """
    This function generates embeddings from XML documents, retrieves data based on the provided prompt, and returns the result as a JSON object.

    Parameters:
    prompt (str): The input prompt for the retrieval process.
    year (int): The year part of the data folder name.
    month (int): The month part of the data folder name.
    day (int): The day part of the data folder name.
    count (int): The file count. Default is 8.

    Returns:
    A JSON string representing the output from the retrieval chain.
    """
    file_path = os.path.join(
        os.getcwd(),
        "data",
        "ipa" + str(year)[2:] + f"{month:02d}" + f"{day:02d}",
        str(count) + ".xml",
    )

    loader = UnstructuredXMLLoader(file_path)
    documents_raw = loader.load()

    documents = split_docs(documents_raw)

    persist_directory = "chroma_db"

    vectordb = Chroma.from_documents(
        documents=documents, embedding=embeddings, persist_directory=persist_directory
    )

    vectordb.persist()

    retrieval_chain = RetrievalQA.from_chain_type(
        llm, chain_type="stuff", retriever=vectordb.as_retriever()
    )

    output = retrieval_chain.run(prompt)

    # Check if the directory 'output' exists, if not create it
    if not os.path.exists("output"):
        os.makedirs("output")

    print(output)
    # Write the output to a file in the 'output' directory
    with open(f"output/{count}.json", "w") as json_file:
        # We need to convert the string to a Python dictionary using json.loads
        json.dump(json.loads(output), json_file, indent=4)

    return documents_raw, output
