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
from langchain.document_loaders import TextLoader

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


def call_QA_to_json(
    prompt, year, month, day, saved_patent_names, count=8, logging=True
):
    """
    Generate embeddings from txt documents, retrieve data based on the provided prompt, and return the result as a JSON object.

    Parameters:
        prompt (str): The input prompt for the retrieval process.
        year (int): The year part of the data folder name.
        month (int): The month part of the data folder name.
        day (int): The day part of the data folder name.
        saved_patent_names (list): A list of strings containing the names of saved patent text files.
        count (int): The index of the saved patent text file to process. Default is 8.
        logging (bool): The boolean to print logs

    Returns:
        tuple: A tuple containing two elements:
            - A list of strings representing the raw documents loaded from the specified XML file.
            - A JSON string representing the output from the retrieval chain.

    This function loads the specified txt file, generates embeddings from its content,
    and uses a retrieval chain to retrieve data based on the provided prompt.
    The retrieved data is returned as a JSON object, and the raw documents are returned as a list of strings.
    The output is also written to a file in the 'output' directory with the name '{count}.json'.
    """
    file_path = os.path.join(
        os.getcwd(),
        "data",
        "ipa" + str(year)[2:] + f"{month:02d}" + f"{day:02d}",
        saved_patent_names[count],
    )

    if logging:
        print(f"Loading documents from: {file_path}")
    loader = TextLoader(file_path)
    documents_raw = loader.load()

    documents = split_docs(documents_raw)

    persist_directory = "chroma_db"

    if logging:
        print("Generating embeddings and persisting...")

    vectordb = Chroma.from_documents(
        documents=documents, embedding=embeddings, persist_directory=persist_directory
    )

    # vectordb.persist()

    retrieval_chain = RetrievalQA.from_chain_type(
        llm, chain_type="stuff", retriever=vectordb.as_retriever()
    )

    if logging:
        print("Running retrieval chain...")
    output = retrieval_chain.run(prompt)

    # Check if the directory 'output' exists, if not create it
    if not os.path.exists("output"):
        os.makedirs("output")

    if logging:
        print("Writing the output to a file...")
    # Write the output to a file in the 'output' directory
    with open(f"output/{saved_patent_names[count]}.json", "w") as json_file:
        # We need to convert the string to a Python dictionary using json.loads
        json.dump(json.loads(output), json_file, indent=4)

    if logging:
        print("Call to 'call_QA_to_json' completed.")

    return documents_raw, output
