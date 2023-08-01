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
from langchain.prompts import PromptTemplate
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

# Move variables and functions that don't need to be in the main function outside
nltk.download("punkt", quiet=True)

from nltk import word_tokenize, sent_tokenize


openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise Exception("OPENAI_KEY not found in environment variables")

embeddings = OpenAIEmbeddings()




def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


def call_QA_to_json(
    prompt, year, month, day, saved_patent_names, count=8, logging=True, model_name="gpt-3.5-turbo"
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

    llm = ChatOpenAI(model_name=model_name)
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


    if logging:
        print("Generating embeddings and persisting...")

    vectordb = Chroma.from_documents(
        documents=documents, embedding=embeddings,
    )

    # vectordb.persist()
    PROMPT_FORMAT = """
    Task: Use the following pieces of context to answer the question at the end.

    {context}

    Question: {question}
    """

    PROMPT = PromptTemplate(
        template=PROMPT_FORMAT, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}



    retrieval_chain = RetrievalQA.from_chain_type(
        llm, chain_type="stuff", 
        retriever=vectordb.as_retriever(), 
        chain_type_kwargs=chain_type_kwargs, 
        # return_source_documents=True

    )

    if logging:
        print("Running retrieval chain...")

    with get_openai_callback() as cb:
        output = retrieval_chain.run(prompt)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Successful Requests: {cb.successful_requests}")
        print(f"Total Cost (USD): ${cb.total_cost}")       
    

    # Convert output to dictionary
    output_dict = json.loads(output)

    # result  = retrieval_chain({"query": prompt})
    # print(result["result"])
    # print(result["source_documents"])

    # Convert output to dictionary
    output_dict = json.loads(output)

    # Manually assign the Patent Identifier
    output_dict["Patent Identifier"] = saved_patent_names[count].split("-")[0]


    # Check if the directory 'output' exists, if not create it
    if not os.path.exists("output"):
        os.makedirs("output")

    if logging:
        print("Writing the output to a file...")

    # Write the output to a file in the 'output' directory
    with open(f"output/{saved_patent_names[count]}.json", "w") as json_file:
        json.dump(output_dict, json_file, indent=4)

    if logging:
        print("Call to 'call_QA_to_json' completed.")

    return documents_raw, output


def call_TA_to_json(
    prompt, year, month, day, saved_patent_names, count=8, logging=True
):
    """
    Retrieve text analytics (TA) data from a specified patent file and convert the output to JSON format.

    This function reads a text document from the patent file specified by the year, month, day, and file name parameters.
    It then applies a QA retrieval process to the document using the provided prompt.
    The result of the QA retrieval process is converted to a JSON object, which is then written to a file.
    Additionally, a patent identifier is manually assigned to the output JSON object.

    Parameters:
        prompt (str): The input prompt for the retrieval process.
        year (int): The year part of the data folder name.
        month (int): The month part of the data folder name.
        day (int): The day part of the data folder name.
        saved_patent_names (list): A list of strings containing the names of saved patent text files.
        count (int, optional): The index of the saved patent text file to process. Default is 8.
        logging (bool, optional): If True, print logs to the console. Default is True.

    Returns:
        tuple: A tuple containing two elements:
            - documents_raw (str): The raw document content loaded from the specified patent file.
            - output (str): A JSON string representing the output from the TA retrieval process.

    Note:
        The output is also written to a file in the 'output' directory with the same name as the input file and a '.json' extension.
    """
    file_path = os.path.join(
        os.getcwd(),
        "data",
        "ipa" + str(year)[2:] + f"{month:02d}" + f"{day:02d}",
        saved_patent_names[count],
    )

    if logging:
        print(f"Loading documents from: {file_path}")

    with open(file_path, 'r') as f:
        documents_raw = f.read()


    PROMPT_FORMAT = """
    Task: Use the following pieces of context to answer the question at the end.
    Question: 
    """

    prompt = PROMPT_FORMAT + prompt

    qa_chain = load_qa_chain(llm, chain_type="map_reduce")

    qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)


    if logging:
        print("Running Analyze Document chain...")

    output = qa_document_chain.run(input_document=documents_raw, question=prompt)

    print(output)


    # Convert output to dictionary
    output_dict = json.loads(output)

    # Manually assign the Patent Identifier
    output_dict["Patent Identifier"] = saved_patent_names[count].split("-")[0]


    # Check if the directory 'output' exists, if not create it
    if not os.path.exists("output"):
        os.makedirs("output")

    if logging:
        print("Writing the output to a file...")

    # Write the output to a file in the 'output' directory
    with open(f"output/{saved_patent_names[count]}.json", "w") as json_file:
        json.dump(output_dict, json_file, indent=4)

    if logging:
        print("Call to 'call_QA_to_json' completed.")

    return documents_raw, output


