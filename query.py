from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.documents import Document
from typing import List, Optional
import os

class Search(BaseModel):
    """Search over a database of pdfs about drug medication."""
    query: str = Field(..., description="Similarity search query applied to pdf.")
    publish_year: Optional[int] = Field(None, description="Year of the pdf publication")
    population: Optional[str] = Field(None, description="Type of population studied in the pdf")
    drug_phase: Optional[str] = Field(None, description="Phase of drug development")
    treatment_method: Optional[str] = Field(None, description="Method of treatment")
    epidemiology: Optional[str] = Field(None, description="Epidemiological data")

# Load embeddings globally
emb_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=emb_model,
    cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME', './cache')
)


def load_and_vectorize_pdfs(directory: str, store_path: str) -> Chroma:
    # Check if the vector store already exists and can be loaded
    """
    if os.path.exists(store_path):
        print("Loading existing vector store from:", store_path)
        return Chroma(persist_directory=store_path, embedding_function=embeddings)
    """
    # Process PDFs if the vector store does not exist
    filenames = os.listdir(directory)
    all_documents = []
    for filename in filenames:
        file_path = os.path.join(directory, filename)
        if not file_path.lower().endswith('.pdf'):
            print(f"Skipped non-PDF file: {file_path}")
            continue
        
        try:
            loader = PyPDFLoader(file_path)
            print(f"Loading PDF: {file_path}")
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
            chunked_docs = text_splitter.split_documents(documents)
            all_documents.extend(chunked_docs)
        except Exception as e:
            print(f"Failed to load PDF: {e}")
            continue

    # Create a new Chroma vector store and save it
    print("Creating new vector store at:", store_path)
    vectorstore = Chroma.from_documents(all_documents, embeddings, persist_directory=store_path)
    vectorstore.save()  # Explicitly save the store
    return vectorstore

# Load and vectorize PDFs
store_directory = "vectorstore"
vectorstore = load_and_vectorize_pdfs("filestest", store_directory)


def retrieval(search: Search) -> List[Document]:
    _filter = {}
    if search.drug_phase:
        _filter["drug_phase"] = {"$eq": search.drug_phase}

    results = vectorstore.similarity_search(search.query, filter=_filter if _filter else None)
    return sorted(results, key=lambda x: x.score, reverse=True)[:3]

# System prompt setup for advanced query handling
print("Setting up system prompt for advanced query handling...")
system = system = """
You are an expert at converting user questions into precise database queries. 
You have access to a database of PDF documents about drug medications, including their clinical strategies, applicable diseases, and operational treatment processes.
Given a user's question, your task is to return a list of database queries optimized to retrieve the most relevant results.
Please maintain the original terminology, including any acronyms or specific medical terms, without attempting to rephrase them.
"""
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm = llm.with_structured_output(Search)
query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm

def execute_query(query: str) -> List[Document]:
    retrieval_chain = query_analyzer | retrieval
    return retrieval_chain.invoke(query)

# Define your specific context as a search query
search_query = """
    This query seeks documents detailing drug development in phase I-II specifically targeting pediatric patients with neurologic diseases. 
    Key focuses are on treatments for epileptic seizures and loss of functional skills. Clinical trials should emphasize changes in seizure frequency and standard safety monitoring.
"""
# Perform the search
print("Executing search query:")



#results = execute_query(search_criteria)
result = query_analyzer.invoke(search_query)
print(result)
retrieval_chain = query_analyzer | retrieval
results = retrieval_chain.invoke("I am looking for documents about drug development for epilepsy in children")
print(results)
# Print the results

print("Number of matching documents:", len(results))
for doc in results:
    print(f"Document Title: {doc.title}")  # Ensure document objects have a
