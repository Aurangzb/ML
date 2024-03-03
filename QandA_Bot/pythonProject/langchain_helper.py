from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.chains import RetrievalQA

env_var = load_dotenv()
api_key = os.environ["GOOGLE_API_KEY"]

llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=api_key, temperature=0.1)

instructor_embeddings = HuggingFaceInstructEmbeddings()
vectordb_file_path = "faiss_index"

def create_vector_db():
    loader = CSVLoader("D:\Projects\Machine Learning\codebasics_faqs.csv", source_column='prompt')
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    # Load the vector database from local folder
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)
    retriever = vectordb.as_retriever()
    prompt_tempate = """Given the following context and the question, generate an answer based on this context.
                     In the answerw try to provide as much text as possible from responce section in the source document
                     if the answer is not found in the context, kindly state "I don't know." Dont try to make up an answer.

                     CONTEXT: {context}

                     QUESTION: {question}"""

    PROMPT = PromptTemplate(template=prompt_tempate, input_variables=["context", "question"])

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT}
                                        )
    return chain

if __name__ == "__main__":
    chain = get_qa_chain()
    print(chain("do you provide internship? Do you have EMI option?"))