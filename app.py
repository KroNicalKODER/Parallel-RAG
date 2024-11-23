import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import re
import sys
import os
import uuid

pdf_path = './pdfs/'
embeddings_directory = './embeddings/'

## CREATING HUGGINGFACE EMBEDDING INSTANCE
embedding_instance = ( HuggingFaceEmbeddings(model_name="hkunlp/instructor-large") )

## MAKING LLM INSTANCE THROUGH OLLAMA
llama = Ollama(model="llama3.2")                    ## Download this model else it will be automatically downloaded

## STARTING OUR STREAMLIT APP
st.title("PDF-RAG Machine")

## Upload a pdf and save its embeddings
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    st.write("File uploaded successfully!", uploaded_file.name)
    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # GENERATING THE EMBEDDINGS
    loader = PyPDFLoader("uploaded_file.pdf")
    docs = loader.load()
    
    # Splitting the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=300)
    final_docs = text_splitter.split_documents(docs)

    # Creating the embeddings
    embedding_path_name = embeddings_directory + uploaded_file.name
    db = Chroma.from_documents(final_docs, embedding_instance, persist_directory=embedding_path_name)
    db.persist()

    st.write("Embeddings are created and saved successfully!")


if "messages" not in st.session_state:
    st.session_state.messages = {"default": []}
# st.session_state.messages = {"default": []}

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = "default"


st.sidebar.title("Chats")

def add_new_session(session_id):
    if session_id and session_id not in st.session_state.messages:
        st.session_state.messages[session_id] = []  # Initialize messages for the new session
        st.session_state.active_session_id = session_id  # Set the new session as active
        

# Create a text box for entering or switching session ID
new_session_id = st.sidebar.text_input("Create a new session", value="")
if st.sidebar.button("Create New Session"):
    add_new_session(new_session_id)

st.sidebar.subheader("Existing Sessions")
for session_id in st.session_state.messages:
    if st.sidebar.button(f"{session_id} >", key=session_id, use_container_width=True):
        st.session_state.active_session_id = session_id
        st.write(f"Active session ID set to: {st.session_state.active_session_id}")
        for message in st.session_state.messages[st.session_state.active_session_id]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

if "user_history" not in st.session_state:
    st.session_state.user_history = {"default": []}


for message in st.session_state.messages[st.session_state.active_session_id]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


## SOME HELPING FUNCTIONS
def is_valid_uuid(value):
    try:
        # Check if value is a valid UUID (either UUID object or string)
        uuid_obj = uuid.UUID(value)
        return True
    except ValueError:
        return False

def decomposing_question_for_specific_db(question):
    decompose_prompt = PromptTemplate(
        input_variables=["question"],
        template="""
            USER INPUT: {question}
            You are a smart assistant that specializes in breaking down a complex question into smaller, domain-specific questions. 
            Your task is to analyze the given question and create sub-questions for the domains: Google, Uber, and Tesla. 
            If the question is unrelated to a specific domain, return an empty string for that domain.

            But generate only when necessary. If the question is not related to any company, return NA.

            Example:
            Question: "What is the difference between Tesla and Google?"
            Breakdown: {{
                "google": "What is Google?",
                "uber": "",
                "tesla": "What is Tesla?"
            }}

            Now, analyze the following question and generate sub-questions for each domain is the JSON format as above, 
            dont write anything else strictly:
            Question: '""" + question + """'
            
            Also keep in mind to generate the responses as a company and please align with the language of original question only dont use any other terms of your own if possible.
            
        """
    )
    
    # Load the QA chain with the correct document_variable_name
    decompose_chain = load_qa_chain(
        llm=llama,
        chain_type="stuff",
        prompt=decompose_prompt,
        document_variable_name="question"  # Align with the input variable
    )

    run_id = str(uuid.uuid4())
    if not is_valid_uuid(run_id):
        raise ValueError(f"Invalid run_id format: {run_id}")

    return decompose_chain.run({"input_documents": [], "run_id": str(run_id)})


def decision_agent(question):
    # DECOMPOSING QUESTION WITH LLM
    generated_decomposed_question_string = decomposing_question_for_specific_db(question)
    questions_dict = {}
    pattern = r'"(google|uber|tesla)":\s*"([^"]+)"'
    matches = re.findall(pattern,generated_decomposed_question_string)
    for company, question in matches:
        questions_dict[company] = question
    
    return questions_dict

def specific_db_agents(db_name, sub_question):
    persistent_dir = './embeddings/'+db_name
    db = Chroma(persist_directory=persistent_dir, embedding_function=embedding_instance)
    
    retriever = db.as_retriever(search_kwargs={"k": 10})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llama, 
        chain_type="stuff",
        retriever=retriever,
    )
    response = qa_chain.invoke(sub_question)
    return response['result']

def retrieval_agents(questions_dict):
    response = {}
    for company, question in questions_dict.items():
        response[company] = specific_db_agents(company, question)
    return response

def get_user_history(session_id):
    if session_id not in st.session_state.user_history:
        st.session_state.user_history[session_id] = []
    return st.session_state.user_history[session_id]

def set_user_history(session_id, history):
    st.session_state.user_history[session_id] = history
    return

def final_agent(data, question, session_id="default"):
    history = get_user_history(session_id)
    print(f"Start: {history}")
    prompt = ChatPromptTemplate.from_template(
        """
            You are an helpful question answering agent. You are provided with the question and the data in the following format:
            <Company Name>: <Data>,
            <Company Name>: <Data>

            Now, You have to answer the question based on the data provided, please dont make up any data and answer the question based on the data provided strictly.
            Strictly stick to the data provided and if the data is not sufficient to answer the question, please say that you don't have enough data to answer the question.
            And if the data is not given for the perticular company just say that data is not provided for the perticular company.
            And if the question is not related to the data provided, please say that you don't have enough data to answer the question.
            Keep these points into strict consideration while answering the question.

            Also keep in mind that the output you generate should strictly answer the question not just the summary of the data.

            Question: {question}
            Data: {data}

            History: {history}
        """
    )

    formatted_prompt = prompt.format(
        question=question,
        data=data,
        history='\n'.join(history)    
    )

    llm = llama
    response = llm(formatted_prompt)
    history.append(f"User: {question}")
    history.append(f"Bot: {response}")
    print(f"Mid: {history}")

    set_user_history(session_id, history)
    return response

def full_chat_bot(question, session_id="default"):
    questions_dict = decision_agent(question)
    print(f'Decision Agent work done: {questions_dict}')
    
    retrieval_agent_responses = retrieval_agents(questions_dict)
    print(f'Retrieval Agent work done: {retrieval_agent_responses}')

    formatted_retrieval_agent_responses = ""
    for company, response in retrieval_agent_responses.items():
        formatted_retrieval_agent_responses += f"\n{company} : {response}\n"

    print(f'Formatted Retrieval Agent work done: {formatted_retrieval_agent_responses}')

    return final_agent(data=formatted_retrieval_agent_responses, question=question)
   

if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages[st.session_state.active_session_id].append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    ## GENERATING THE RESPONSE
    response = full_chat_bot(prompt)


    with st.chat_message("assistant"):
        st.write(response)
    st.session_state.messages[st.session_state.active_session_id].append({"role": "assistant", "content": response})
