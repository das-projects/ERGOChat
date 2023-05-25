import openai
import os
import pathlib
import streamlit as st
from streamlit_chat import message
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT


def create_vectordb(pdf_folder_path=f'data/'):
    loader = PyPDFDirectoryLoader(pdf_folder_path)
    docs = loader.load()
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory=".")
    vectordb.persist()


def create_chatbot():
    embeddings = OpenAIEmbeddings()
    llm = OpenAI(temperature=0.8)
    vectordb = Chroma(persist_directory=".", embedding_function=embeddings)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")

    qa = ConversationalRetrievalChain(
        retriever=vectordb.as_retriever(),
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        memory=memory,
    )

    return qa


# generate a response
def generate_response(query, qa):
    chat_history = list(zip(st.session_state['past'], st.session_state['generated']))
    response = qa({"question": query, "chat_history": chat_history})
    st.session_state['messages'].append({"role": "assistant", "content": response['answer']})
    return response


# create_vectordb(pdf_folder_path=f'data/')


# Setting page title and header
st.set_page_config(page_title="ERGOChat", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>ERGOChat - All ERGO products on your fingertips 😬</h1>",
            unsafe_allow_html=True)

# Set org ID and API key
openai.organization = st.secrets['OPENAI_ORG']
openai.api_key = st.secrets['OPENAI_API_KEY']

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
qa = create_chatbot()

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
st.sidebar.title("Sidebar")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    qa = create_chatbot()


# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = generate_response(user_input, qa)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
