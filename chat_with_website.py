import os
import bs4
import streamlit as st
from langchain import hub
from langchain_community.utilities import SQLDatabase
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from constant import LANGUAGE_MODEL, EMBEDDING_MODEL
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def get_vectorstore_from_urls(urls):
    # Load the webpages
    loaders = [WebBaseLoader(url) for url in urls]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    document_chunks = text_splitter.split_documents(documents)
    
    # Create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings(model=EMBEDDING_MODEL))

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(temperature=0, model=LANGUAGE_MODEL)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    llm = ChatOpenAI(temperature=0, model=LANGUAGE_MODEL)
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input, vector_store, chat_history):
    retriever_chain = get_context_retriever_chain(vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    })
    
    return response['answer']

def main():

    st.header("_Chat with the_ :blue[websites]", divider="green")
    st.write("Enter website links and ask questions")

    # Sidebar
    with st.sidebar:
        st.header("Menu")

        # Text area for website URLs
        website_urls = st.text_area("Enter website links (one per line)", value='\n'.join(st.session_state.get('website_urls', [])))
        
        if st.button("Submit"):
            if website_urls:
                urls = [url.strip() for url in website_urls.split('\n') if url.strip()]
                st.session_state.website_urls = urls
                with st.spinner("Processing website links..."):
                    st.session_state.vector_store = get_vectorstore_from_urls(urls)
                st.success(f"Processed {len(urls)} website links successfully!")
            else:
                st.info("Please enter a webite link")

    if 'website_urls' not in st.session_state or not st.session_state.website_urls:
        # st.markdown("---")
        st.info("Please enter website links and click 'Submit'")

    elif 'vector_store' not in st.session_state:
        # st.markdown("---")
        st.warning("Please click 'Submit' button to start")

    else:
        # Display processed websites
        # st.markdown("---")

        # Session state
        if "chat_history" not in st.session_state:
            # st.markdown("---")
            st.session_state.chat_history = [
                AIMessage(content="Hello, I am a chatbot. How can I help you with links you provided?"),
            ]

        # User input
        user_query = st.chat_input("Enter your message here...")
        if user_query is not None and user_query != "":
            with st.spinner("Generating response..."):
                response = get_response(user_query, st.session_state.vector_store, st.session_state.chat_history)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))
            
        # Conversation
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)

if __name__ == "__main__":
    main()