import os
import tempfile
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from langchain.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import Tool, AgentExecutor, create_openai_tools_agent
from langchain.tools import StructuredTool
from constant import LANGUAGE_MODEL
from functools import partial
from dotenv import load_dotenv
import langchain
langchain.debug = False

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(model_name=LANGUAGE_MODEL, api_key=openai_api_key)

def save_uploaded_csv(uploadedfile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(uploadedfile.read())
        return tmp_file.name

def convert_csv_to_sql(csv_path):
    df = pd.read_csv(csv_path)
    engine = create_engine("sqlite:///uploaded_data.db")
    df.to_sql("uploaded_table", engine, index=False, if_exists="replace")
    db = SQLDatabase(engine=engine)
    return db

def run_agent(question):
    agent_executor = create_sql_agent(llm, db=SQLDatabase(engine=create_engine("sqlite:///uploaded_data.db")), agent_type="openai-tools", verbose=True)
    result = agent_executor.invoke({'input': question})
    return result['output']

def create_sql_tool():
    sql_database_tool = StructuredTool.from_function(
        name="SQL_DATABASE_TOOL",
        func=run_agent,
        description="Use this tool to answer user questions from the SQL database."
    )
    return sql_database_tool

def get_answer(question, chat_history):
    tool = [create_sql_tool()]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are an intelligent chatbot whose task is to help users answer their queries.
            Users will provide you with a question, and you need to answer it.
            You have access to `SQL_DATABASE_TOOL`, which you can use to answer the question from the SQL database."""),
            ("placeholder", "{chat_history}"),
            ("user", "{question}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_openai_tools_agent(llm, tool, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)
    result = agent_executor.invoke(
        {
            "question": question,
            "chat_history": chat_history,
        }
    )
    return result["output"]


def main():

    st.header("_Chat with your_ :blue[CSV]", divider="green")
    st.write("Upload a CSV file and ask questions")

    with st.sidebar:
        st.header("Menu")

        uploaded_files = st.file_uploader("Upload a CSV file and click on 'Submit' button", type=["csv"])
        
        if st.button("Submit"):
            if uploaded_files:
                files = save_uploaded_csv(uploaded_files)
                st.session_state.file_paths = files
            
                with st.spinner("Processing CSV file..."):
                    st.session_state.db = convert_csv_to_sql(st.session_state.file_paths)
                st.sidebar.success("Processed CSV file successfully!")
            else:
                st.info("Please choose a CSV file")

    if 'file_paths' not in st.session_state or not st.session_state.file_paths:
        # st.markdown("---")
        st.info("Please upload CSV file and click 'Submit'")

    elif 'db' not in st.session_state:
        # st.markdown("---")
        st.warning("Please click 'Submit' button to start")

    else:
        # Display processed websites
        # st.markdown("---")

        # Session state
        if "chat_history" not in st.session_state:
            st.markdown("---")
            st.session_state.chat_history = [
                AIMessage(content="Hello, I am a chatbot. How can I help you with your csv file?"),
            ]

        # User input
        user_query = st.chat_input("Enter your message here...")
        if user_query is not None and user_query != "":
            with st.spinner("Generating response..."):
                response = get_answer(user_query, st.session_state.chat_history)
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
