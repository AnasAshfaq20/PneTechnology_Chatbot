import streamlit as st
import os
from langchain.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message

groqapi_key = "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"

def main():
    load_dotenv()
    st.header("Pne Technology")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    # Always process the web links and create the vectorstore on the fly
    links = [
        "https://pnetechnology.com/about-us/",
        "https://pnetechnology.com/our-services/",
        "https://pnetechnology.com/contact-us/",
    ]
    web_text = get_web_text(links)
    text_chunks = get_text_chunks(web_text)
    vectorstore = get_vectorstore(text_chunks)

    st.session_state.conversation = get_conversation_chain(vectorstore, groqapi_key)
    st.session_state.processComplete = True

    if st.session_state.processComplete:
        user_question = st.chat_input("Ask a question about your files.")
        if user_question:
            handle_userinput(user_question)

    # Display chat history
    display_chat_history()

def get_web_text(links):
    text = ""
    for link in links:
        loader = WebBaseLoader(link)
        documents = loader.load()
        # Extract and join the content from web pages
        text += " ".join([doc.page_content for doc in documents])
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=300,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore

from langchain.prompts import PromptTemplate

def get_conversation_chain(vectorstore, openai_api_key):
    # Define a custom prompt template for confident responses
    prompt_template = """
    You are an expert assistant providing precise and confident responses. 
    Answer user queries directly and assertively. Avoid speculative phrases.
    Question: {question}
    Context: {context}
    Answer:
    """
    PROMPT = PromptTemplate(
        input_variables=["question", "context"],
        template=prompt_template,
    )

    # Create the LLM
    llm = ChatGroq(groq_api_key=groqapi_key, model_name="mixtral-8x7b-32768", temperature=0) # type: ignore
    
    # Create the conversational chain with the custom prompt
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},  # Apply the custom prompt here
    )
    return conversation_chain



def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history.extend(response['chat_history'])

def display_chat_history():
    response_container = st.container()
    with response_container:
        for i, message_content in enumerate(st.session_state.chat_history):
            # Display user messages on the right
            if message_content.type == "human":
                message(message_content.content, is_user=True, key=str(i) + '_user')
            # Display bot messages on the left
            else:
                message(message_content.content, key=str(i) + '_bot')

if __name__ == '__main__':
    main()
