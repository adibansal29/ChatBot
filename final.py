import streamlit as st
import os
import cohere
import json
import requests
from dotenv import load_dotenv, find_dotenv
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, HumanMessage
import pandas as pd
import matplotlib.pyplot as plt

# Load the API key from the .env file
_ = load_dotenv(find_dotenv())
cohere.api_key  = os.environ['COHERE_API_KEY']

# Load the embeddings
persist_directory = "DB"
embedding=CohereEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
retriever=vectordb.as_retriever()

# Define the LLM
llm=ChatCohere()

# Define the tools
@tool
def get_installer_info(zip:int):
    """Get the info of some of the installers in the given zip code and always return it in a table with the website url, phone no and email"""

    data = requests.get(f"http://127.0.0.1:5000/api/{zip}").json()
    installer_info={}
    if len(data)==0:
        return "No installers could be found"
    
    else:
        for i in range(len(data)):
            installer_info[f"Installer {i+1}"] = {
            "Name": data[i]["name"],
            "Website url" : data[i]["url"],
            "Phone no." : data[i]["support_phone"],
            "Email" : data[i]["support_email"]
            }

    return json.dumps(installer_info)

@tool
def anomaly_check(date:str):
    """Find potential anomalies in the solar system of the user on a given date by running some ML models"""
    data = requests.get(f"http://127.0.0.1:8000/predict/{date}").json()
    micro_df = pd.DataFrame.from_records(data["Microinverter Dataframe"])
    meter_df1 = pd.DataFrame.from_records(data["Meter Dataframe 1"])
    meter_df2 = pd.DataFrame.from_records(data["Meter Dataframe 2"])
    micro_anomalies = micro_df[micro_df['anomaly'] == True]
    meter_anomalies1 = meter_df1[meter_df1['anomaly'] == True]
    meter_anomalies2 = meter_df2[meter_df2['anomaly'] == True]
    micro_anomalies_count = micro_anomalies.shape[0]
    meter_anomalies_count1 = meter_anomalies1.shape[0]
    meter_anomalies_count2 = meter_anomalies2.shape[0]

    # Plotting the graphs
    plt.figure(figsize=(10, 6))
    plt.plot(pd.to_datetime(micro_df['date']), micro_df['energy_produced'], label='Energy Produced')
    plt.scatter(pd.to_datetime(micro_anomalies['date']), micro_anomalies['energy_produced'], color='red', label='Anomalies')
    plt.xlabel('Date')
    plt.ylabel('Energy Produced')
    plt.title('Energy Produced by Microinverter Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('micro.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(pd.to_datetime(meter_df1['date']), meter_df1['curr_w'], label='Current')
    plt.scatter(pd.to_datetime(meter_anomalies1['date']), meter_anomalies1['curr_w'], color='red', label='Anomalies')
    plt.xlabel('Date')
    plt.ylabel('Current')
    plt.title('Current At Production Meter Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('meter1.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(pd.to_datetime(meter_df2['date']), meter_df2['curr_w'], label='Current')
    plt.scatter(pd.to_datetime(meter_anomalies2['date']), meter_anomalies2['curr_w'], color='red', label='Anomalies')
    plt.xlabel('Date')
    plt.ylabel('Current')
    plt.title('Current At Consumption Meter Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('meter2.png')
    plt.close()

    # Returning the results
    result = {"Microinverter Dataframe": micro_df.to_dict(orient='records'),
              "Meter Dataframe 1": meter_df1.to_dict(orient='records'),
              "Meter Dataframe 2": meter_df2.to_dict(orient='records'),
              "Microinverter Anomalies": micro_anomalies.to_dict(orient='records'),
              "Meter Anomalies 1": meter_anomalies1.to_dict(orient='records'),
              "Meter Anomalies 2": meter_anomalies2.to_dict(orient='records'),
              "Microinverter Anomalies Count": micro_anomalies_count,
              "Meter Anomalies Count 1": meter_anomalies_count1,
              "Meter Anomalies Count 2": meter_anomalies_count2,
              "Microinverter Graph": 'micro.png',
              "Meter Graph 1": 'meter1.png',
              "Meter Graph 2": 'meter2.png'}
    return json.dumps(result)

# Bind the tools to the LLM
tools = [get_installer_info,anomaly_check]
llm_with_tools = llm.bind_tools(tools)


# Define the memory
memory = ConversationSummaryBufferMemory(
    memory_key="chat_history",
    llm=llm,
    max_token_limit=100,
    return_messages=True
)

# llm_with_memory = llm_with_tools.bind(memory = memory)
# llm_with_retriever = llm_with_memory.bind(retriever = retriever)

# Define the QA chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    chain_type='map_reduce',
    memory=memory,
    return_source_documents=False
)


# Edit the style of the chatbot
st.markdown(
    """
    <style>
    .st-emotion-cache-h4xjwg.ezrtsby2{
        background-color: black !important;}
    .st-emotion-cache-6qob1r.eczjsme11{
        background-color: #FAF6EF !important;}
    .block-container.st-emotion-cache-1eo1tir.ea3mdgi5{
        background-color: #FAF6EF !important;}
    .main.st-emotion-cache-bm2z3a.ea3mdgi8{
        background-color: #FAF6EF !important;}
    .st-emotion-cache-qcqlej.ea3mdgi1{
        background-color: #FAF6EF !important;}
    .st-emotion-cache-0.e1f1d6gn0{
        background-color: #FAF6EF !important;}
    .st-emotion-cache-arzcut.ea3mdgi2{
        background-color: #FAF6EF !important;}
    .st-emotion-cache-vj1c9o.ea3mdgi6{
        background-color: black !important;}
    .st-emotion-cache-12fmjuu.ezrtsby2{

        background-color: black !important;}
    .st-emotion-cache-1uj96rm.ea3mdgi7{
        background-color: #F47721 !important;}
    .st-emotion-cache-uhkwx6.ea3mdgi6{
        background-color: #FAF6EF !important;}
    </style>
    """,
    unsafe_allow_html=True
)


# Add a sidebar
with st.sidebar:
    st.markdown("<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
    st.markdown("[Contact Us](https://enphase.com/en-in)")


# Add a title
st.markdown('<h1><span style="color: #F47721;">Enphase</span> Chatbot💬</h1>', unsafe_allow_html=True)


# Checking if the session state exists
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    
# Displaying all the messages in the session state
for msg in st.session_state.messages:
    if len(st.session_state.messages) == 1:
        st.chat_message(msg["role"]).write(msg["content"])
        # st.markdown("<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
    else:
        st.chat_message(msg["role"]).write(msg["content"])
        if "image1" in msg:
            st.image(msg["image1"], caption='Energy Produced vs Date')
        if "image2" in msg:
            st.image(msg["image2"], caption='Current vs Date')
        if "image3" in msg:
            st.image(msg["image3"], caption='Current vs Date')

# Getting the user input
if question := st.chat_input():
    i=1
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    # Getting the tool call from the LLM
    messages = [HumanMessage(question)]
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    # Checking if the LLM has no tool calls
    if ai_msg.tool_calls ==[]:
        res = qa_chain.invoke({"question":question})
        msg = res['answer']
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)

    # Checking if the LLM calls the get_installer_info tool
    elif ai_msg.tool_calls[0]["name"].lower() == "get_installer_info":
        args = ai_msg.tool_calls[0]['args']
        # print(args)
        answer = get_installer_info.invoke(args)
        messages.append(ToolMessage(answer, tool_call_id = ai_msg.tool_calls[0]["id"]))
        final = llm_with_tools.invoke(messages)
        msg = final.content
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
    
    # Checking if the LLM calls the anomaly_check tool
    else:
        args = ai_msg.tool_calls[0]['args']
        answer = anomaly_check.invoke(args)
        answer_dict = json.loads(answer)


        msg = (f" There are {answer_dict['Microinverter Anomalies Count']} anomalies in your Microinverter, {answer_dict['Meter Anomalies Count 1']} anomalies in Meter 49812427 and {answer_dict['Meter Anomalies Count 2']} anomalies in Meter 49812428")
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
        img1 = answer_dict['Microinverter Graph']
        st.session_state.messages[-1]["image1"] = img1
        st.image(img1, caption='Energy Produced vs Date')
        img2 = answer_dict['Meter Graph 1']
        st.session_state.messages[-1]["image2"] = img2
        st.image(img2, caption='Current vs Date')
        img3 = answer_dict['Meter Graph 2']
        st.session_state.messages[-1]["image3"] = img3
        st.image(img3, caption='Current vs Date')


