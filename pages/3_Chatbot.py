import streamlit as st
from streamlit_chat import message
from datetime import datetime
from gpt_index import GPTSimpleVectorIndex, LLMPredictor
import os
from dotenv import load_dotenv
from langchain import OpenAI
from modules.utils import Utilities





st.markdown("""
    <style>
        body {
            background-image: url("https://www.zfort.com/media/cache/blogArticle/images/blog/5ece5dfcccab1_top_10_AI_chatbots.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .stApp {
            background-color: rgba(255, 255, 255, 0.3);
        }
    </style>
""", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: Blue;'>Plant Disease Classification Chatbot👋</h1>", unsafe_allow_html=True)

st.markdown("<p>The Plant Disease Chatbot, PlantBot, is a chatbot developed using GPT Index and LangChain libraries. It serves as a convenient chatbot for plant enthusiasts, gardeners, and farmers to get accurate information regarding different plant diseases, their causes, symptoms, and impacts. The chatbot can also provide recommendations for the preventive measures and treatment of various plant diseases.</p>", unsafe_allow_html=True)

    

























load_dotenv()



utils = Utilities()

user_api_key = utils.load_api_key()

if not user_api_key:
    st.write("Please provide your OpenAI API key to communicate with chat bot.")
    st.stop()
else:
    os.environ["OPENAI_API_KEY"] = user_api_key

# Get the path of the current file
current_path = os.path.dirname(os.path.abspath(__file__))

# Specify the relative path to the vectorIndex.json file
vector_index_path = os.path.join(current_path, "vectorIndex.json")


llmPredictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=os.environ["OPENAI_API_KEY"]))

def get_bot_response(user_query):
    vIndex = GPTSimpleVectorIndex.load_from_disk(vector_index_path)
    response = vIndex.query(user_query, response_mode="compact")
    return str(response)

def display_messages(all_messages):
    for idx, msg in enumerate(all_messages):
        if msg['user'] == 'user':
            message(f"User \n {msg['text']}", is_user=True, key=f"user-{idx}")
        else:
            message(f"Bot \n  {msg['text']}", key=f"bot-{idx}")

def send_message(user_query, all_messages):
    if user_query:
        #current_time = datetime.now().strftime("%H:%M:%S")
        all_messages.append({'user': 'user',  'text': user_query})
        bot_response = get_bot_response(user_query)
        all_messages.append({'user': 'bot',  'text': bot_response})

        st.session_state.all_messages = all_messages
        display_messages(all_messages)

if 'all_messages' not in st.session_state:
    st.session_state.all_messages = []
st.header(":blue[Please start conversion with plant chat bot regarding the plant disease!]:sunglasses:")

user_query = st.text_input("You: ", "", key="input")
send_button = st.button("Send")

if send_button:
    if not user_api_key:
        st.error("API key is missing. Please provide your OpenAI API key.")
    else:
        send_message(user_query, st.session_state.all_messages)
