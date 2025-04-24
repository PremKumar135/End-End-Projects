import streamlit as st
from vector_store import FlowerShopVectorStore
from chatbot import app
from langchain_core.messages import HumanMessage, AIMessage

# set page config
st.set_page_config(layout='wide', page_icon='🌹', page_title='Flower Shop Chatbot')


# initializing session state with message history
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = [AIMessage(content='Hiya, welcome to flower shop chatbot. How can i help you?')]


# dividing screen into three columns
left_col, main_col, right_col = st.columns([1,2,1])


# 1.Left Col: Clear chat, Collection Choicee
with left_col:
    if st.button('Clear Chat'):
        initial_message = AIMessage(content='Hiya, welcome to flower shop chatbot. How can i help you?')
        st.session_state['message_history'] = [initial_message.content]


# 2.Main Col: Main chatbot
with main_col:
    user_input = st.chat_input("Type Here...")
    if user_input:
        st.session_state['message_history'].append(HumanMessage(content=user_input))
        response = app.invoke({'messages':st.session_state['message_history']})
        st.session_state['message_history'] = response['messages']

    for i in range(1, len(st.session_state['message_history'])+1):
        this_message = st.session_state['message_history'][-i]
        if isinstance(this_message, AIMessage):
            message_box = st.chat_message('assistant')
        else:
            message_box = st.chat_message('user')
        message_box.markdown(this_message.content)
        

# 3.Right Col: Printing session history
with right_col:
    st.text(st.session_state['message_history'])