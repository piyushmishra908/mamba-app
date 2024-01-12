import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import streamlit_authenticator as stauth
import yaml
import numpy as np
from yaml.loader import SafeLoader

# Config
st.set_page_config(page_title='MambaQA', layout='wide')


with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)


authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# You can always call this function where ever you want

def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

st.sidebar.image(add_logo(logo_path="img/emergentq_logo.png", width=220, height=60))

st.sidebar.title('EmergentQ AI')

name, authentication_status, username = authenticator.login('Login', 'main')

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("havenhq/mamba-chat")
tokenizer.eos_token = "<|endoftext|>"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.chat_template = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").chat_template

model = MambaLMHeadModel.from_pretrained("havenhq/mamba-chat", device="cuda", dtype=torch.float16)

def mamba_chat_output(messages):
    # messages.append(dict(role="user",
    #     content=user_message
    #     ))

    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")

    out = model.generate(input_ids=input_ids, max_length=2000, temperature=0.9, top_p=0.7, eos_token_id=tokenizer.eos_token_id)

    decoded = tokenizer.batch_decode(out)
    messages.append(dict(
        role="assistant",
        content=decoded[0].split("<|assistant|>\n")[-1])
    )

    print("Model:", decoded[0].split("<|assistant|>\n")[-1])

    assistant_message = decoded[0].split("<|assistant|>\n")[-1]

    return assistant_message

if st.session_state["authentication_status"]:
    st.sidebar.header("QA-V1.1")
    
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.write(f'Welcome *{st.session_state["name"]}*')

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

        for response in mamba_chat_output(
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]):
            full_response += (response or "")
            message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

elif st.session_state["authentication_status"] == False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] == None:
    st.warning('Please enter your username and password')
