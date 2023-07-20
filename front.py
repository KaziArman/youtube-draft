import openai
import streamlit as st
import utils as utl
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from youtube_transcript import YT_transcript
from chunk import chunking

ss = st.session_state
st.set_page_config(
	#page_icon="https://i.pinimg.com/originals/4b/85/c9/4b85c95c93eff810b0fe0a755be081a6.png",
	page_icon="web.png",
	layout="wide",
	page_title='Ask Youtube',
	initial_sidebar_state="expanded"
)
st.markdown(f'<div class="header"><figure><img src="https://i.pinimg.com/originals/41/f6/4d/41f64d3b4b21cb08eb005b11016bf707.png" width="500"><figcaption><h1>Welcome to Ask Youtube</h1></figcaption></figure><h3>Ask Youtube is a conversional AI based tool, where you can ask about any youtube video and it will answer.</h3></div>', unsafe_allow_html=True)
#st.markdown(f'<div class="header"><figure><img src="logo.png" width="500"><figcaption><h1>Welcome to Ask Youtube</h1></figcaption></figure><h3>Ask Youtube is a conversional AI based tool, where you can ask about any youtube video and it will answer.</h3></div>', unsafe_allow_html=True)

with st.expander("How to use Ask Youtube 🤖", expanded=False):

	st.markdown(
		"""
		Please refer to [our dedicated guide](https://www.impression.co.uk/resources/tools/oapy/) on how to use Ask Youtube.
		"""
    )

with st.expander("Credits 🏆", expanded=True):

	st.markdown(
		"""
		Ask Youtube was created by [Kazi Arman Ahmed](https://www.linkedin.com/in/r4h4t/) and [Md Shamim Hasan](https://www.linkedin.com/in/md-shamim-hasan/)  at [LandQuire](https://www.linkedin.com/company/landquire/) in Bangladesh.
	    """
    )

#st.markdown("---")
# Load your API key
api_key = st.text_input('Enter your API key')
# print(api_key)
openai.api_key = api_key
os.environ["OPENAI_API_KEY"] = api_key

# Load your Youtube Video Link
youtube_link = st.text_input('Enter your YouTube video link')

temp_slider = st.sidebar.slider('Set the temperature of the completion. Higher values make the output more random,  lower values make it more focused.', 0.0, 1.0, 0.7)
def ui_question():
	st.write('### Ask Questions')
	disabled = False
	st.text_area('question', key='question', height=100, placeholder='Enter question here', help='', label_visibility="collapsed", disabled=disabled)
ui_question()

def output_add(q,a):
	if 'output' not in ss: ss['output'] = ''
	q = q.replace('$',r'\$')
	a = a.replace('$',r'\$')
	new = f'#### {q}\n{a}\n\n'
	ss['output'] = new + ss['output']
	st.markdown(new)
generate = st.button('Generate!')
def ui_output():
	output = ss.get('output','')
	st.markdown(output)
ui_output()
if generate:
	with st.spinner('Classifying...'):
		openai.api_key = api_key
		os.environ["OPENAI_API_KEY"] = api_key
		yt_script = YT_transcript(youtube_link)
		transcript = yt_script.script()
		text = "\n\n\nI have given you the caption of a Youtube video. " \
			   "I will ask you specific question about this video. " \
			   "You have to understand the theme of the video and" \
			   " answer me very precisely according to the questions"
		transcript = transcript+text
		chnk = chunking(transcript)
		chunks = chnk.yt_data()
		# print(chunks)
		# Get embedding model
		embeddings = OpenAIEmbeddings()
		# Create vector database
		chat_history=[]
		question = ss.get('question', '')
		temperature = ss.get('temperature', 0.0)
		temperature = temp_slider
		model = OpenAI(model="text-davinci-003", temperature=temp_slider)
		db = FAISS.from_documents(chunks, embeddings)
		qa = ConversationalRetrievalChain.from_llm(model, db.as_retriever())
	result = qa({"question": question, "chat_history": chat_history})
	chat_history.append((question, result['answer']))
	# print("\n\n\n\n")
	# print(chat_history)
	q = question.strip()
	a = result['answer'].strip()
	ss['answer'] = a
	output_add(q,a)

# Loading CSS
utl.local_css("frontend.css")
utl.remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')
utl.remote_css('https://fonts.googleapis.com/css2?family=Red+Hat+Display:wght@300;400;500;600;700&display=swap')
