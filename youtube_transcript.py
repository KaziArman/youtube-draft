class YT_transcript:
    def __init__(self,lnk):
        self.lnk = lnk

    def script(self):
        from llama_index import download_loader
        import os
        import pandas as pd
        import matplotlib.pyplot as plt
        from transformers import GPT2TokenizerFast
        from langchain.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.embeddings import OpenAIEmbeddings
        from langchain.vectorstores import FAISS
        from langchain.chains.question_answering import load_qa_chain
        from langchain.llms import OpenAI
        from langchain.chains import ConversationalRetrievalChain
        # Step 2: Save to .txt and reopen (helps prevent issues)
        #with open('output.txt', 'r') as f:
        #    text = f.read()
        YoutubeTranscriptReader = download_loader("YoutubeTranscriptReader")
        loader = YoutubeTranscriptReader()
        documents = loader.load_data(ytlinks=[self.lnk])
        text=documents[0].text
        return text


    '''# Step 3: Create function to count tokens
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))
    
    # Step 4: Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 512,
        chunk_overlap  = 52,
        length_function = count_tokens,
    )
    
    chunks = text_splitter.create_documents([text])'''