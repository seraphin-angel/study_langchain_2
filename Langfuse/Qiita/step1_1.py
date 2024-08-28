from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
]

url = (
    "https://www.apple.com/newsroom/2023/09/apple-debuts-iphone-15-and-iphone-15-plus/"
)

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs = html_splitter.split_text_from_url(url)

# embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)
# db = Chroma.from_documents(docs, embeddings)

db.save_local("faiss_index")
