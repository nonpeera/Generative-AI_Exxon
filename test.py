from flask import Flask, request, abort
import requests
import json

import os
import getpass
import requests
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import DocArrayInMemorySearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_text_splitters import CharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.document_loaders import TextLoader

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display, Markdown

import json
from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key")
    
model = ChatGoogleGenerativeAI(model="gemini-pro")
print(
    model(
    [
        HumanMessage(content="Answer with Simple 'Yes' or 'No'. Question: Is apple a Fruit?"),
    ]
).content
)

import os
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# Set up Azure Form Recognizer
endpoint = 'https://exxontugoatdi.cognitiveservices.azure.com/'
api_key = '40ed1c5af99c4439a6e5cee00729d205'
document_analysis_client = DocumentAnalysisClient(endpoint, AzureKeyCredential(api_key))

# Directory containing PDF files
pdf_directory = r"C:\Users\tanat\Downloads\Generative-AI_Exxon-main\doc\PRODUCT"

# Function to read and extract text from a PDF file
def read_pdf(file_path):
    with open(file_path, "rb") as f:
        poller = document_analysis_client.begin_analyze_document("prebuilt-read", document=f)
        result = poller.result()

    pdf_text = ""
    for page in result.pages:
        for line in page.lines:
            pdf_text += line.content + "\n"
    return pdf_text

# List to store all extracted texts
all_texts = []

# Read all PDF files in the directory
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, filename)
        extracted_text = read_pdf(pdf_path)

        # Display the extracted text for verification
        print(f"Extracted text from {filename}:\n{extracted_text}\n")

        # Add the extracted text to the list
        all_texts.append(extracted_text)

# Step 3: Create text chunks for each document using langchain
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = [chunk for text in all_texts for chunk in text_splitter.split_documents([Document(page_content=text)])]

# Step 4: Embedding and Vectorstore
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(docs, embedding=embeddings)
retriever = vectorstore.as_retriever()

llm_text = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
template = """
```
{context}
```

{information}

You are ExxonMobil Employ try to sell product
Provide ExxonMobilThailand information and store location.
ถ้าลูกค้าถามน้ำมันที่เหมาะกับรถตัวเองให้แนะนำ 3 product พร้อมเปรียบเทียบกัน
ใช้ emoji บางครั้งให้ดูเป็นกันเอง
อธิบายให้เข้าใจง่ายแล้วเห็นภาพ
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "information": RunnablePassthrough()}
    | prompt
    | llm_text
    | StrOutputParser()
)

llm_vision = ChatGoogleGenerativeAI(model="gemini-pro-vision", temperature=0.0)
full_chain = (
    RunnablePassthrough() | llm_vision | StrOutputParser() | rag_chain
)

app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def webhook():
    if request.method == 'POST':
        payload = request.json
        Reply_token = payload['events'][0]['replyToken']
        print(Reply_token)
        message = payload['events'][0]['message']['text']
        print(message)
        
        text = message
        Reply_message = rag_chain.invoke(text)
        Line_Access_Token = 'eLfr6uht2e/SeGjFvGsWn+qbKryF4mO58j6W+qDrDvmBxFG8SnIr7/+pgrszGQ9BTneL/+bFszzUkV6w4OUF3ouqo0DndUZC2UEZgmPriYVqrD132rDyCiV19rc+1v8dMNHNZbk5eYM6XzAKrE9ztQdB04t89/1O/w1cDnyilFU='  # แทนที่ด้วย Channel Access Token ของคุณ
        ReplyMessage(Reply_token, Reply_message, Line_Access_Token)
            
        return request.json, 200
    else:
        abort(400)

def ReplyMessage(Reply_token, TextMessage, Line_Access_Token):
    LINE_API = 'https://api.line.me/v2/bot/message/reply'
    Authorization = 'Bearer {}'.format(Line_Access_Token)
    print(Authorization)
    headers = {
        'Content-Type': 'application/json; charset=UTF-8',
        'Authorization': Authorization
    }
    data = {
        "replyToken": Reply_token,
        "messages": [{
            "type": "text",
            "text": TextMessage
        }]
    }
    data = json.dumps(data)
    r = requests.post(LINE_API, headers=headers, data=data)
    
    return r.status_code  # ส่งค่า status code กลับ

if __name__ == '__main__':
    app.run(debug=True)
