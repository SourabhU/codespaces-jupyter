import os
import azure.functions as func
import logging
"""
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline

app = func.FunctionApp()

@app.blob_trigger(
    arg_name = 'myblob',
    path = 'pdfstorage/input_files/{name}.pdf',
    connection = 'storagefordocuments_STORAGE'
)

def BlobTrigfunc(myblob: func.InputStream):
    logging.info(
        f'Trigger is triggered'
        f'Blob name: {myblob.name}'
        f'Blob size: {myblob.length} bytes'
    )

#Loading the PDF file
file_path = 'pdfstorage/input_files/{name}.pdf'
loader = PyPDFLoader(file_path)
pdf_file = loader.load()

#Check pages count
#len(pdf_file)

#Check page contents and metadata
print(pdf_file[0].page_content)
#print(pdf_file[0].metadata)

#Split PDF file into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(pdf_file)

print(f"{len(chunked_documents)} chunks created")
#print(chunked_documents[0])

#Embed the created chunks using AzureOpenAIEmbeddings
embedding_model = AzureOpenAIEmbeddings(
    azure_deployment='pdf_embeddings'
)

#AzureOpenAIEmbeddings
chunk_list = []
for chunk in chunked_documents:
    chunk_list.append(chunk.page_content)

#print(chunk_list)

embedded_document = embedding_model.embed_documents(chunk_list)
print(embedded_document)