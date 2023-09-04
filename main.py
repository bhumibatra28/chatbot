import logging
import sys
import torch
from langchain.llms import CTransformers
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from llama_index import GPTListIndex, SimpleDirectoryReader, ServiceContext,GPTSimpleVectorIndex
from huggingface_hub import hf_hub_download
from langchain.llms import HuggingFacePipeline
documents = SimpleDirectoryReader("/customchatbotques").load_data()
from llama_index.prompts.prompts import SimpleInputPrompt
system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."
# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext

embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
)
llm = load_llm()
service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)
index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine()
response = query_engine.query("What is amazon prime")
print(response)