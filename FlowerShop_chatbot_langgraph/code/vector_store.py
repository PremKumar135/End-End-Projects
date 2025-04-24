from chromadb import PersistentClient, EmbeddingFunction, Embeddings
from langchain_openai import OpenAIEmbeddings
from typing import List
import json
import os
from dotenv import load_dotenv

#loading openai api_key
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# constants
MODEL_NAME = 'text-embedding-3-large'
DB_PATH = './.chroma_db'
FAQ_FILE_PATH = './data/FAQ.json'
INVENTORY_FILE_PATH = './data/inventory.json'

## Load data 
class LoadData:
    def __init__(self):
        pass

    def load_faq_data(self, faq_path):
        with open(faq_path, 'r') as f:
            faqs_json = json.load(f)
        faq_docs = [faq['question'] for faq in faqs_json] + [faq['answer'] for faq in faqs_json]
        return faqs_json, faq_docs
    
    def load_inventory(self, inventory_path):
        with open(inventory_path, 'r') as f:
            inventory_json = json.load(f)
        inventory_docs = [data['description'] for data in inventory_json]
        return inventory_json, inventory_docs



## Embedding Class to embedding the text
class MyEmbeddingClass(EmbeddingFunction):
    def __init__(self, model_name):
        self.embedding_model = OpenAIEmbeddings(model=model_name)
    
    def __call__(self, input_texts:List[str])->Embeddings:
        embedding_list = []
        for text in input_texts:
            text_embed = self.embedding_model.embed_query(text)
            embedding_list.append(text_embed)
        return embedding_list


        
## FlowerShop Vector Store
class FlowerShopVectorStore:
    def __init__(self):
        db = PersistentClient(path=DB_PATH)
        custom_embedding_function = MyEmbeddingClass(MODEL_NAME)

        # create collection 
        self.faq_collection = db.get_or_create_collection(name='FAQ', embedding_function = custom_embedding_function)
        self.inventory_collection = db.get_or_create_collection(name='inventory', embedding_function = custom_embedding_function)


    # load data in collection
    def load_data_in_collection(self, faq_path, inventory_path):
        load_data = LoadData()

        if self.faq_collection.count()==0:
            faq_json, faq_docs = load_data.load_faq_data(faq_path)
            self.faq_collection.add(documents = faq_docs,
                                    ids = [str(i) for i in range(0, len(faq_docs))],
                                    metadatas = faq_json + faq_json
                                    )
            
            print('Done 1')
        
        if self.inventory_collection.count()==0:
            inventory_json, inventory_docs = load_data.load_inventory(inventory_path)
            self.inventory_collection.add(documents = inventory_docs,
                                        ids = [str(i) for i in range(len(inventory_docs))],
                                        metadatas = inventory_json
                                        )
            print('Done 2')
            
    def query_faqs(self, query:str):
        return self.faq_collection.query(query_texts=[query], n_results=5)
    
    def query_inventories(self, query:str):
        return self.inventory_collection.query(query_texts=[query], n_results=5)


