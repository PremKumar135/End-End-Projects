from langchain_core.tools import tool
from typing import List, Dict
from vector_store import FlowerShopVectorStore

vector_store = FlowerShopVectorStore()

@tool
def query_knowledge_base_tool(query:str):
    """
    Looks up the information in knowledge base to help with answering customer questions and getting information on business processes.
    
    Args:
        query(str) : Question to ask the knowledge base
    
    Return:
        Dict[str, str] : Potentially relevant question and answering pairs for the knowledge base
    """
    return vector_store.query_faqs(query=query)


@tool
def search_for_product_recommendations_tool(description:str):
    """
    Looks up information in a knowledge base to help with product recommendation for customers. For example:
    <example>
        1. Boquets are suitable for birthdays, maybe with red flowers
        2. A large boquet for a wedding
        3. A cheap boquet with wildflowers
    </example>

    Args:
        description (str) : Description of product features

    Return:
        Dict[str, str] : Potentially relevant features
    """
    return vector_store.query_inventories(query=description)