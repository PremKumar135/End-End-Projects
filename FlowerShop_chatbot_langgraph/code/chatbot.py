from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from tools import query_knowledge_base_tool, search_for_product_recommendations_tool
from dotenv import load_dotenv
import os

# loading api_key
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# creating prompt template
prompt = """
        You are a customer service chatbot for a flower shop company. You can help the customer achieve the goals listed below.
        <Goals>
            1. Answer the questions the user might have relating to services offered
            2. Recommend products to the user based on their preferences.
        </Goals>

        <Tone>
            Helpful and frienly. Use some gen-z emojis to keep things fun and lighthearted. You MUST always include a funny related pun in every response.
        </Tone>
    """

prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', prompt),
        ('placeholder', "{messages}")
    ]
)

# creating LLM and binding tools
llm = ChatOpenAI(model = "gpt-4o-mini", openai_api_key = api_key)
tools = [query_knowledge_base_tool, search_for_product_recommendations_tool]
llm_with_prompt = ChatPromptTemplate | llm.bind_tools(tools=tools)

# creating state graph
def call_agent(message_state: MessagesState):
    response = llm_with_prompt.invoke(message_state)
    return {'messages': response}


def is_there_tool_calls(message_state:MessagesState):
    last_message = message_state['messages'][-1]
    if last_message.tool_calls:
        return 'use tools'
    else:
        return 'END'

### Note : 2 nodes (agent, tool_node) , 1 conditional edge between agent and tool
graph_builder = StateGraph(MessagesState)

graph_builder.add_node('agent', call_agent)
graph_builder.add_node('tool_node', ToolNode(tools))

graph_builder.add_conditional_edges('agent', 
                                    is_there_tool_calls, 
                                    {'use tools':'tool_node', 'END':END}
                                    )
graph_builder.add_edge('tool_node', 'agent')
graph_builder.add_edge(START, 'agent')

# creating app
app = graph_builder.compile()


