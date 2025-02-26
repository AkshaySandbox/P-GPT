from typing import TypedDict, Annotated, List, Optional, Dict, Any
import os
from dotenv import load_dotenv
import chainlit as cl
import json
import sys

# Debug mode flag with verbose option
DEBUG = True
VERBOSE = True  # Set to True for even more detailed output

def debug_print(*args, **kwargs):
    if DEBUG:
        print("\033[94m[DEBUG]\033[0m", *args, **kwargs)
        if VERBOSE and len(args) > 0 and isinstance(args[0], str):
            if "response" in args[0].lower() or "result" in args[0].lower():
                print("\033[93m[CONTENT]\033[0m", args[1] if len(args) > 1 else "No content")

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_qdrant import Qdrant as QdrantVectorStore
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings

from pregnancy_kb.config import QDRANT_PATH, COLLECTION_NAME, LLM_MODEL
from pregnancy_kb.prompts import (
    PREGNANCY_ADVISOR_PROMPT, 
    WELCOME_MESSAGE, 
    TAVILY_SEARCH_PROMPT,
    NO_INFO_MESSAGE,
    FOLLOW_UP_PROMPT
)

# Load environment variables
load_dotenv()

# Initialize Qdrant client with local storage
try:
    client = QdrantClient(path=str(QDRANT_PATH))
    debug_print("Qdrant client initialized successfully")
    
    # Check if collection exists
    collections = client.get_collections().collections
    if not any(c.name == COLLECTION_NAME for c in collections):
        debug_print(f"Warning: Collection '{COLLECTION_NAME}' not found")
except Exception as e:
    debug_print(f"Error connecting to Qdrant: {e}")
    client = QdrantClient(":memory:")

# Initialize vector store
try:
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=HuggingFaceEmbeddings(model_name="AkshaySandbox/pregnancy-mpnet-embeddings")
    )
    debug_print("Vector store initialized successfully")
except Exception as e:
    debug_print(f"Error initializing vector store: {e}")
    from langchain_community.vectorstores import FAISS
    vector_store = FAISS(
        embeddings=HuggingFaceEmbeddings(model_name="AkshaySandbox/pregnancy-mpnet-embeddings"), 
        index=None, 
        docstore={}, 
        index_to_docstore_id={}
    )

# Initialize tools
@tool
def canada_pregnancy_search(query: str) -> str:
    """
    Search the internet for pregnancy, childbirth, and parenting information specific to Canada.
    Focuses on official Canadian sources like government websites, health authorities, and Canadian
    medical associations.
    """
    debug_print(f"Canada pregnancy search called with query: {query}")
    try:
        canadian_query = f"{query} Canada official pregnancy childbirth parenting information"
        
        tavily_tool = TavilySearchResults(
            max_results=5,
            k=5,
            search_depth="advanced",
            include_domains=[
                "canada.ca", 
                "healthycanadians.gc.ca",
                "pregnancyinfo.ca",
                "caringforkids.cps.ca",
                "sogc.org",
                "cmaj.ca",
                "phac-aspc.gc.ca"
            ]
        )
        
        debug_print(f"Invoking Tavily search with query: {canadian_query}")
        results = tavily_tool.invoke(canadian_query)
        debug_print(f"Received {len(results)} results from Tavily")
        
        if not results:
            debug_print("No results found from Tavily search")
            return "I couldn't find specific Canadian information on this topic. Please try a different search or consult with your healthcare provider."
        
        # Format the results with metadata
        sources = []
        content_parts = []
        
        for result in results:
            try:
                # Extract content first - if no content, skip this result
                content = result.get('content', '').strip()
                if not content:
                    continue
                
                # Create source metadata with safe defaults
                source_metadata = {
                    "title": result.get('title') or "Canadian Health Resource",  # Safe default if title is missing
                    "section": "Web Search",
                    "category": "Canadian Resources",
                    "url": result.get('url', '')  # Empty string if URL is missing
                }
                
                # Only add sources that have either a title or URL
                if source_metadata["title"] or source_metadata["url"]:
                    sources.append(source_metadata)
                    content_parts.append(content)
                    
            except Exception as e:
                debug_print(f"Error processing individual search result: {e}")
                continue  # Skip this result and continue with others
        
        if not content_parts:
            return """I found some Canadian resources but couldn't extract meaningful information. 
            
Here are some reliable Canadian pregnancy resources you can check directly:
- Health Canada: www.canada.ca/en/public-health/services/pregnancy.html
- Canadian Paediatric Society: www.caringforkids.cps.ca
- Society of Obstetricians and Gynaecologists of Canada: www.pregnancyinfo.ca"""
        
        # Combine and format content
        combined_content = "\n\n".join(content_parts)
        
        # Create a summary prompt for better formatting
        summary_prompt = f"""Summarize the following information about {query} in a clear, organized way:

{combined_content}

Format the response with clear sections and bullet points where appropriate."""
        
        # Get a well-formatted summary
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        debug_print("Invoking LLM with pregnancy advisor prompt")
        summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
        
        formatted_response = format_response_with_metadata(
            content=summary_response.content,
            sources=sources,
            query=query
        )
        
        debug_print(f"Formatted response: \n\n{formatted_response[:200]}...\n\n")
        return formatted_response
        
    except Exception as e:
        debug_print(f"Error in canada_pregnancy_search: {str(e)}")
        return """I apologize, but I'm having trouble accessing Canadian pregnancy information at the moment. 

Here are some reliable Canadian resources you can check directly:
- Health Canada: www.canada.ca/en/public-health/services/pregnancy.html
- Canadian Paediatric Society: www.caringforkids.cps.ca
- Society of Obstetricians and Gynaecologists of Canada: www.pregnancyinfo.ca

You can also consult with your healthcare provider for specific information."""

def format_response_with_metadata(content: str, sources: List[Dict[str, str]], query: str) -> str:
    """Format the response with metadata, sources, and key points."""
    try:
        # Extract key points using bullet points or numbered lists
        key_points = []
        for line in content.split('\n'):
            if line.strip().startswith(('•', '-', '*', '1.', '2.', '3.')):
                key_points.append(line.strip().lstrip('•-* ').strip())
        
        # If no bullet points found, try to extract sentences that look like key points
        if not key_points:
            sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
            key_points = [s for s in sentences if any(kw in s.lower() for kw in ['important', 'should', 'recommend', 'key', 'essential', 'crucial'])][:3]
        
        # Format the response
        formatted_response = "### Response\n\n"
        formatted_response += content
        
        # Add key takeaways section
        formatted_response += "\n\n### Key Takeaways\n"
        if key_points:
            for i, point in enumerate(key_points[:5], 1):  # Limit to top 5 key points
                formatted_response += f"{i}. {point}\n"
        else:
            formatted_response += "• " + content.split('.')[0] + "\n"  # Use first sentence if no key points found
        
        # Add sources section with better formatting
        formatted_response += "\n### Sources\n"
        unique_sources = {}
        for source in sources:
            title = source.get('title', 'Untitled')
            if title not in unique_sources:
                unique_sources[title] = source
        
        for title, source in unique_sources.items():
            formatted_response += f"- **{title}**"
            
            # Add section information if available
            if source.get('section'):
                formatted_response += f"\n  Section: {source['section']}"
            
            # Add category information if available
            if source.get('category'):
                formatted_response += f"\n  Category: {source['category']}"
            
            # Add URL for web sources
            if source.get('url'):
                formatted_response += f"\n  Link: {source['url']}"
            
            # Add document information for knowledge base sources
            if source.get('document_info'):
                formatted_response += f"\n  Document: {source['document_info']}"
            
            formatted_response += "\n\n"  # Add extra line break between sources
        
        return formatted_response
    except Exception as e:
        debug_print(f"Error in format_response_with_metadata: {str(e)}")
        # Return the original content if formatting fails
        return content

@tool
def pregnancy_knowledge_base(query: str, category: Optional[str] = None) -> str:
    """
    Search the pregnancy and early parenthood knowledge base for relevant information.
    
    Args:
        query: The user's question about pregnancy or early parenthood
        category: Optional category to filter results
    """
    debug_print(f"Pregnancy knowledge base search called with query: {query}, category: {category}")
    try:
        # Prepare filter if category is provided
        filter_condition = None
        if category:
            filter_condition = {
                "must": [
                    {
                        "key": "category",
                        "match": {"value": category}
                    }
                ]
            }
        
        # Search with optional filter
        debug_print(f"Searching vector store with query: {query}")
        docs = vector_store.similarity_search(
            query, 
            k=4,
            filter=filter_condition
        )
        debug_print(f"Retrieved {len(docs)} documents from vector store")
        
        if not docs:
            debug_print("No documents found in knowledge base")
            return NO_INFO_MESSAGE
        
        # Prepare context and collect source information
        context_parts = []
        sources = []
        for doc in docs:
            # Create a detailed document info string
            doc_info = []
            if doc.metadata.get("title"):
                doc_info.append(doc.metadata["title"])
            if doc.metadata.get("date"):
                doc_info.append(f"Updated: {doc.metadata['date']}")
            if doc.metadata.get("version"):
                doc_info.append(f"Version: {doc.metadata['version']}")
            
            metadata = {
                'title': doc.metadata.get("title", "Untitled"),
                'section': doc.metadata.get("section", ""),
                'category': doc.metadata.get("category", "general").replace("_", " ").title(),
                'document_info': " | ".join(doc_info) if doc_info else None
            }
            sources.append(metadata)
            context_parts.append(f"[From: {metadata['title']} | Section: {metadata['section']} | Category: {metadata['category']}]\n{doc.page_content}")
        
        context = "\n\n---\n\n".join(context_parts)
        debug_print(f"Prepared context with {len(context_parts)} parts")
        
        # Use the enhanced prompt
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        debug_print("Invoking LLM with pregnancy advisor prompt")
        response = llm.invoke(
            [HumanMessage(content=PREGNANCY_ADVISOR_PROMPT.format(query=query, context=context))]
        )
        
        # Format the response with metadata
        formatted_response = format_response_with_metadata(
            content=response.content,
            sources=sources,
            query=query
        )
        
        debug_print(f"Formatted response: \n\n{formatted_response[:200]}...\n\n")
        return formatted_response
    except Exception as e:
        debug_print(f"Error in pregnancy_knowledge_base: {e}")
        return """I encountered an error while searching the knowledge base. Here are some reliable Canadian resources you can check:

- Health Canada Pregnancy Resources: www.canada.ca/en/public-health/services/pregnancy.html
- Canadian Paediatric Society: www.caringforkids.cps.ca
- Society of Obstetricians and Gynaecologists of Canada: www.pregnancyinfo.ca

Please try rephrasing your question or consult these resources directly."""

# Setup tools
tool_belt = [
    canada_pregnancy_search,
    pregnancy_knowledge_base
]

# Initialize model with tools
model = ChatOpenAI(model=LLM_MODEL, temperature=0)
model = model.bind_tools(tool_belt)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    conversation_history: List[Dict[str, Any]]

# Create graph nodes
tool_node = ToolNode(tool_belt)

def call_model(state):
    messages = state["messages"]
    conversation_history = state.get("conversation_history", [])
    
    debug_print(f"Call model received state with {len(messages)} messages")
    debug_print("Messages content:", [m.content[:100] + "..." for m in messages])
    
    try:
        if conversation_history:
            history_context = "\n\n".join([
                f"User: {item['user']}\nAssistant: {item['assistant'][:150]}..." 
                for item in conversation_history[-3:]
            ])
            
            if messages and isinstance(messages[0], HumanMessage):
                enhanced_message = HumanMessage(
                    content=FOLLOW_UP_PROMPT.format(
                        conversation_history=history_context,
                        user_query=messages[0].content
                    )
                )
                messages = [enhanced_message]
        
        # Add system message to guide the model
        system_message = """You are a knowledgeable pregnancy advisor focusing on Canadian healthcare and parenting information. 
        Use the available tools when needed:
        - canada_pregnancy_search: For finding official Canadian pregnancy and parenting information
        - pregnancy_knowledge_base: For searching the knowledge base about pregnancy and early parenthood
        
        Always provide helpful, accurate information and use the tools when you need to find specific information."""
        
        messages = [HumanMessage(content=system_message)] + messages
        
        debug_print("Invoking model with messages:", [m.content[:100] + "..." for m in messages])
        response = model.invoke(messages)
        debug_print("Raw model response:", response)
        
        # Check for tool calls first
        if response.additional_kwargs.get("tool_calls"):
            debug_print("Tool calls detected in response")
            return {
                "messages": [response],
                "conversation_history": conversation_history
            }
            
        # Handle regular response
        if not response.content:
            debug_print("WARNING: Empty content received from model")
            return {
                "messages": [AIMessage(content="I apologize, but I received an empty response. Please try asking your question again.")],
                "conversation_history": conversation_history
            }
        
        content = response.content
        debug_print("Response content:", content[:200] + "..." if content else "No content")
        
        # Process regular response
        if len(content) > 300 and "##" not in content and "**" not in content:
            lines = content.split("\n")
            formatted_lines = []
            for i, line in enumerate(lines):
                if i > 0 and line.strip() and len(line) < 80 and line.strip()[-1] not in ".,:;?!":
                    formatted_lines.append(f"\n### {line}")
                else:
                    formatted_lines.append(line)
            content = "\n".join(formatted_lines)
        
        if "follow-up" not in content.lower() and "next steps" not in content.lower():
            content += "\n\n### Follow-up Information\n"
            content += "If you have more questions about this topic, feel free to ask! "
            content += "I can provide additional details on specific aspects or related topics that might be helpful for your situation."
        
        return {
            "messages": [AIMessage(content=content)],
            "conversation_history": conversation_history
        }
    except Exception as e:
        debug_print(f"Error in call_model: {str(e)}")
        import traceback
        debug_print("Traceback:", traceback.format_exc())
        return {
            "messages": [AIMessage(content=f"I encountered an error: {str(e)}. Please try again.")],
            "conversation_history": conversation_history
        }

def should_continue(state):
    """Determine if we should continue processing or end the conversation."""
    try:
        last_message = state["messages"][-1]
        debug_print(f"Checking if should continue. Last message: {last_message}")
        
        if hasattr(last_message, "additional_kwargs") and \
           "tool_calls" in last_message.additional_kwargs and \
           last_message.additional_kwargs["tool_calls"]:
            debug_print("Tool calls detected, continuing to action node")
            return "action"
        
        debug_print("No tool calls detected, ending conversation turn")
        return "end"
    except Exception as e:
        debug_print(f"Error in should_continue: {e}")
        # If there's any error in processing, end the conversation
        return "end"

# Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("action", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "action": "action",
        "end": END
    }
)
graph.add_edge("action", "agent")

# Compile graph
compiled_graph = graph.compile()

@cl.on_chat_start
async def start():
    """Initialize the chat session."""
    try:
        await cl.Message(content=WELCOME_MESSAGE).send()
        cl.user_session.set("graph", compiled_graph)
        cl.user_session.set("conversation_history", [])
        debug_print("Chat session initialized")
    except Exception as e:
        debug_print(f"Chat start error: {e}")
        await cl.Message(content=f"Startup error: {str(e)}. Please refresh.").send()

@cl.on_message
async def handle(message: cl.Message):
    """Process user messages."""
    debug_print(f"Received message: {message.content}")
    try:
        graph = cl.user_session.get("graph")
        if not graph:
            debug_print("WARNING: Graph not found in session")
            await cl.Message(content="Session error. Please refresh the page.").send()
            return
            
        conversation_history = cl.user_session.get("conversation_history", [])
        debug_print(f"Current conversation history has {len(conversation_history)} entries")
        
        state = {
            "messages": [HumanMessage(content=message.content)],
            "conversation_history": conversation_history
        }
        
        msg = cl.Message(content="")
        await msg.send()
        
        full_response = ""
        has_tool_calls = False
        
        try:
            debug_print("Starting graph stream")
            async for chunk in graph.astream(state):
                debug_print(f"Received chunk: {chunk}")
                for node, values in chunk.items():
                    debug_print(f"Processing node: {node}")
                    if node == "agent" and values.get("messages"):
                        message = values["messages"][-1]
                        debug_print("Message content:", message.content[:200] + "..." if message.content else "Empty content")
                        debug_print("Message kwargs:", message.additional_kwargs)
                        
                        # Check for tool calls
                        if message.additional_kwargs.get("tool_calls"):
                            has_tool_calls = True
                            debug_print("Tool call detected, waiting for results...")
                            continue
                            
                        response = message.content
                        if response:
                            debug_print("Streaming response chunk:", response[:100] + "...")
                            await msg.stream_token(response)
                            full_response = response
                        else:
                            debug_print("WARNING: Empty response chunk received")
                    elif node == "action":
                        debug_print("Processing action node result")
                        if values.get("messages"):
                            action_response = values["messages"][-1].content
                            if action_response:
                                debug_print("Action response:", action_response[:100] + "...")
                                await msg.stream_token(action_response)
                                full_response = action_response
                            else:
                                debug_print("WARNING: Empty action response")
        
        except Exception as e:
            debug_print(f"Error in graph streaming: {e}")
            import traceback
            debug_print("Traceback:", traceback.format_exc())
            error_message = f"\n\nI encountered an error while processing your request: {str(e)}. Please try again with a different question."
            await msg.stream_token(error_message)
            full_response = error_message
        
        if not full_response and not has_tool_calls:
            debug_print("WARNING: No response generated and no tool calls detected")
            full_response = "I apologize, but I wasn't able to generate a response. Please try asking your question again."
            await msg.stream_token(full_response)
        
        await msg.send()
        
        if full_response:  # Only update history if we have a response
            conversation_history.append({
                "user": message.content,
                "assistant": full_response
            })
            
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]
                
            cl.user_session.set("conversation_history", conversation_history)
            
            # Only generate follow-ups for non-error responses
            if not full_response.startswith("I apologize") and len(conversation_history) >= 2:
                await generate_follow_up_suggestions(conversation_history)
                
    except Exception as e:
        debug_print(f"Error in message handler: {e}")
        await cl.Message(content=f"An error occurred: {str(e)}. Please try again.").send()

async def generate_follow_up_suggestions(conversation_history):
    """Generate and display follow-up suggestions."""
    try:
        follow_up_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        
        history_text = "\n".join([
            f"User: {item['user']}\nAssistant: {item['assistant'][:150]}..." 
            for item in conversation_history[-3:]
        ])
        
        follow_up_prompt = f"""Based on this conversation about pregnancy and parenting, suggest 3 natural follow-up questions the user might want to ask next. Make them specific to the conversation context and helpful for a new parent or expecting parent in Canada.

Conversation:
{history_text}

Provide exactly 3 follow-up questions, each on a new line starting with a bullet point (•).
"""
        
        debug_print("Generating follow-up suggestions")
        follow_up_response = follow_up_llm.invoke([HumanMessage(content=follow_up_prompt)])
        debug_print(f"Follow-up response: {follow_up_response.content}")
        
        suggestions = [
            line.strip().replace('•', '').strip() 
            for line in follow_up_response.content.split('\n') 
            if '•' in line or '-' in line
        ][:3]
        
        if suggestions:
            await cl.Message(content="You might want to ask:").send()
            for suggestion in suggestions:
                await cl.Message(
                    content=suggestion,
                    actions=[
                        cl.Action(
                            name="ask",
                            value=suggestion,
                            label="Ask",
                            description="Ask this follow-up question",
                            payload={"question": suggestion, "type": "follow_up"}
                        )
                    ]
                ).send()
    except Exception as e:
        debug_print(f"Error generating follow-up suggestions: {e}")

@cl.action_callback("ask")
async def on_action(action):
    """Handle follow-up question selection."""
    debug_print(f"Action callback received: {action}")
    
    try:
        # Get the question from either the value or payload
        question = action.value or action.payload.get("question")
        if question:
            await cl.Message(content=question, author="You").send()
            # Process the selected question
            await handle(cl.Message(content=question))
        else:
            debug_print(f"No question found in action: {action}")
            await cl.Message(content="Sorry, I couldn't process that action. Please type your question instead.").send()
    except Exception as e:
        debug_print(f"Error in action callback: {e}")
        await cl.Message(content="Sorry, there was an error processing your selection. Please type your question instead.").send()