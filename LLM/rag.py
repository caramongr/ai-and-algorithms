import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import openai
import tiktoken
from dotenv import load_dotenv
from langchain.document_loaders import Docx2txtLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# Setup OpenAI API
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-api-key-here')
openai.api_key = os.environ['OPENAI_API_KEY']

@dataclass
class Message:
    role: str
    content: str
    timestamp: str = None
    context: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.context is None:
            self.context = []

class RAGConversationManager:
    def __init__(
        self,
        model_name: str = "gpt-4-turbo-preview",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.conversation_history: List[Message] = []
        self.token_counter = tiktoken.encoding_for_model(model_name)
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        
        # Document processing settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Set default system prompt if none provided
        if system_prompt is None:
            system_prompt = """You are a knowledgeable AI assistant specialized in AI and VR in higher education. 
            Use the provided context from the knowledge base to answer questions accurately and cite specific 
            parts of the source material when relevant. If the context doesn't contain relevant information, 
            say so while providing general knowledge about the topic."""
        
        # Initialize conversation with system prompt
        self.add_message("system", system_prompt)
    
    def process_document(self, file_path: str) -> None:
        """Process a DOCX document and create vector store."""
        try:
            # Load DOCX document
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            
            # Split documents into chunks
            texts = self.text_splitter.split_documents(documents)
            
            # Create or update vector store
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(texts, self.embeddings)
            else:
                self.vector_store.add_documents(texts)
                
            print(f"Processed document: {file_path}")
            print(f"Created {len(texts)} text chunks")
            
        except Exception as e:
            print(f"Error processing document {file_path}: {str(e)}")
    
    def retrieve_context(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant context for a query."""
        if self.vector_store is None:
            return []
        
        # Get relevant documents and scores
        docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Format context with relevance scores
        context = []
        for doc, score in docs_and_scores:
            context.append(f"[Relevance: {score:.2f}] {doc.page_content}")
        
        return context
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.token_counter.encode(text))
    
    def add_message(self, role: str, content: str, context: Optional[List[str]] = None) -> None:
        """Add a message to the conversation history."""
        message = Message(role=role, content=content, context=context)
        self.conversation_history.append(message)
    
    def get_messages_for_context(self, token_limit: int = 3000) -> List[Dict]:
        """Get messages that fit within the token limit."""
        messages = []
        token_count = 0
        
        # Always include system message first
        system_message = self.conversation_history[0]
        messages.append(asdict(system_message))
        token_count += self.count_tokens(system_message.content)
        
        # Add remaining messages in reverse chronological order until token limit
        for message in reversed(self.conversation_history[1:]):
            message_tokens = self.count_tokens(message.content)
            if token_count + message_tokens > token_limit:
                break
            messages.insert(1, asdict(message))
            token_count += message_tokens
        
        return messages
    
    def generate_response(self, user_input: str) -> Tuple[str, List[str]]:
        """Generate a response based on conversation history and retrieved context."""
        try:
            # Retrieve relevant context
            context = self.retrieve_context(user_input)
            
            # Add user input to history
            self.add_message("user", user_input)
            
            # Get messages that fit within context window
            messages = self.get_messages_for_context()
            
            # Prepare prompt with context
            context_text = "\n\n".join(context) if context else "No specific context available."
            prompt = f"""Context from knowledge base:
            {context_text}
            
            User question: {user_input}
            
            Please provide a response based on the context above and previous conversation history. 
            If citing specific information from the context, indicate this clearly."""
            
            # Add prompt to messages
            messages.append({"role": "user", "content": prompt})
            
            # Generate response
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[{
                    'role': msg['role'],
                    'content': msg['content']
                } for msg in messages],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract and store response
            assistant_response = response.choices[0].message.content
            self.add_message("assistant", assistant_response, context=context)
            
            return assistant_response, context
            
        except Exception as e:
            error_message = f"Error generating response: {str(e)}"
            self.add_message("system", error_message)
            return error_message, []
    
    def save_conversation(self, filename: str) -> None:
        """Save the conversation history to a JSON file."""
        with open(filename, 'w') as f:
            json.dump([asdict(msg) for msg in self.conversation_history], f, indent=2)
    
    def load_conversation(self, filename: str) -> None:
        """Load conversation history from a JSON file."""
        with open(filename, 'r') as f:
            messages = json.load(f)
            self.conversation_history = [Message(**msg) for msg in messages]
    
    def save_vector_store(self, path: str) -> None:
        """Save the vector store to disk."""
        if self.vector_store is not None:
            self.vector_store.save_local(path)
            print(f"Vector store saved to {path}")
    
    def load_vector_store(self, path: str) -> None:
        """Load the vector store from disk."""
        if os.path.exists(path):
            self.vector_store = FAISS.load_local(path, self.embeddings)
            print(f"Vector store loaded from {path}")

def chat_session(docx_path: str = None):
    """Run an interactive chat session with document context."""
    conversation = RAGConversationManager()
    
    # Process document if provided
    if docx_path and os.path.exists(docx_path):
        print(f"Processing document: {docx_path}")
        conversation.process_document(docx_path)
    
    print("\nStarting chat session (type 'quit' to exit, 'save' to save, 'load' to load)")
    print("You can also type 'summary' to see conversation statistics")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'save':
            filename = input("Enter filename to save: ")
            conversation.save_conversation(filename)
            conversation.save_vector_store(f"{filename}_vectors")
            print(f"Conversation and vectors saved")
            continue
        elif user_input.lower() == 'load':
            filename = input("Enter filename to load: ")
            try:
                conversation.load_conversation(filename)
                conversation.load_vector_store(f"{filename}_vectors")
                print(f"Conversation and vectors loaded")
            except FileNotFoundError:
                print("File not found!")
            continue
        
        response, context = conversation.generate_response(user_input)
        
        print("\nRelevant Context:")
        for ctx in context:
            print(f"- {ctx}")
        
        print(f"\nAssistant: {response}")

# Example usage
if __name__ == "__main__":
    # Verify API key
    if openai.api_key == 'your-api-key-here':
        print("Please set your OpenAI API key either in the environment or in the code.")
        exit(1)
    
    # Start chat session with document
    chat_session("aivr.docx")