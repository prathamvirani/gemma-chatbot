import streamlit as st
import requests
import json
import chromadb
import datetime

# --- 1. App Configuration ---

# Set the page title and icon
st.set_page_config(page_title="Dina & Dyno Chatbot", page_icon="🤖")

# --- 2. Setup Ollama & ChromaDB ---

# Define constants
OLLAMA_URL = "http://localhost:11434/api/chat"
CHROMA_PATH = "./multi_persona_db" # One folder for all persona DBs

# --- 3. Sidebar for Persona Selection & Controls ---

st.sidebar.title("🤖 Persona Controls")
st.sidebar.markdown("Choose who you want to talk to. Their chat history and memory are separate.")

# Radio button for persona selection
selected_persona = st.sidebar.radio(
    "Choose a persona:",
    ["Dina", "Dyno"],
    key="selected_persona"
)

# Get the lowercase model name (assumes you have 'dina' and 'dyno' in Ollama)
model_name = selected_persona.lower()

# Get the unique, persona-specific collection name for ChromaDB
collection_name = f"{model_name}_chat_history"

# --- 4. ChromaDB Connection (Dynamic) ---

# This function is cached, but it's called *once per collection*
# Streamlit's cache is smart enough to store results for different inputs.
@st.cache_resource
def get_chroma_collection(collection_name):
    """
    Initializes a persistent ChromaDB client for a SPECIFIC collection.
    """
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_or_create_collection(name=collection_name)
        print(f"ChromaDB client initialized for collection: {collection_name}")
        return client, collection
    except Exception as e:
        st.error(f"Failed to initialize ChromaDB: {e}")
        return None, None

# Get the client and the *correct* collection for the selected persona
client, collection = get_chroma_collection(collection_name)

# --- 5. Session State for Chat Histories ---

# We now create a *separate* message list for each persona
if "dina_messages" not in st.session_state:
    st.session_state.dina_messages = []

if "dyno_messages" not in st.session_state:
    st.session_state.dyno_messages = []

# Get the correct message list key (e.g., 'dina_messages' or 'dyno_messages')
current_messages_key = f"{model_name}_messages"

# --- 6. "Clear History" Button ---

if st.sidebar.button(f"Clear {selected_persona}'s Memory"):
    if client and collection:
        try:
            # Delete the collection from ChromaDB
            client.delete_collection(name=collection_name)
            
            # Clear the displayed messages from session state
            st.session_state[current_messages_key] = []
            
            # Re-create the collection so we can keep chatting
            collection = client.get_or_create_collection(name=collection_name)
            
            st.sidebar.success(f"{selected_persona}'s memory cleared!")
            # st.rerun() is needed to refresh the page immediately
            st.rerun() 
        except Exception as e:
            st.sidebar.error(f"Error clearing memory: {e}")
    else:
        st.sidebar.error("ChromaDB not initialized.")


# --- 7. Main Chat Interface ---

st.title(f"Chat with {selected_persona}! 🤖")
st.caption(f"This is a persistent chat. {selected_persona} will remember your conversation.")

# Display past chat messages for the *selected* persona
for message in st.session_state[current_messages_key]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 8. Generator Function for Streaming (No Change Needed) ---

def stream_response(payload):
    """
    Yields tokens from the Ollama API stream and returns the full response.
    """
    full_response = ""
    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                token = data['message']['content']
                full_response += token
                yield token # Yield each token for live streaming
                
                if data.get('done'):
                    break
                    
    except requests.exceptions.ConnectionError:
        yield f"Mamma mia! Could not connect to the server at {OLLAMA_URL}."
    except Exception as e:
        yield f"Mamma mia! An error occurred: {e}"
    
    return full_response

# --- 9. Handle User Input and Run the Chat ---

if prompt := st.chat_input(f"Ask {selected_persona} a question!"):
    
    # 1. Add user's message to the *correct* session state list
    st.session_state[current_messages_key].append({"role": "user", "content": prompt})
    
    # 2. Display user's message in the chat
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- 3. RAG: Retrieve Memories (from the correct collection) ---
    
    api_messages = [] 
    try:
        results = collection.query(query_texts=[prompt], n_results=5)
        
        if results['documents']:
            zipped_results = list(zip(results['documents'][0], results['metadatas'][0]))
            zipped_results.sort(key=lambda x: x[1].get('timestamp', 0))
            for doc, meta in zipped_results:
                api_messages.append({"role": meta['role'], "content": doc})
        
        api_messages.append({"role": "user", "content": prompt})

    except Exception as e:
        st.error(f"Error querying ChromaDB: {e}")
        api_messages = [{"role": "user", "content": prompt}]
        
    # --- 4. RAG: Generate Response (using the correct model) ---

    payload = {
        "model": model_name, # This is now dynamic!
        "messages": api_messages,
        "stream": True 
    }

    # 5. Display the assistant's response
    with st.chat_message("assistant"):
        full_response = st.write_stream(stream_response(payload))

    # --- 6. Save to Session State & ChromaDB ---

    # Add response to the correct session state list (for display)
    st.session_state[current_messages_key].append({"role": "assistant", "content": full_response})
    
    # Save both prompt and response to the correct ChromaDB collection
    try:
        user_ts = datetime.datetime.now().isoformat()
        asst_ts = (datetime.datetime.now() + datetime.timedelta(seconds=1)).isoformat()

        collection.add(
            documents=[prompt, full_response],
            metadatos=[
                {"role": "user", "timestamp": user_ts}, 
                {"role": "assistant", "timestamp": asst_ts}
            ],
            ids=[f"msg_{user_ts}_user", f"msg_{asst_ts}_asst"]
        )
    except Exception as e:
        st.error(f"Error saving message to ChromaDB: {e}")
