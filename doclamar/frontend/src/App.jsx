import { useState, useEffect } from 'react'
import './App.css'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import QueryInput from './components/QueryInput'

function App() {
  const [directory, setDirectory] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [conversations, setConversations] = useState([])
  const [currentChatId, setCurrentChatId] = useState(null) // This will now store the SQLite session_id

  const [chatMode, setChatMode] = useState('directory'); 
  const [activeFile, setActiveFile] = useState(null);
  const [sessionId, setSessionId] = useState(null);

  // --- 1. LOAD CHAT FROM SQLITE ---
  const handleLoadChat = async (sessionId) => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/history/${sessionId}`);
      const data = await response.json();
      
      // Map SQLite rows to your frontend message format
      const history = data.messages.map((m, index) => ({
        id: index,
        user: m.role === 'user' ? { text: m.content, sender: 'user', timestamp: '' } : null,
        assistant: m.role === 'ai' ? { text: m.content, sender: 'assistant', timestamp: '' } : null
      }));

      // Grouping logic: Your UI expects pairs of {user, assistant}
      // This simple loop groups them into the "conversation" structure your ChatWindow likes
      const groupedConversations = [];
      for (let i = 0; i < history.length; i += 2) {
        groupedConversations.push({
          id: i,
          user: history[i]?.user,
          assistant: history[i+1]?.assistant || { text: "...", sender: 'assistant' }
        });
      }
      
      setConversations(groupedConversations);
      setCurrentChatId(sessionId);
      
      // Determine if it's a folder or file chat based on ID prefix or metadata
      // (For now, we default back to directory mode for history viewing)
      setChatMode('directory'); 
    } catch (err) {
      console.error("Failed to load session content:", err);
    }
  };

  const handleChatWithFile = async (filePath, fileName) => {
    try {
      const response = await fetch("http://127.0.0.1:8000/chat/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_path: filePath })
      });
      
      const data = await response.json();
      
      setSessionId(data.session_id);
      setCurrentChatId(data.session_id); // Sync with sidebar
      setActiveFile(fileName);
      setChatMode('file');
      
      setConversations(prev => [...prev, {
        id: Date.now(),
        user: { text: `Focus on file: ${fileName}`, sender: 'system', timestamp: new Date().toISOString() },
        assistant: { 
          text: `**File Loaded:** I am now answering questions strictly based on the contents of **${fileName}**.`, 
          citations: null, 
          sender: 'assistant', 
          timestamp: new Date().toISOString() 
        }
      }]);
    } catch (error) {
      console.error("Failed to load file:", error);
    }
  };

  const handleReturnToDirectory = () => {
    setChatMode('directory');
    setActiveFile(null);
    setSessionId(null);
    setCurrentChatId(null);
    setConversations(prev => [...prev, {
      id: Date.now(),
      user: { text: `Return to directory search`, sender: 'system', timestamp: new Date().toISOString() },
      assistant: { text: `**Directory Mode:** I am now searching across all files in your selected folder again.`, citations: null, sender: 'assistant', timestamp: new Date().toISOString() }
    }]);
  };

  const handleDirectorySubmit = (path) => {
    setDirectory(path)
  }

  const handleNewChat = () => {
    setConversations([])
    setCurrentChatId(null)
    setSessionId(null)
    // Keep the directory so the user doesn't have to re-select it
  }

  const handleNewMessage = async (message) => {
    if (!directory) {
      alert('Please select a directory first')
      return
    }

    const tempId = Date.now();
    const newConversationPair = {
      id: tempId,
      user: { id: tempId + 1, text: message, sender: 'user', timestamp: new Date().toISOString() },
      assistant: { text: "Thinking...", sender: 'assistant', timestamp: new Date().toISOString() }
    };
    
    setConversations(prev => [...prev, newConversationPair]);

    try {
      let endpoint = "http://127.0.0.1:8000/chat";
      let payload = { query: message, directory: directory };

      if (chatMode === 'file' && sessionId) {
        endpoint = "http://127.0.0.1:8000/chat/message";
        payload = { message: message, session_id: sessionId };
      }

      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!response.ok) throw new Error("Backend failed");
      const data = await response.json();

      // IMPORTANT: After the first message, the backend generates a session_id.
      // We should capture it if we aren't already in a session.
      if (!currentChatId) {
          // If the backend doesn't return the generated ID, 
          // we can infer it or update the backend to return it.
          // For now, let's refresh history after a short delay.
      }

      setConversations(prev => prev.map(conv => {
        if (conv.id === tempId) {
          return {
            ...conv,
            assistant: {
              text: data.response,
              citations: data.citations,
              sender: 'assistant',
              timestamp: new Date().toISOString()
            }
          };
        }
        return conv;
      }));
    } catch (error) {
      console.error("Error communicating with AI backend:", error);
    }
  }

  return (
    <div className="app-container">
      <Sidebar 
        directory={directory} 
        onDirectorySubmit={handleDirectorySubmit}
        isProcessing={isProcessing}
        onLoadChat={handleLoadChat} // Now takes sessionId
        onNewChat={handleNewChat}
        currentChatId={currentChatId}
        chatMode={chatMode}
        activeFile={activeFile}
        onReturnToDirectory={handleReturnToDirectory}
      />
      <main className="main-content">
        <ChatWindow 
          messages={conversations} 
          onChatWithFile={handleChatWithFile} 
        />
        <QueryInput onSubmit={handleNewMessage} isDisabled={!directory} />
      </main>
    </div>
  )
}

export default App