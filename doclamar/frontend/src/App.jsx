import { useState } from 'react'
import './App.css'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import QueryInput from './components/QueryInput'

function App() {
  const [directory, setDirectory] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [conversations, setConversations] = useState([])
  const [savedChats, setSavedChats] = useState([])
  const [currentChatId, setCurrentChatId] = useState(null)

  const handleDirectorySubmit = (path) => {
    setIsProcessing(true)
    // Here you would integrate with your ML backend to process the directory
    setDirectory(path)
    setIsProcessing(false)
  }

  const autoSaveChat = (conversationsToSave) => {
    try {
      const existingSavedChats = JSON.parse(localStorage.getItem('savedChats') || '[]')
      
      if (currentChatId) {
        // Update existing chat
        const chatIndex = existingSavedChats.findIndex(chat => chat.id === currentChatId)
        if (chatIndex !== -1) {
          existingSavedChats[chatIndex].conversations = conversationsToSave
          existingSavedChats[chatIndex].timestamp = new Date().toISOString()
        }
      } else {
        // Create new chat
        const newChatId = Date.now()
        setCurrentChatId(newChatId)
        
        const chatData = {
          id: newChatId,
          title: `Chat - ${new Date().toLocaleString()}`,
          directory: directory,
          conversations: conversationsToSave,
          timestamp: new Date().toISOString()
        }
        
        existingSavedChats.push(chatData)
      }
      
      localStorage.setItem('savedChats', JSON.stringify(existingSavedChats))
      setSavedChats(existingSavedChats)
    } catch (error) {
      console.error('Error auto-saving chat:', error)
    }
  }

  const handleNewMessage = async (message) => {
    if (!directory) {
      alert('Please select a directory first')
      return
    }

    // 1. Immediately add the user's message to the UI
    const tempId = Date.now();
    const newConversationPair = {
      id: tempId,
      user: {
        id: tempId + 1,
        text: message,
        sender: 'user',
        timestamp: new Date().toISOString()
      },
      assistant: {
        text: "Thinking...", // Temporary loading state
        sender: 'assistant',
        timestamp: new Date().toISOString()
      }
    };
    
    setConversations(prev => [...prev, newConversationPair]);

    try {
      // 2. Call your FastAPI backend
      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: message,
          directory: directory // This is the path from your Sidebar [cite: 91]
        })
      });

      if (!response.ok) throw new Error("Backend failed");
      
      const data = await response.json();

      // 3. Update the conversation with the real AI answer
      setConversations(prev => prev.map(conv => {
        if (conv.id === tempId) {
          return {
            ...conv,
            assistant: {
              text: data.response,
              citations: data.citations, // <-- ADD THIS LINE
              sender: 'assistant',
              timestamp: new Date().toISOString()
            }
          };
        }
        return conv;
      }));

    } catch (error) {
      console.error("Error communicating with AI backend:", error);
      // Handle error state in UI
    }
  }

  const handleLoadChat = (chat) => {
    setConversations(chat.conversations)
    setDirectory(chat.directory)
    setCurrentChatId(chat.id)
  }

  const handleNewChat = () => {
    setConversations([])
    setCurrentChatId(null)
    setDirectory('')
  }

  // Load saved chats on component mount
  const [isInitialized, setIsInitialized] = useState(false)
  if (!isInitialized) {
    try {
      const existingSavedChats = JSON.parse(localStorage.getItem('savedChats') || '[]')
      setSavedChats(existingSavedChats)
      setIsInitialized(true)
    } catch (error) {
      console.error('Error loading saved chats:', error)
      setIsInitialized(true)
    }
  }

  return (
    <div className="app-container">
      <Sidebar 
        directory={directory} 
        onDirectorySubmit={handleDirectorySubmit}
        isProcessing={isProcessing}
        savedChats={savedChats}
        onLoadChat={handleLoadChat}
        onNewChat={handleNewChat}
        currentChatId={currentChatId}
      />
      <main className="main-content">
        <ChatWindow messages={conversations} />
        <QueryInput onSubmit={handleNewMessage} isDisabled={!directory} />
      </main>
    </div>
  )
}

export default App
