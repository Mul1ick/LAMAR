import { useState } from 'react'
import './App.css'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import QueryInput from './components/QueryInput'

function App() {
  const [directory, setDirectory] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [chatHistory, setChatHistory] = useState([])
  const [currentChat, setCurrentChat] = useState(null)

  const handleDirectorySubmit = (path) => {
    setIsProcessing(true)
    // Here you would integrate with your ML backend to process the directory
    setDirectory(path)
    setIsProcessing(false)
  }

  const handleNewMessage = (message) => {
    if (!directory) {
      alert('Please select a directory first')
      return
    }

    const newMessage = {
      id: Date.now(),
      text: message,
      sender: 'user',
      timestamp: new Date().toISOString()
    }
    
    // Simulating response - replace with actual ML processing
    const response = {
      id: Date.now() + 1,
      text: "This is a sample response. Your ML model will provide the actual response based on the files in the directory.",
      sender: 'assistant',
      timestamp: new Date().toISOString(),
      citations: [
        { file: 'example/path/file1.txt', snippet: 'Relevant text from file 1' },
        { file: 'example/path/file2.txt', snippet: 'Relevant text from file 2' }
      ]
    }

    setChatHistory([...chatHistory, newMessage, response])
  }

  return (
    <div className="app-container">
      <Sidebar 
        directory={directory} 
        onDirectorySubmit={handleDirectorySubmit}
        isProcessing={isProcessing}
        chatHistory={chatHistory}
        setCurrentChat={setCurrentChat}
      />
      <main className="main-content">
        <ChatWindow messages={chatHistory} />
        <QueryInput onSubmit={handleNewMessage} isDisabled={!directory} />
      </main>
    </div>
  )
}

export default App
