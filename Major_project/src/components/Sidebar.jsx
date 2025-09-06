import { useState } from 'react'

const Sidebar = ({ directory, onDirectorySubmit, isProcessing, chatHistory, setCurrentChat }) => {
  const [isCollapsed, setIsCollapsed] = useState(false)
  const [directoryInput, setDirectoryInput] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (directoryInput.trim()) {
      onDirectorySubmit(directoryInput)
    }
  }

  return (
    <aside className={`sidebar ${isCollapsed ? 'collapsed' : ''}`}>
      <div className="sidebar-header">
        <div className="directory-section">
          <h3>File Directory</h3>
          <form onSubmit={handleSubmit} className="directory-form">
            <div className="directory-input-wrapper">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>
              </svg>
              <input
                type="text"
                value={directoryInput}
                onChange={(e) => setDirectoryInput(e.target.value)}
                placeholder="Enter directory path..."
                className="directory-input"
              />
            </div>
            <button 
              type="submit" 
              className="directory-submit-btn"
              disabled={isProcessing || !directoryInput.trim()}
            >
              {isProcessing ? (
                <span className="loading-spinner"></span>
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
              )}
            </button>
          </form>
          {directory && (
            <div className="current-directory">
              <span>Current Directory:</span>
              <code>{directory}</code>
            </div>
          )}
        </div>
        <button 
          className="collapse-btn"
          onClick={() => setIsCollapsed(!isCollapsed)}
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d={isCollapsed ? "M9 18l6-6-6-6" : "M15 18l-6-6 6-6"} />
          </svg>
        </button>
      </div>

      <div className="chat-history">
        <h3>Chat History</h3>
        {chatHistory.map((chat) => (
          <div 
            key={chat.id}
            className="chat-item"
            onClick={() => setCurrentChat(chat)}
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
            </svg>
            <span>{chat.text.substring(0, 30)}...</span>
          </div>
        ))}
      </div>
    </aside>
  )
}

export default Sidebar
