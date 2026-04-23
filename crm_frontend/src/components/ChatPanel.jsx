import React, { useState, useRef, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Send, Bot } from 'lucide-react';
import { addUserMessage, sendChatMessage } from '../features/crm/crmSlice';

const ChatPanel = () => {
  const [inputText, setInputText] = useState('');
  const dispatch = useDispatch();
  const { chatHistory, isLoading } = useSelector((state) => state.crm);
  const endOfMessagesRef = useRef(null);

  const handleSend = (e) => {
    e.preventDefault();
    if (!inputText.trim() || isLoading) return;

    // 1. Dispatch user message to local state instantly
    dispatch(addUserMessage(inputText));
    
    // 2. Dispatch async thunk to call FastAPI
    dispatch(sendChatMessage(inputText));
    
    setInputText('');
  };

  // Auto-scroll to bottom
  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory, isLoading]);

  return (
    <div className="panel-container">
      <div className="panel-header">
        <Bot size={24} className="text-accent" />
        AI Assistant
      </div>
      
      <div className="panel-content chat-history">
        {chatHistory.map((msg, index) => (
          <div key={index} className={`message-bubble ${msg.sender}`}>
            {msg.text}
          </div>
        ))}
        
        {isLoading && (
          <div className="loading-indicator">
            <Bot size={16} />
            <span>AI is thinking</span>
            <div className="dot-pulse">
              <div className="dot"></div>
              <div className="dot"></div>
              <div className="dot"></div>
            </div>
          </div>
        )}
        <div ref={endOfMessagesRef} />
      </div>

      <form className="chat-input-area" onSubmit={handleSend}>
        <input
          type="text"
          className="chat-input"
          placeholder="e.g., Met Dr. Sharma today, left 20 samples..."
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          disabled={isLoading}
        />
        <button 
          type="submit" 
          className="send-btn"
          disabled={!inputText.trim() || isLoading}
        >
          <Send size={18} />
        </button>
      </form>
    </div>
  );
};

export default ChatPanel;
