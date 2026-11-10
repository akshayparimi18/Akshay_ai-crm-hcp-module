import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import axios from 'axios';

// Async thunk to handle the API call to FastAPI
export const sendChatMessage = createAsyncThunk(
  'crm/sendChatMessage',
  async (text, { rejectWithValue }) => {
    try {
      const response = await axios.post('https://crm-api-backend-4yam.onrender.com/api/chat', {
        text,
        message: text
      });
      return response.data; // { reply: "...", form_data: {...}, extracted_data: {...} }
    } catch (error) {
      return rejectWithValue(error.response?.data?.detail || error.message || 'Could not connect to the CRM backend.');
    }
  }
);

const initialState = {
  chatHistory: [
    { sender: 'ai', text: 'Hello! I am your AI CRM Assistant. How can I help you log your interaction today?' }
  ],
  formData: {
    hcp_name: '',
    interaction_type: 'Meeting',
    date: '',
    time: '',
    attendees: '',
    topics_discussed: '',
    materials_shared: '',
    samples_distributed: '',
    sentiment: 'Positive',
    outcomes: '',
    follow_up_actions: '',
    ai_suggested_follow_ups: []
  },
  isLoading: false,
  error: null,
};

const crmSlice = createSlice({
  name: 'crm',
  initialState,
  reducers: {
    addUserMessage: (state, action) => {
      state.chatHistory.push({ sender: 'user', text: action.payload });
    },
    updateFormField: (state, action) => {
      const { field, value } = action.payload;
      state.formData[field] = value;
    }
  },
  extraReducers: (builder) => {
    builder
      .addCase(sendChatMessage.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(sendChatMessage.fulfilled, (state, action) => {
        state.isLoading = false;
        // 1. Add AI reply to chat history
        if (action.payload.reply) {
          state.chatHistory.push({ sender: 'ai', text: action.payload.reply });
        }
        
        // 2. Synchronize formData from either form_data or extracted_data
        const incomingData = action.payload.form_data || action.payload.extracted_data;
        if (incomingData && typeof incomingData === 'object' && Object.keys(incomingData).length > 0) {
          const formatted = { ...incomingData };
          
          // Format ai_suggested_follow_ups if string
          if (typeof formatted.ai_suggested_follow_ups === 'string') {
            try {
              formatted.ai_suggested_follow_ups = JSON.parse(formatted.ai_suggested_follow_ups);
            } catch (e) {
              formatted.ai_suggested_follow_ups = [formatted.ai_suggested_follow_ups];
            }
          }
          if (!Array.isArray(formatted.ai_suggested_follow_ups)) {
            formatted.ai_suggested_follow_ups = [];
          }
          
          // Format samples_distributed cleanly (e.g., 10 -> "10 samples")
          if (formatted.samples_distributed) {
            const rawSamples = String(formatted.samples_distributed).trim();
            if (/^\d+$/.test(rawSamples)) {
              formatted.samples_distributed = `${rawSamples} samples`;
            }
          }
          
          // Format materials_shared cleanly if array
          if (Array.isArray(formatted.materials_shared)) {
            formatted.materials_shared = formatted.materials_shared.join(', ');
          }

          // Normalize sentiment capitalization
          if (formatted.sentiment) {
            const s = String(formatted.sentiment).toLowerCase();
            if (s === 'positive') formatted.sentiment = 'Positive';
            else if (s === 'neutral') formatted.sentiment = 'Neutral';
            else if (s === 'negative') formatted.sentiment = 'Negative';
          }

          // Merge each valid field into state.formData
          Object.keys(formatted).forEach((key) => {
            const val = formatted[key];
            if (val !== null && val !== undefined && val !== '') {
              state.formData[key] = val;
            }
          });
        }
      })
      .addCase(sendChatMessage.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload;
        // Push graceful error message to chat
        state.chatHistory.push({ sender: 'ai', text: `Error: ${action.payload}` });
      });
  }
});

export const { addUserMessage, updateFormField } = crmSlice.actions;
export default crmSlice.reducer;
