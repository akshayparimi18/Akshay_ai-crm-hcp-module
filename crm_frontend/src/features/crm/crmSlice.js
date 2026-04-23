import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import axios from 'axios';

// Async thunk to handle the API call to FastAPI
export const sendChatMessage = createAsyncThunk(
  'crm/sendChatMessage',
  async (text, { rejectWithValue }) => {
    try {
      const response = await axios.post('http://127.0.0.1:8000/api/chat', { text });
      return response.data; // { reply: "...", form_data: {...} }
    } catch (error) {
      return rejectWithValue(error.message || 'Could not connect to the CRM backend.');
    }
  }
);

const initialState = {
  chatHistory: [
    { sender: 'ai', text: 'Hello! I am your AI CRM Assistant. How can I help you log your interaction today?' }
  ],
  formData: {
    hcp_name: '',
    interaction_type: '',
    date: '',
    time: '',
    attendees: '',
    topics_discussed: '',
    materials_shared: '',
    samples_distributed: '',
    sentiment: '',
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
        // 2. Update formData completely if backend returns it
        if (action.payload.form_data && Object.keys(action.payload.form_data).length > 0) {
          state.formData = {
            ...state.formData,
            ...action.payload.form_data
          };
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

export const { addUserMessage } = crmSlice.actions;
export default crmSlice.reducer;
