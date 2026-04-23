import { configureStore } from '@reduxjs/toolkit';
import crmReducer from './features/crm/crmSlice';

const store = configureStore({
  reducer: {
    crm: crmReducer,
  },
});

export default store;
