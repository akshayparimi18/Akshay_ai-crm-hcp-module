import React from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { Search, Calendar, Clock, Mic, Plus } from 'lucide-react';
import { updateFormField } from '../features/crm/crmSlice';

const FormPanel = () => {
  const dispatch = useDispatch();
  const formData = useSelector((state) => state.crm.formData);

  const handleChange = (e) => {
    const { name, value } = e.target;
    dispatch(updateFormField({ field: name, value }));
  };

  const handleRadioChange = (value) => {
    dispatch(updateFormField({ field: 'sentiment', value }));
  };

  return (
    <div className="panel-container">
      <div className="panel-header" style={{ fontSize: '1rem', padding: '1rem 1.5rem' }}>
        Log HCP Interaction
      </div>
      <div className="panel-content">
        <form onSubmit={(e) => e.preventDefault()}>
          <div className="form-section-title" style={{marginTop: 0}}>Interaction Details</div>
          
          {/* Row 1: HCP Name & Interaction Type */}
          <div className="form-row">
            <div className="form-group" style={{ flex: 2 }}>
              <label className="form-label">HCP Name</label>
              <div className="form-input-wrapper">
                <Search size={16} className="form-input-icon" />
                <input
                  type="text"
                  name="hcp_name"
                  className="form-input with-icon"
                  value={formData.hcp_name || ''}
                  onChange={handleChange}
                  placeholder="Search or select HCP..."
                />
              </div>
            </div>
            <div className="form-group" style={{ flex: 1 }}>
              <label className="form-label">Interaction Type</label>
              <select 
                name="interaction_type"
                className="form-input" 
                value={formData.interaction_type || 'Meeting'}
                onChange={handleChange}
              >
                <option value="Meeting">Meeting</option>
                <option value="Call">Call</option>
                <option value="Email">Email</option>
              </select>
            </div>
          </div>

          {/* Row 2: Date & Time */}
          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Date</label>
              <div className="form-input-wrapper">
                <input
                  type="text"
                  name="date"
                  className="form-input"
                  value={formData.date || ''}
                  onChange={handleChange}
                  placeholder="DD-MM-YYYY"
                />
                <Calendar size={16} style={{ position: 'absolute', right: '0.75rem', color: 'var(--text-muted)' }} />
              </div>
            </div>
            <div className="form-group">
              <label className="form-label">Time</label>
              <div className="form-input-wrapper">
                <input
                  type="text"
                  name="time"
                  className="form-input"
                  value={formData.time || ''}
                  onChange={handleChange}
                  placeholder="HH:MM"
                />
                <Clock size={16} style={{ position: 'absolute', right: '0.75rem', color: 'var(--text-muted)' }} />
              </div>
            </div>
          </div>

          {/* Attendees */}
          <div className="form-group">
            <label className="form-label">Attendees</label>
            <input
              type="text"
              name="attendees"
              className="form-input"
              value={formData.attendees || ''}
              onChange={handleChange}
              placeholder="Enter names or search..."
            />
          </div>

          {/* Topics Discussed */}
          <div className="form-group">
            <label className="form-label">Topics Discussed</label>
            <div className="form-input-wrapper">
              <textarea
                name="topics_discussed"
                className="form-input"
                value={formData.topics_discussed || ''}
                onChange={handleChange}
                rows={3}
                placeholder="Enter key discussion points..."
                style={{ resize: 'none' }}
              />
              <Mic size={16} style={{ position: 'absolute', bottom: '0.75rem', right: '0.75rem', color: 'var(--text-muted)' }} />
            </div>
          </div>

          <div className="form-section-title">Materials Shared / Samples Distributed</div>
          
          {/* Materials Shared */}
          <div className="form-group">
            <label className="form-label">Materials Shared</label>
            <div className="form-input-wrapper">
              <input
                type="text"
                name="materials_shared"
                className="form-input"
                value={formData.materials_shared || ''}
                onChange={handleChange}
                placeholder="No materials added."
              />
              <button className="form-input-action-btn" type="button">
                <Search size={14} /> Search/Add
              </button>
            </div>
          </div>

          {/* Samples Distributed */}
          <div className="form-group">
            <label className="form-label">Samples Distributed</label>
            <div className="form-input-wrapper">
              <input
                type="text"
                name="samples_distributed"
                className="form-input"
                value={formData.samples_distributed || ''}
                onChange={handleChange}
                placeholder="No samples added."
              />
              <button className="form-input-action-btn" type="button">
                <Plus size={14} /> Add Sample
              </button>
            </div>
          </div>

          <div className="form-section-title">Observed/Inferred HCP Sentiment</div>
          
          {/* Sentiment Radio */}
          <div className="form-group">
            <div className="radio-group">
              <label className="radio-option">
                <input 
                  type="radio" 
                  name="sentiment" 
                  checked={formData.sentiment === 'Positive'} 
                  onChange={() => handleRadioChange('Positive')} 
                /> Positive
              </label>
              <label className="radio-option">
                <input 
                  type="radio" 
                  name="sentiment" 
                  checked={formData.sentiment === 'Neutral'} 
                  onChange={() => handleRadioChange('Neutral')} 
                /> Neutral
              </label>
              <label className="radio-option">
                <input 
                  type="radio" 
                  name="sentiment" 
                  checked={formData.sentiment === 'Negative'} 
                  onChange={() => handleRadioChange('Negative')} 
                /> Negative
              </label>
            </div>
          </div>

          <div className="form-section-title">Outcomes & Next Steps</div>

          {/* Outcomes */}
          <div className="form-group">
            <label className="form-label">Outcomes</label>
            <textarea
              name="outcomes"
              className="form-input"
              value={formData.outcomes || ''}
              onChange={handleChange}
              rows={2}
              placeholder="Key outcomes or agreements..."
              style={{ resize: 'none' }}
            />
          </div>

          {/* Follow-up Actions */}
          <div className="form-group">
            <label className="form-label">Follow-up Actions</label>
            <textarea
              name="follow_up_actions"
              className="form-input"
              value={formData.follow_up_actions || ''}
              onChange={handleChange}
              rows={2}
              placeholder="Enter next steps or tasks..."
              style={{ resize: 'none' }}
            />
          </div>

          {/* AI Suggested Follow-ups */}
          {formData.ai_suggested_follow_ups && formData.ai_suggested_follow_ups.length > 0 && (
            <div className="form-group" style={{ marginTop: '1.5rem' }}>
              <label className="form-label" style={{ fontWeight: 600, color: 'var(--text-main)' }}>
                AI Suggested Follow-ups:
              </label>
              <div className="checkbox-list">
                {formData.ai_suggested_follow_ups.map((suggestion, index) => (
                  <label key={index} className="checkbox-item">
                    <input type="checkbox" className="checkbox-input" />
                    <span>{suggestion}</span>
                  </label>
                ))}
              </div>
            </div>
          )}
        </form>
      </div>
    </div>
  );
};

export default FormPanel;

