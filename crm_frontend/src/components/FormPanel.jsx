import React from 'react';
import { useSelector } from 'react-redux';
import { Search, Calendar, Clock, Mic, Plus } from 'lucide-react';

const FormPanel = () => {
  const formData = useSelector((state) => state.crm.formData);

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
                  className="form-input with-icon"
                  value={formData.hcp_name || ''}
                  readOnly
                  placeholder="Search or select HCP..."
                />
              </div>
            </div>
            <div className="form-group" style={{ flex: 1 }}>
              <label className="form-label">Interaction Type</label>
              <select className="form-input" disabled value={formData.interaction_type || 'Meeting'}>
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
                  className="form-input"
                  value={formData.date || ''}
                  readOnly
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
                  className="form-input"
                  value={formData.time || ''}
                  readOnly
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
              className="form-input"
              value={formData.attendees || ''}
              readOnly
              placeholder="Enter names or search..."
            />
          </div>

          {/* Topics Discussed */}
          <div className="form-group">
            <label className="form-label">Topics Discussed</label>
            <div className="form-input-wrapper">
              <textarea
                className="form-input"
                value={formData.topics_discussed || ''}
                readOnly
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
                className="form-input"
                value={formData.materials_shared || ''}
                readOnly
                placeholder="No materials added."
              />
              <button className="form-input-action-btn">
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
                className="form-input"
                value={formData.samples_distributed || ''}
                readOnly
                placeholder="No samples added."
              />
              <button className="form-input-action-btn">
                <Plus size={14} /> Add Sample
              </button>
            </div>
          </div>

          <div className="form-section-title">Observed/Inferred HCP Sentiment</div>
          
          {/* Sentiment Radio */}
          <div className="form-group">
            <div className="radio-group">
              <label className="radio-option">
                <input type="radio" checked={formData.sentiment === 'Positive'} readOnly disabled /> Positive
              </label>
              <label className="radio-option">
                <input type="radio" checked={formData.sentiment === 'Neutral'} readOnly disabled /> Neutral
              </label>
              <label className="radio-option">
                <input type="radio" checked={formData.sentiment === 'Negative'} readOnly disabled /> Negative
              </label>
            </div>
          </div>

          <div className="form-section-title">Outcomes & Next Steps</div>

          {/* Outcomes */}
          <div className="form-group">
            <label className="form-label">Outcomes</label>
            <textarea
              className="form-input"
              value={formData.outcomes || ''}
              readOnly
              rows={2}
              placeholder="Key outcomes or agreements..."
              style={{ resize: 'none' }}
            />
          </div>

          {/* Follow-up Actions */}
          <div className="form-group">
            <label className="form-label">Follow-up Actions</label>
            <textarea
              className="form-input"
              value={formData.follow_up_actions || ''}
              readOnly
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
