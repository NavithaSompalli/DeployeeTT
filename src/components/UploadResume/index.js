import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './index.css';
 
const UploadResume = () => {
  const navigate = useNavigate();
 
  // State for the textarea inputs
  const [resumeContent, setResumeContent] = useState('');
  const [loading, setLoading] = useState(false); // State to show a loading indicator
 
  // Function to handle form submission
  const handleSubmit = async () => {
    setLoading(true); // Show loading while sending request
 
    try {
      const response = await fetch('http://127.0.0.1:8000/api/upload_resume', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json', // Sending JSON data
        },
        body: JSON.stringify({
          resume_content: resumeContent.trim(), // Send the edited resume content
        }),
      });
 
      if (!response.ok) {
        throw new Error('Failed to upload resume');
      }
 
      // Parse the response from the backend
      const data = await response.json();
      alert("Resume uploaded successfully!");
 
      // Redirect to the interview page after upload with session ID
      navigate('/interview', { state: { session_id: data.session_id } });
    } catch (error) {
      console.error("Error uploading the resume:", error);
      alert("There was an error uploading the resume.");
    } finally {
      setLoading(false); // Stop loading indicator
    }
  };
 
  return (
<div className="upload-resume">
<h2>Upload Resume</h2>
 
      {/* Textarea for resume content */}
<div className="resume-content">
<h3>Content:</h3>
<textarea
          value={resumeContent}
          onChange={(e) => setResumeContent(e.target.value)}
          rows={20}
          className="textarea-content"
          placeholder="Resume content"
        />
</div>
 
      {/* Submit button */}
<button className="submit-button" onClick={handleSubmit} disabled={loading}>
        {loading ? 'Uploading...' : 'Submit Resume'}
</button>
</div>
  );
};
 
export default UploadResume;