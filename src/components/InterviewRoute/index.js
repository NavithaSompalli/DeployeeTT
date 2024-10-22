import React, { useState } from 'react';
import { useNavigate } from "react-router-dom";
import './index.css';

function InterviewRoute() {
  const [content, setContent] = useState(''); // State to store the file content
  const [lines, setLines] = useState([]); // State to store individual lines from the content
  const [selectedLine, setSelectedLine] = useState(''); // State to store the selected line
  const [question, setQuestion] = useState(''); // State to store question or input text
  const [responses, setResponses] = useState([]); // State to store responses
  const [currentIndex, setCurrentIndex] = useState(-1); // State for response navigation
  const [loading, setLoading] = useState(false); // State to show a loading indicator
  const navigate = useNavigate();

  

  // Function to handle question submission and API call
  const handleQuestionSubmit = async () => {
    if (!question) {
      alert("Please enter a question or select a line to submit.");
      return;
    }
    setLoading(true); // Start loading indicator

    try {
      // API call to fetch response from the backend
      const response = await fetch("http://127.0.0.1:8000/api/get_answer?session_id=1234567", {
        method: "POST",
        headers: {
          "Content-Type": "application/json", // Sending JSON data
        },
        body: JSON.stringify({
          prompt_text: question, // Send the current question as a payload
        }),
      });

      // Ensure the response is OK
      if (!response.ok) {
        throw new Error("Failed to fetch the answer");
      }

      // Parse the response data
      const data = await response.json();
      console.log(data)

      // Update responses array with new question and fetched response
      const newResponses = [...responses, { question, response: data.response }];
      setResponses(newResponses);
      setContent(data.response)
      setCurrentIndex(newResponses.length - 1); // Set the current index to the newly added response
      

    } catch (error) {
      console.error("Error submitting the question:", error);
    } finally {
      setLoading(false); // Stop loading indicator
    }
  };

  // Handle selecting a line
  const handleLineSelect = (line) => {
    setSelectedLine(line); // Set the selected line
    setQuestion((prev) => (prev ? `${prev}\n${line}` : line)); // Append the selected line to the question input
  };

  // Navigate to the previous response
  const handlePrevious = () => {
    if (currentIndex > 0) {
      const previousIndex = currentIndex - 1;
      setCurrentIndex(previousIndex);
      setQuestion(responses[previousIndex].question);
      setContent(responses[previousIndex].response);
    }
  };

  // Navigate to the next response
  const handleNext = () => {
    if (currentIndex < responses.length - 1) {
      const nextIndex = currentIndex + 1;
      setCurrentIndex(nextIndex);
      setQuestion(responses[nextIndex].question);
      setContent(responses[nextIndex].response);
    }
  };

  // Handle Enter key press in question input
  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      setQuestion((prev) => prev + '\n'); // Add a new line to the question state
    }
  };

  return (
    <div className="dashboard">
      <header className="header">WELCOME TO MOCK INTERVIEW UI</header>

      {/* Main Content Area */}
      <div className="main-content">
        

        {/* Upload Button */}
        <button className="quick-tips" onClick={() => navigate('/upload')}>
          Upload Resume
        </button>

        {/* Input Element for Scenario Question */}
        <input
          type="text"
          className="input-element"
          placeholder="Enter your scenario questions"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />

        <div className="panel-container">
          {/* Content Area where the uploaded resume content will be displayed */}
          <div className="content-area">
            {content ? content : "Your resume content will appear here..."} {/* Show the content */}
          </div>
        </div>

        {/* Lines List for selection */}
        <div className="lines-list">
          {lines.map((line, index) => (
            <div
              key={index}
              className="line-item"
              onClick={() => handleLineSelect(line)}
            >
              Line {index + 1}: {line}
            </div>
          ))}
        </div>
      </div>

      {/* Footer for Question and Navigation */}
      <footer className="footer">
        <button className="footer-button" onClick={handlePrevious} disabled={currentIndex <= 0}>
          Previous Ans
        </button>

        <textarea
          className="question-input"
          placeholder="Enter your question or selected line will appear here"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyPress={handleKeyPress}
          rows={4}
        />

        <button className="footer-button" onClick={handleQuestionSubmit} disabled={loading}>
          {loading ? 'Loading...' : 'Get Results'}
        </button>

        <button className="footer-button" onClick={handleNext} disabled={currentIndex >= responses.length - 1}>
          Next
        </button>
      </footer>
    </div>
  );
}

export default InterviewRoute;
