import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Login from './components/Login'; // Importing LoginPage component
import InterviewRoute from './components/InterviewRoute'; // Importing InterviewRoute component
import ProtectedRoute from './components/ProtectedRoute';

import UploadResume from './components/UploadResume';

function App() {
  return (
    <Router>
      <div className="app-container">
        <Routes>
          {/* Default route to render LoginPage */}
          <Route path="/" element={<Login />} />
          
          {/* Route to render InterviewRoute component */}
          <Route path="/interview" element={<InterviewRoute />} /> 
          <Route path="/upload" element={<UploadResume/>}/>
        </Routes>
      </div>
    </Router>
  );
}

export default App;
