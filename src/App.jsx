import React, { useState } from 'react';
import axios from 'axios'; // <-- ADD THIS LINE
import './App.css';

function App() {
  // --- STATE MANAGEMENT ---
  // We need state for the file object, the file name, and the result
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('No file chosen');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // --- FUNCTIONS ---
  // This function now saves both the file name and the file object
  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setFileName(selectedFile.name);
      setResult(null); // Clear previous results
      setError(null);  // Clear previous errors
    }
  };

  // This function sends the file to the backend
  const handleUpload = async () => {
    if (!file) {
      setError("Please choose a file first.");
      return;
    }
    setError(null); // Clear any previous errors
    
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("http://127.0.0.1:8000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(res.data);
    } catch (err) {
      setError("An error occurred during analysis. Please try again.");
      console.error("Upload error:", err);
    }
  };

  // --- JSX (HTML) ---
  return (
    <div className="App">
      {/* (Your existing header and nav code can stay here) */}
       <header className="header">
          <nav className="navbar container">
          <a href="#home" className="logo">Knee<span>-Ray</span></a>
          <ul className="nav-links">
          <li><a href="#home">Home</a></li>
          <li><a href="#how-it-works">How It Works</a></li>
          <li><a href="#about">About</a></li>
          </ul>
         </nav>
      </header>

      <main id="home">
        <section className="hero">
          <div className="hero-content container">
            <h1>AI-Powered Knee X-Ray Analysis</h1>
            <h2>(Automated Knee X-Ray Analysis using Deep Learning)</h2>
            <p>Get instant, AI-driven insights from your knee X-ray images. Our advanced algorithms help in identifying potential issues with high accuracy.</p>
            
            <div className="diagnosis-box">
              <h3>Upload Your X-Ray Image</h3>
              
              <label htmlFor="file-input" className="upload-area">
                <svg className="upload-area-icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <p>Click to browse or drag & drop</p>
                <span>PNG, JPG, or DICOM files</span>
              </label>
              
              <input 
                type="file" 
                id="file-input" 
                accept="image/png, image/jpeg, .dcm"
                onChange={handleFileChange} 
              />
              
              <div id="file-name">{fileName}</div>
              
              {/* THIS IS THE KEY FIX: Connect the button to the handleUpload function */}
              <button className="predict-btn" onClick={handleUpload}>
                Analyze Image
              </button>

              {/* --- ADD THIS SECTION TO DISPLAY THE RESULT --- */}
              {error && <div className="result-error">{error}</div>}
              {result && (
                <div className="result-display">
                  <h3>Analysis Result:</h3>
                  <p className={`prediction ${result.prediction.toLowerCase()}`}>
                    Prediction: <strong>{result.prediction}</strong>
                  </p>
                  <p className="confidence">
                    Confidence: <strong>{result.confidence.toFixed(2)}%</strong>
                  </p>
                </div>
              )}
              {/* --- END OF NEW SECTION --- */}

            </div>
          </div>
        </section>
      </main>
      
 {/* (Your existing "How It Works" and Footer sections can stay here) */}
        <section id="how-it-works" className="info-section">
                <div className="container">
           <h2 className="section-title">Simple Steps to Your Analysis</h2>
         <div className="info-grid">
            <div className="info-card">
              <div className="icon">1️⃣</div>
              <h3>Upload Your Image</h3>
              <p>Click the upload area and select the knee X-ray image from your device.</p>
            </div>
            <div className="info-card">
              <div className="icon">2️⃣</div>
              <h3>AI-Powered Analysis</h3>
        _     <p>Our system processes the image using a deep learning model to detect key features.</p>
            </div>
            <div className="info-card">
              <div className="icon">3️⃣</div>
              <h3>Receive Results</h3>
              <p>Get an instant, easy-to-understand report on the potential findings.</p>
            </div>
          </div>
        </div>
      </section>
      
      {/* Footer */}
      <footer className="footer">
        <div className="container">
          <p>&copy; 2024 Knee-Ray. All Rights Reserved. For informational purposes only.</p>
         </div>
      </footer>
    </div>
  );
}

export default App;