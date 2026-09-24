import { useEffect, useState } from "react";
import "./App.css";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [backendStatus, setBackendStatus] = useState("Checking backend...");
  const [uploadStatus, setUploadStatus] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [documentReady, setDocumentReady] = useState(false);
  const [totalPages, setTotalPages] = useState(null);

  // Check backend connection
  useEffect(() => {
    fetch(`${API_BASE_URL}/health`)
      .then((response) => {
        if (!response.ok) {
          throw new Error("Backend health check failed");
        }

        return response.json();
      })
      .then((data) => {
        setBackendStatus(data.message || "Connected");
      })
      .catch(() => {
        setBackendStatus("Backend connection failed");
      });
  }, []);

  // Handle PDF selection
  const handleFileChange = (event) => {
    const file = event.target.files[0];

    setMessage("");
    setError("");
    setUploadStatus("");
    setDocumentReady(false);
    setTotalPages(null);
    setAnswer("");
    setSources([]);

    if (!file) {
      setSelectedFile(null);
      return;
    }

    if (file.type !== "application/pdf") {
      setSelectedFile(null);
      setError("Please select a PDF file.");
      return;
    }

    setSelectedFile(file);
  };

  // Upload PDF to FastAPI backend
  const handleUpload = async () => {
    if (!selectedFile) {
      setError("Please select a PDF first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    setUploading(true);
    setMessage("");
    setError("");
    setUploadStatus("Uploading PDF...");
    setDocumentReady(false);

    try {
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || "PDF upload failed.");
      }

      if (data.status === "success") {
        setUploadStatus(
          `Upload successful • ${data.total_pages} pages extracted`
        );

        setMessage(
          `PDF uploaded successfully. ${data.total_pages} pages extracted.`
        );

        setDocumentReady(true);
        setTotalPages(data.total_pages);
      } else {
        setUploadStatus("");
        setError(data.message || "PDF upload failed.");
        setDocumentReady(false);
      }
    } catch (error) {
      console.error("Upload error:", error);

      setUploadStatus("");
      setError(
        error.message ||
          "PDF upload failed. Make sure the backend is running."
      );
      setDocumentReady(false);
    } finally {
      setUploading(false);
    }
  };

  // Ask question using the RAG backend
  const handleAskQuestion = async () => {
    if (!question.trim()) {
      setError("Please enter a question.");
      setMessage("");
      return;
    }

    setAsking(true);
    setMessage("");
    setError("");
    setAnswer("");
    setSources([]);

    try {
      const response = await fetch(
        `${API_BASE_URL}/ask?question=${encodeURIComponent(question)}`,
        {
          method: "POST",
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || "Failed to generate answer.");
      }

      if (data.status === "success") {
        setAnswer(data.answer);
        setSources(data.sources || []);
        setMessage("Answer generated successfully.");
      } else {
        setAnswer("");
        setSources([]);
        setError(data.message || "Failed to generate answer.");
      }
    } catch (error) {
      console.error("Question error:", error);

      setAnswer("");
      setSources([]);
      setError(error.message || "Failed to connect to the backend.");
    } finally {
      setAsking(false);
    }
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div>
            <h1>Academic RAG QA System</h1>

            <p>
              Context-aware question answering over academic documents
            </p>

            <div className="backend-status">
              Backend: {backendStatus}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-container">

        {/* Upload Section */}
        <section className="card">
          <div className="section-header">
            <h2>Upload Academic Document</h2>

            <p>
              Upload a PDF document to build your academic knowledge base.
            </p>
          </div>

          <label className="upload-area">
            <input
              type="file"
              accept=".pdf,application/pdf"
              onChange={handleFileChange}
              disabled={uploading}
            />

            <div className="upload-content">
              <div className="upload-icon">📄</div>

              <h3>
                {selectedFile
                  ? selectedFile.name
                  : "Choose an academic PDF"}
              </h3>

              <p>
                {selectedFile
                  ? "PDF selected successfully"
                  : "Click here to browse your files"}
              </p>
            </div>
          </label>

          {/* Selected File Information */}
          {selectedFile && (
            <div className="file-info">
              <span>Selected file:</span>
              <strong>{selectedFile.name}</strong>
            </div>
          )}

          {/* Document Ready Status */}
          {documentReady && selectedFile && (
            <div className="document-ready">
              <div className="document-ready-icon">✓</div>

              <div>
                <strong>Document ready</strong>

                <p>
                  {selectedFile.name}
                  {totalPages !== null && ` • ${totalPages} pages`}
                </p>
              </div>
            </div>
          )}

          {/* Upload Button */}
          {selectedFile && (
            <button
              className="ask-button"
              onClick={handleUpload}
              disabled={uploading}
            >
              {uploading ? "Uploading..." : "Upload PDF"}
            </button>
          )}

          {/* Upload Status */}
          {uploadStatus && !error && (
            <p className="upload-status">
              {uploadStatus}
            </p>
          )}

          {/* Success Message */}
          {message && (
            <div className="success-message">
              ✓ {message}
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="error-message">
              ⚠ {error}
            </div>
          )}
        </section>

        {/* Question Section */}
        <section className="card">
          <div className="section-header">
            <h2>Ask a Question</h2>

            <p>
              Ask a natural-language question about your uploaded documents.
            </p>
          </div>

          <div className="question-area">
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Example: What is the purpose of the OSI model?"
              rows="4"
              disabled={asking}
            />

            <button
              className="ask-button"
              onClick={handleAskQuestion}
              disabled={asking}
            >
              {asking ? "Generating..." : "Ask Question"}
            </button>
          </div>
        </section>

        {/* Answer Section */}
        <section className="card">
          <div className="section-header">
            <h2>Answer</h2>

            <p>
              The context-grounded response will appear here.
            </p>
          </div>

          <div className="answer-placeholder">
            {asking ? (
              <div className="loading-answer">
                <div className="loading-spinner"></div>

                <p>Generating a context-grounded answer...</p>
              </div>
            ) : answer ? (
              <div className="answer-content">
                {answer}
              </div>
            ) : (
              <>
                <span>💬</span>

                <p>
                  Upload a document and ask a question to get an answer.
                </p>
              </>
            )}
          </div>
        </section>

        {/* Sources Section */}
        <section className="card">
          <div className="section-header">
            <h2>Sources</h2>

            <p>
              Supporting document passages and page references.
            </p>
          </div>

          <div className="sources-placeholder">
            {sources.length > 0 ? (
              sources.map((source, index) => (
                <div
                  className="source-item"
                  key={`${source.document_name}-${source.chunk_id}-${source.page_number}-${index}`}
                >
                  <div className="source-header">
                    <strong>
                      {source.document_name || "Academic Document"}
                    </strong>

                    <span>
                      Page {source.page_number}
                    </span>
                  </div>

                  <div className="source-meta">
                    Chunk {source.chunk_id}
                  </div>

                  <p>{source.chunk_text}</p>
                </div>
              ))
            ) : (
              <>
                <span>📚</span>

                <p>
                  Retrieved document sources will appear here.
                </p>
              </>
            )}
          </div>
        </section>

      </main>

      {/* Footer */}
      <footer className="footer">
        <p>
          Academic RAG QA System • Minor Project
        </p>
      </footer>
    </div>
  );
}

export default App;