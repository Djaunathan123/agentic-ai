import { useState, useRef, useEffect } from "react";
import api from "./api";

const INITIAL_PLACEHOLDER = "Type your question here or anything you'd like...";

function Chat() {
  const [q, setQ] = useState("");
  const [messages, setMessages] = useState([
    { role: "assistant", text: INITIAL_PLACEHOLDER },
  ]);
  const [enableEvaluation, setEnableEvaluation] = useState(false);
  const [loading, setLoading] = useState(false);

  const responseRef = useRef(null);

  useEffect(() => {
    if (responseRef.current) {
      responseRef.current.scrollTop = responseRef.current.scrollHeight;
    }
  }, [messages]);

  const ask = async () => {
    const trimmedQuestion = q.trim();
    if (!trimmedQuestion) return;

    setQ("");
    setLoading(true);

    setMessages((prev) => {
      const filtered = prev.filter((m) => m.text !== INITIAL_PLACEHOLDER);
      return [...filtered, { role: "user", text: trimmedQuestion }];
    });

    try {
      const res = await api.get("/chat", {
        params: { 
          question: trimmedQuestion,
          evaluate: enableEvaluation
        },
      });

      const botMessage = {
        role: "assistant",
        text: res.data.answer || "No answer received.",
        evaluation: res.data.evaluation || null,
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      setMessages((prev) => [...prev, {
        role: "assistant",
        text: "Error: " + (error.response?.data?.detail || error.message),
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey && q.trim()) {
      ask();
    }
  };

  return (
    <div className="chat-panel">
      <h2>Chat</h2>
      
      <div style={{ marginBottom: "10px", display: "flex", alignItems: "center", gap: "10px" }}>
        <label style={{ display: "flex", alignItems: "center", gap: "5px", cursor: "pointer" }}>
          <input
            type="checkbox"
            checked={enableEvaluation}
            onChange={(e) => setEnableEvaluation(e.target.checked)}
            disabled={loading}
          />
          <span>Evaluate responses (RAG Triad)</span>
        </label>
      </div>

      <div className="chat-window">
        <div className="chat-response" ref={responseRef}>
          {messages.map((message, idx) => (
            <div key={idx}>
              <div
                className={`message-row ${message.role === "user" ? "row-right" : "row-left"}`}>
                <div className="message-meta">{message.role === "user" ? "You" : "Assistant"}</div>
                <div className={`chat-bubble ${message.role === "user" ? "user" : "bot"}`}>
                  {message.text}
                </div>
              </div>
              
              {message.evaluation && (
                <div style={{
                  margin: "10px 0 10px 20px",
                  padding: "10px",
                  backgroundColor: "#f5f5f5",
                  borderRadius: "8px",
                  fontSize: "0.9em",
                  borderLeft: "4px solid #2196F3"
                }}>
                  <div style={{ fontWeight: "bold", marginBottom: "8px" }}>RAG Triad Evaluation:</div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px" }}>
                    <div>
                      <span>Context Relevance:</span>
                      <div style={{ 
                        color: message.evaluation.context_relevance >= 0.7 ? "green" : "orange",
                        fontWeight: "bold"
                      }}>
                        {(message.evaluation.context_relevance * 100).toFixed(0)}%
                      </div>
                    </div>
                    <div>
                      <span>Groundedness:</span>
                      <div style={{ 
                        color: message.evaluation.groundedness >= 0.7 ? "green" : "orange",
                        fontWeight: "bold"
                      }}>
                        {(message.evaluation.groundedness * 100).toFixed(0)}%
                      </div>
                    </div>
                    <div>
                      <span>Answer Relevance:</span>
                      <div style={{ 
                        color: message.evaluation.answer_relevance >= 0.7 ? "green" : "orange",
                        fontWeight: "bold"
                      }}>
                        {(message.evaluation.answer_relevance * 100).toFixed(0)}%
                      </div>
                    </div>
                    <div>
                      <span>Average:</span>
                      <div style={{ 
                        color: message.evaluation.average_score >= 0.7 ? "green" : "orange",
                        fontWeight: "bold"
                      }}>
                        {(message.evaluation.average_score * 100).toFixed(0)}%
                      </div>
                    </div>
                  </div>
                  <div style={{ marginTop: "8px", fontSize: "0.85em" }}>
                    <strong>Decision:</strong> {message.evaluation.decision}
                  </div>
                </div>
              )}
            </div>
          ))}
          {loading && (
            <div style={{ textAlign: "center", padding: "20px", color: "#999" }}>
              ⏳ Processing... (evaluation may take a moment)
            </div>
          )}
        </div>

        <div className="chat-input-row">
          <input
            className="chat-input"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask anything from your uploaded file..."
            disabled={loading}
          />
          <button 
            className="chat-button" 
            onClick={ask} 
            disabled={!q.trim() || loading}
          >
            {loading ? "..." : "Ask"}
          </button>
        </div>
      </div>
    </div>
  );
}

export default Chat;