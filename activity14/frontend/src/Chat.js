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
      // Use the ReAct endpoint — keeps existing /chat endpoint working as-is
      const res = await api.get("/react-chat", {
        params: {
          question: trimmedQuestion,
          evaluate: enableEvaluation,
        },
      });

      const botMessage = {
        role: "assistant",
        text: res.data.answer || "No answer received.",
        transcript: res.data.transcript || [],
        evaluation: res.data.evaluation || null,
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: "Error: " + (error.response?.data?.detail || error.message),
        },
      ]);
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

      <div
        style={{
          marginBottom: "10px",
          display: "flex",
          alignItems: "center",
          gap: "10px",
        }}
      >
        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: "5px",
            cursor: "pointer",
          }}
        >
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
              {/* Message bubble */}
              <div
                className={`message-row ${
                  message.role === "user" ? "row-right" : "row-left"
                }`}
              >
                <div className="message-meta">
                  {message.role === "user" ? "You" : "Assistant"}
                </div>
                <div
                  className={`chat-bubble ${
                    message.role === "user" ? "user" : "bot"
                  }`}
                >
                  {message.text}
                </div>
              </div>

              {/* ReAct transcript (collapsed by default) */}
              {message.transcript && message.transcript.length > 0 && (
                <TranscriptPanel transcript={message.transcript} />
              )}

              {/* RAG Triad scores */}
              {message.evaluation && (
                <div
                  style={{
                    margin: "10px 0 10px 20px",
                    padding: "10px",
                    backgroundColor: "#f5f5f5",
                    borderRadius: "8px",
                    fontSize: "0.9em",
                    borderLeft: "4px solid #2196F3",
                  }}
                >
                  <div style={{ fontWeight: "bold", marginBottom: "8px" }}>
                    RAG Triad Evaluation:
                  </div>
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "1fr 1fr",
                      gap: "8px",
                    }}
                  >
                    <ScoreCell
                      label="Context Relevance"
                      value={message.evaluation.context_relevance}
                    />
                    <ScoreCell
                      label="Groundedness"
                      value={message.evaluation.groundedness}
                    />
                    <ScoreCell
                      label="Answer Relevance"
                      value={message.evaluation.answer_relevance}
                    />
                    <ScoreCell
                      label="Average"
                      value={message.evaluation.average_score}
                    />
                  </div>
                  <div style={{ marginTop: "8px", fontSize: "0.85em" }}>
                    <strong>Decision:</strong> {message.evaluation.decision}
                  </div>
                  {message.evaluation.was_corrected && (
                    <div
                      style={{
                        marginTop: "6px",
                        fontSize: "0.85em",
                        color: "#e65100",
                      }}
                    >
                      🔄 Answer was self-corrected.
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div
              style={{ textAlign: "center", padding: "20px", color: "#999" }}
            >
              ⏳ Processing… (ReAct loop running)
            </div>
          )}
        </div>

        <div className="chat-input-row">
          <input
            className="chat-input"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask anything — e.g. What is ReAct?"
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

// ---------------------------------------------------------------------------
// Sub-components (no style changes to main design)
// ---------------------------------------------------------------------------

function ScoreCell({ label, value }) {
  const color = value >= 0.7 ? "green" : "orange";
  return (
    <div>
      <span>{label}:</span>
      <div style={{ color, fontWeight: "bold" }}>
        {(value * 100).toFixed(0)}%
      </div>
    </div>
  );
}

function TranscriptPanel({ transcript }) {
  const [open, setOpen] = useState(false);

  const phaseEmoji = { USER: "💬", ACTION: "🔧", OBSERVE: "👁", ANSWER: "✅", SYSTEM: "⚙️" };

  return (
    <div
      style={{
        margin: "6px 0 6px 20px",
        fontSize: "0.82em",
        borderLeft: "3px solid #93c5fd",
        paddingLeft: "10px",
      }}
    >
      <button
        onClick={() => setOpen((o) => !o)}
        style={{
          background: "none",
          border: "none",
          cursor: "pointer",
          color: "#2563eb",
          fontWeight: 600,
          padding: 0,
          fontSize: "0.9em",
        }}
      >
        {open ? "▼ Hide ReAct trace" : "▶ Show ReAct trace"}
      </button>

      {open && (
        <div style={{ marginTop: "6px" }}>
          {transcript.map((entry, i) => (
            <div key={i} style={{ marginBottom: "4px", color: "#374151" }}>
              <span style={{ fontWeight: 700, marginRight: "4px" }}>
                {phaseEmoji[entry.phase] || ""} [{entry.phase}]
              </span>
              {entry.content}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default Chat;
