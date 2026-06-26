import { useState, useRef, useEffect } from "react";
import api from "./api";

const INITIAL_PLACEHOLDER = "Type your question here or anything you'd like...";

function Chat() {
  const [q, setQ] = useState("");
  const [messages, setMessages] = useState([
    { role: "assistant", text: INITIAL_PLACEHOLDER },
  ]);

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

    setMessages((prev) => {
      const filtered = prev.filter((m) => m.text !== INITIAL_PLACEHOLDER);
      return [...filtered, { role: "user", text: trimmedQuestion }];
    });

    const res = await api.get("/chat", {
      params: { question: trimmedQuestion },
    });

    const botMessage = {
      role: "assistant",
      text: res.data.answer || "No answer received.",
    };

    setMessages((prev) => [...prev, botMessage]);
  };

  return (
    <div className="chat-panel">
      <h2>Chat</h2>

      <div className="chat-window">
        <div className="chat-response" ref={responseRef}>
          {messages.map((message, idx) => (
            <div
              key={idx}
              className={`message-row ${message.role === "user" ? "row-right" : "row-left"}`}>
              <div className="message-meta">{message.role === "user" ? "You" : "Assistant"}</div>
              <div className={`chat-bubble ${message.role === "user" ? "user" : "bot"}`}>
                {message.text}
              </div>
            </div>
          ))}
        </div>

        <div className="chat-input-row">
          <input
            className="chat-input"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Ask anything from your uploaded file..."
          />
          <button className="chat-button" onClick={ask} disabled={!q.trim()}>
            Ask
          </button>
        </div>
      </div>
    </div>
  );
}

export default Chat;