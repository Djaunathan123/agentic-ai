import "./App.css";
import Upload from "./Upload";
import Chat from "./Chat";

function App() {
  return (
    <div className="app-shell">
      <div className="app-card">
        <header className="app-header">
          <h1>Simple RAG Chatbot</h1>
          <p>Upload a file and ask questions using the chat interface.</p>
        </header>

        <div className="app-body">
          <Upload />
          <Chat />
        </div>
      </div>
    </div>
  );
}

export default App;