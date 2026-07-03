import { useState } from "react";
import api from "./api";

function Upload() {
  const [file, setFile] = useState();

  const upload = async () => {
    const form = new FormData();
    form.append("file", file);
    await api.post("/upload", form);
    alert("Uploaded!");
  };

  return (
    <div className="upload-panel">
      <h2>Upload File</h2>
      <div className="upload-controls">
        <label className="file-input-label">
          <input type="file" onChange={(e) => setFile(e.target.files[0])} />
          Choose file
        </label>
        <button className="upload-button" onClick={upload} disabled={!file}>
          Upload
        </button>
      </div>
      {file && <p className="upload-file-name">Selected: {file.name}</p>}
    </div>
  );
}

export default Upload;