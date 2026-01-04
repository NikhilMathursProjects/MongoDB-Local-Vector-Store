import { useState } from "react";
import { connectDB, fetchNData, queryDB } from "./api";
import "./App.css";

function Apple() {
  const [status, setStatus] = useState("");
  const [data, setData] = useState([]);
  const [n, setN] = useState(5);
  const [query, setQuery] = useState("");

  const [mongoConfig, setMongoConfig] = useState({
    mongo_uri: "mongodb://localhost:27017",
    database_name: "",
    collection_name: "",
  });

  const handleConnect = async () => {
    const res = await connectDB(mongoConfig);
    setStatus(JSON.stringify(res, null, 2));
  };

  const handleFetch = async () => {
    const res = await fetchNData(n);
    setData(res.data || []);
  };

  const handleQuery = async () => {
    const res = await queryDB(query);
    setData(res.data || []);
  };

  return (
    <div className="container">
      <h1>MongoDB Vector Store Demo</h1>

      {/* Connection Section */}
      <div className="card">
        <h2>Connect to MongoDB</h2>
        <input
          placeholder="Mongo URI"
          value={mongoConfig.mongo_uri}
          onChange={(e) =>
            setMongoConfig({ ...mongoConfig, mongo_uri: e.target.value })
          }
        />
        <input
          placeholder="Database Name"
          onChange={(e) =>
            setMongoConfig({ ...mongoConfig, database_name: e.target.value })
          }
        />
        <input
          placeholder="Collection Name"
          onChange={(e) =>
            setMongoConfig({ ...mongoConfig, collection_name: e.target.value })
          }
        />
        <button onClick={handleConnect}>Connect</button>
        <pre>{status}</pre>
      </div>

      {/* Fetch Section */}
      <div className="card">
        <h2>Fetch N Documents</h2>
        <input
          type="number"
          value={n}
          onChange={(e) => setN(Number(e.target.value))}
        />
        <button onClick={handleFetch}>Fetch</button>
      </div>

      {/* Query Section */}
      <div className="card">
        <h2>Query MongoDB</h2>
        <input
          placeholder='Example: {"category":"news"}'
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button onClick={handleQuery}>Run Query</button>
      </div>

      {/* Data Viewer */}
      <div className="card">
        <h2>Data Viewer</h2>
        <pre>{JSON.stringify(data, null, 2)}</pre>
      </div>
    </div>
  );
}

export default Apple;
