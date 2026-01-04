import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Workstation from './components/Workstation';
import ConnectionManager from './components/ConnectionManager';

function App() {
  const [connection, setConnection] = useState(null); // 'mongodb://...' URI CONNECTION
  const [isConnModalOpen, setConnModalOpen] = useState(false);

  const [selectedDb, setSelectedDb] = useState(null);
  const [selectedColl, setSelectedColl] = useState(null);

  const handleSelectCollection = (db, coll) => {
    setSelectedDb(db);
    setSelectedColl(coll);
  };

  const handleConnect = (uri) => {
    setConnection(uri);
    // Reset selection on new connection
    setSelectedDb(null);
    setSelectedColl(null);
  };

  return (
    <div className="flex h-screen bg-white overflow-hidden font-sans text-gray-900">

      {/* Sidebar */}
      <Sidebar
        activeConnection={connection}
        onSelectCollection={handleSelectCollection}
        onOpenConnection={() => setConnModalOpen(true)}
      />

      {/* Main Workstation Area */}
      <Workstation
        connection={connection}
        database={selectedDb}
        collection={selectedColl}
      />

      {/* Modals */}
      <ConnectionManager
        isOpen={isConnModalOpen || !connection} // Force open if no connection
        onClose={() => setConnModalOpen(false)}
        onConnect={handleConnect}
      />

    </div>
  );
}

export default App;
