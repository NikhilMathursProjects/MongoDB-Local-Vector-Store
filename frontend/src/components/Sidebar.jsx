import React, { useState, useEffect } from 'react';
import { Database, Folder, ChevronRight, ChevronDown, Plus, Globe } from 'lucide-react';

const Sidebar = ({ activeConnection, onSelectCollection, onOpenConnection }) => {
    const [databases, setDatabases] = useState([]);
    const [expandedDbs, setExpandedDbs] = useState({}); // { dbName: boolean }
    const [collections, setCollections] = useState({}); // { dbName: [colls] }
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (activeConnection) {
            if (activeConnection.startsWith('mock://')) {
                setDatabases(['admin', 'local', 'chatbot_db', 'vector_store']);
            } else {
                fetchDatabases();
            }
        } else {
            setDatabases([]);
        }
    }, [activeConnection]);

    const fetchDatabases = async () => {
        setLoading(true);
        try {
            const res = await fetch('http://localhost:8000/databases', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ uri: activeConnection }),
            });
            const data = await res.json();
            if (data.databases) setDatabases(data.databases);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    const toggleDb = async (dbName) => {
        setExpandedDbs(prev => {
            const isExpanded = !prev[dbName];
            if (isExpanded && !collections[dbName]) {
                if (activeConnection && activeConnection.startsWith('mock://')) {
                    // Mock collections
                    const mockColls = {
                        'admin': ['system.users', 'system.version'],
                        'local': ['startup_log'],
                        'chatbot_db': ['chat_messages', 'chat_sessions', 'config', 'evchatbotdb', 'langchain_test_db', 'local', 'tester'],
                        'vector_store': ['vectors', 'embeddings', 'docs']
                    };
                    setCollections(prev => ({ ...prev, [dbName]: mockColls[dbName] || ['default_collection'] }));
                } else {
                    fetchCollections(dbName);
                }
            }
            return { ...prev, [dbName]: isExpanded };
        });
    };

    const fetchCollections = async (dbName) => {
        try {
            const res = await fetch('http://localhost:8000/collections', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ uri: activeConnection, database: dbName }),
            });
            const data = await res.json();
            if (data.collections) {
                setCollections(prev => ({ ...prev, [dbName]: data.collections }));
            }
        } catch (error) {
            console.error(error);
        }
    };

    return (
        <div className="w-64 bg-gray-50 border-r border-gray-200 h-screen flex flex-col">
            {/* Header / Connections */}
            <div className="p-4 border-b border-gray-200 flex justify-between items-center">
                <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">Connections</span>
                <button
                    onClick={onOpenConnection}
                    className="p-1 hover:bg-gray-200 rounded text-gray-600" title="New Connection"
                >
                    <Plus size={16} />
                </button>
            </div>

            {/* Connection Node */}
            <div className="flex-1 overflow-y-auto p-2">
                {!activeConnection ? (
                    <div className="p-4 text-sm text-gray-500 text-center">
                        No active connection. <br /> Click + to connect.
                    </div>
                ) : (
                    <div>
                        <div className="flex items-center space-x-2 p-2 rounded bg-green-50 text-green-800 mb-2">
                            <Globe size={16} />
                            <span className="text-sm font-medium truncate" title={activeConnection}>
                                {activeConnection.includes('localhost') ? 'Localhost' : 'Remote Cluster'}
                            </span>
                            <span className="w-2 h-2 rounded-full bg-green-500 ml-auto"></span>
                        </div>

                        {loading ? (
                            <div className="p-2 text-xs text-gray-400">Loading databases...</div>
                        ) : (
                            databases.map(db => (
                                <div key={db}>
                                    <div
                                        className="flex items-center space-x-2 p-1.5 hover:bg-gray-200 rounded cursor-pointer text-gray-700 text-sm"
                                        onClick={() => toggleDb(db)}
                                    >
                                        {expandedDbs[db] ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                                        <Database size={14} className="text-gray-500" />
                                        <span className="truncate">{db}</span>
                                    </div>

                                    {/* Collections */}
                                    {expandedDbs[db] && collections[db] && (
                                        <div className="ml-6 border-l border-gray-200 pl-2 mt-1 space-y-0.5">
                                            {collections[db].map(coll => (
                                                <div
                                                    key={coll}
                                                    className="flex items-center space-x-2 p-1.5 hover:bg-blue-50 hover:text-blue-600 rounded cursor-pointer text-gray-600 text-sm"
                                                    onClick={() => onSelectCollection(db, coll)}
                                                >
                                                    <Folder size={14} />
                                                    <span className="truncate">{coll}</span>
                                                </div>
                                            ))}
                                            {collections[db].length === 0 && (
                                                <div className="text-xs text-gray-400 pl-2 py-1">No collections</div>
                                            )}
                                        </div>
                                    )}
                                </div>
                            ))
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default Sidebar;
