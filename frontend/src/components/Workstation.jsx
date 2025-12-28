import React, { useState, useEffect } from 'react';
import { Search, FileText, Settings, Database, Filter, Plus, Download, Edit3, Trash2, List, Code, MoreHorizontal, Activity } from 'lucide-react';
import BenchmarksTab from './BenchmarksTab';

const Workstation = ({ connection, database, collection }) => {
    const [activeTab, setActiveTab] = useState('documents'); // 'documents', 'indexes', 'search'
    const [viewMode, setViewMode] = useState('json'); // 'json', 'table'
    const [documents, setDocuments] = useState([]);
    const [loading, setLoading] = useState(false);

    // Mock Data State
    useEffect(() => {
        if (connection && connection.startsWith('mock://') && collection) {
            // Generate realistic mock data based on collection name
            const mockDocs = Array.from({ length: 12 }).map((_, i) => {
                const isUser = i % 2 === 0;
                return {
                    _id: `ObjectId("${Math.random().toString(16).slice(2, 26)}")`,
                    chat_id: "09d1ad82-5127-4f61-a041-7358a04e9f59",
                    user_id: "1234",
                    sender: isUser ? "user" : "bot",
                    message: isUser ?
                        "Can you explain how I can use FAISS to build a local MongoDB Vector Store?" :
                        "Certainly! To build a local vector store, you'll need...",
                    timestamp: new Date(Date.now() - i * 1000 * 60).toISOString(),
                    vector_embedding: Array.from({ length: 34 }, () => __Math_random_fixed(4)) // truncated for view
                };
            });
            setDocuments(mockDocs);
        } else {
            setDocuments([]);
            // In real mode we'd fetch actual docs here
        }
    }, [collection, connection]);

    if (!database || !collection) {
        return (
            <div className="flex-1 flex flex-col items-center justify-center p-8 text-gray-400 bg-white">
                <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mb-4">
                    <Database size={32} className="text-gray-400" />
                </div>
                <h2 className="text-xl font-medium text-gray-600">No Collection Selected</h2>
                <p className="max-w-md text-center mt-2 text-sm">
                    Select a collection from the sidebar to view documents, manage indexes, and perform vector searches.
                </p>
            </div>
        );
    }

    return (
        <div className="flex-1 flex flex-col bg-white h-screen font-sans">
            {/* Header Path */}
            <div className="px-6 py-3 border-b border-gray-200 flex items-center text-sm text-green-700 bg-white">
                <span className="font-medium hover:underline cursor-pointer">Vector</span>
                <span className="mx-2 text-gray-400">›</span>
                <span className="font-medium hover:underline cursor-pointer">{database}</span>
                <span className="mx-2 text-gray-400">›</span>
                <span className="font-bold flex items-center gap-2">
                    <FileText size={16} />
                    {collection}
                </span>
            </div>

            {/* Main Tabs */}
            <div className="px-6 pt-4 border-b border-gray-200 flex gap-6 bg-white">
                {['Documents', 'Indexes', 'Vector Search', 'Benchmarks'].map(tab => (
                    <button
                        key={tab}
                        onClick={() => setActiveTab(tab.toLowerCase().replace(' ', '_'))}
                        className={`pb-3 px-1 text-sm font-bold border-b-2 transition-colors flex items-center gap-2
                            ${activeTab === tab.toLowerCase().replace(' ', '_')
                                ? 'border-green-600 text-green-700'
                                : 'border-transparent text-gray-500 hover:text-gray-700'}`}
                    >
                        {tab}
                        {tab === 'Documents' && <span className="bg-gray-100 text-gray-600 px-1.5 py-0.5 rounded-full text-xs font-normal">{documents.length}</span>}
                    </button>
                ))}
            </div>

            {/* Toolbar (Only for Documents) */}
            {activeTab === 'documents' && (
                <div className="px-6 py-3 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <div className="relative">
                            <input
                                type="text"
                                placeholder='{ field: "value" }'
                                className="pl-8 pr-4 py-1.5 w-64 text-sm border border-gray-300 rounded focus:ring-1 focus:ring-green-500 outline-none font-mono"
                            />
                            <Search size={14} className="absolute left-2.5 top-2.5 text-gray-400" />
                        </div>
                        <button className="px-3 py-1.5 bg-green-600 text-white text-sm font-bold rounded hover:bg-green-700 transition-colors">Find</button>
                        <button className="px-3 py-1.5 bg-white border border-gray-300 text-gray-700 text-sm font-medium rounded hover:bg-gray-50 transition-colors">Options</button>
                    </div>

                    <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-500 hidden xl:inline">1-{documents.length} of {documents.length}</span>
                        <div className="flex bg-white border border-gray-300 rounded">
                            <button
                                onClick={() => setViewMode('json')}
                                className={`p-1.5 ${viewMode === 'json' ? 'bg-gray-100 text-gray-900' : 'text-gray-500 hover:text-gray-700'}`} title="JSON View">
                                <Code size={16} />
                            </button>
                            <div className="w-px bg-gray-300"></div>
                            <button
                                onClick={() => setViewMode('table')}
                                className={`p-1.5 ${viewMode === 'table' ? 'bg-gray-100 text-gray-900' : 'text-gray-500 hover:text-gray-700'}`} title="Table View">
                                <List size={16} />
                            </button>
                        </div>
                    </div>
                </div>
            )}
            {activeTab === 'documents' && (
                <div className="px-6 py-2 border-b border-gray-200 bg-white flex items-center gap-3">
                    <button className="flex items-center gap-1.5 px-3 py-1.5 bg-green-600 text-white text-xs font-bold rounded shadow-sm hover:bg-green-700">
                        <Plus size={14} /> ADD DATA
                    </button>
                    <button className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-gray-300 text-gray-700 text-xs font-bold rounded shadow-sm hover:bg-gray-50">
                        <Download size={14} /> EXPORT DATA
                    </button>
                    <button className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-gray-300 text-gray-700 text-xs font-bold rounded shadow-sm hover:bg-gray-50 disabled:opacity-50">
                        <Edit3 size={14} /> UPDATE
                    </button>
                    <button className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-gray-300 text-gray-700 text-xs font-bold rounded shadow-sm hover:bg-gray-50 disabled:opacity-50">
                        <Trash2 size={14} /> DELETE
                    </button>
                </div>
            )}

            {/* Content Area */}
            <div className="flex-1 overflow-y-auto bg-white p-6">

                {/* Documents List */}
                {activeTab === 'documents' && (
                    <div className="space-y-4 font-mono text-sm">
                        {documents.map((doc, idx) => (
                            <div key={idx} className="border border-gray-200 rounded hover:border-green-500 hover:shadow-sm transition-all group relative">
                                {/* Hover Actions */}
                                <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 flex gap-1">
                                    <button className="p-1 hover:bg-gray-100 rounded text-gray-500"><Edit3 size={14} /></button>
                                    <button className="p-1 hover:bg-gray-100 rounded text-gray-500"><Download size={14} /></button>
                                    <button className="p-1 hover:bg-red-50 text-red-500 rounded"><Trash2 size={14} /></button>
                                </div>

                                <div className="p-4 bg-white rounded">
                                    {Object.entries(doc).map(([key, value]) => (
                                        <div key={key} className="flex gap-2 leading-relaxed">
                                            <span className="text-gray-500 w-32 flex-shrink-0 text-right select-none">{key} :</span>
                                            <span className={`${typeof value === 'string' ? 'text-green-700' :
                                                typeof value === 'number' ? 'text-blue-600' : 'text-gray-800'
                                                } break-words flex-1`}>
                                                {formatValue(value)}
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>
                )}

                {/* Simulating Indexes Tab from Screenshot */}
                {activeTab === 'indexes' && (
                    <div className="p-4">
                        <h3 className="font-bold text-gray-700 mb-4">Indexes</h3>
                        <table className="w-full text-sm text-left border border-gray-200">
                            <thead className="bg-gray-50 text-gray-600 font-bold border-b border-gray-200">
                                <tr>
                                    <th className="p-3">Name</th>
                                    <th className="p-3">Type</th>
                                    <th className="p-3">Size</th>
                                    <th className="p-3">Usage</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr className="border-b border-gray-100">
                                    <td className="p-3 text-gray-800">_id_</td>
                                    <td className="p-3 text-gray-600">Regular</td>
                                    <td className="p-3 text-gray-600">4.2 KB</td>
                                    <td className="p-3"><div className="w-24 h-2 bg-blue-100 rounded overflow-hidden"><div className="w-1/2 h-full bg-blue-500"></div></div></td>
                                </tr>
                                <tr className="border-b border-gray-100 bg-green-50/30">
                                    <td className="p-3 text-gray-800">vector_index</td>
                                    <td className="p-3 text-green-700 font-bold">FAISS (HNSW)</td>
                                    <td className="p-3 text-gray-600">12.5 MB</td>
                                    <td className="p-3"><div className="w-24 h-2 bg-green-100 rounded overflow-hidden"><div className="w-3/4 h-full bg-green-500"></div></div></td>
                                </tr>
                            </tbody>
                        </table>

                        <div className="mt-8 border p-4 rounded bg-gray-50">
                            <h4 className="font-bold text-gray-700 mb-2">Index Lifecycle</h4>
                            <div className="h-32 bg-white border border-gray-200 rounded flex items-center justify-center text-gray-400 italic">
                                [Performance Graph Placeholder matches screenshot]
                            </div>
                        </div>
                    </div>
                )}

                {/* Vector Search Tab - Reusing Logic */}
                {activeTab === 'vector_search' && (
                    <div className="max-w-4xl mx-auto">
                        <div className="p-8 text-center">
                            <h2 className="text-2xl font-bold text-gray-800 mb-2">Semantic Search</h2>
                            <p className="text-gray-500 mb-6">Search your <span className="font-mono text-green-600 bg-green-50 px-1 rounded">{collection}</span> collection using vector embeddings.</p>

                            <div className="flex gap-2 max-w-xl mx-auto mb-8">
                                <input type="text" className="input-field shadow-sm" placeholder="Try 'database reliability'..." />
                                <button className="btn-primary whitespace-nowrap px-6">Search</button>
                            </div>

                            {/* Mock Results */}
                            {connection.startsWith('mock://') && (
                                <div className="text-left space-y-4">
                                    <div className="bg-white p-4 rounded-lg border border-gray-200 hover:border-green-400 shadow-sm">
                                        <div className="flex justify-between mb-2">
                                            <span className="text-xs font-bold text-green-700 bg-green-50 px-2 py-1 rounded">Score: 0.9821</span>
                                            <span className="text-xs text-gray-400">ID: 550e8400...</span>
                                        </div>
                                        <p className="text-gray-800 text-sm"> MongoDB's replication facilities provide high availability and offer a measure of data redundancy.</p>
                                    </div>
                                    <div className="bg-white p-4 rounded-lg border border-gray-200 hover:border-green-400 shadow-sm">
                                        <div className="flex justify-between mb-2">
                                            <span className="text-xs font-bold text-green-700 bg-green-50 px-2 py-1 rounded">Score: 0.8540</span>
                                            <span className="text-xs text-gray-400">ID: 440e8400...</span>
                                        </div>
                                        <p className="text-gray-800 text-sm">Sharding is a method for distributing data across multiple machines. </p>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* Benchmarks Tab */}
                {activeTab === 'benchmarks' && (
                    <BenchmarksTab connection={connection} />
                )}

            </div>
        </div>
    );
};

// Helper for random formatting
function __Math_random_fixed(len) {
    return Number(Math.random().toFixed(len));
}

function formatValue(value) {
    if (Array.isArray(value)) return `[ ${value.join(', ')} ... ]`;
    if (typeof value === 'string' && value.startsWith('ObjectId')) return <span className="text-gray-600 font-bold">{value}</span>;
    return JSON.stringify(value);
}

export default Workstation;
