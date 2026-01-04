import React, { useState } from 'react';
import { X } from 'lucide-react';

const ConnectionManager = ({ isOpen, onClose, onConnect }) => {
    const [uri, setUri] = useState('mongodb://localhost:27017');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    if (!isOpen) return null;

    const handleConnect = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');

        try {
            const res = await fetch('http://127.0.0.1:8000/connect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ uri }),
            });
            const data = await res.json();

            if (!res.ok) {
                throw new Error(data.detail || 'Failed to connect');
            }

            onConnect(uri);
            onClose();
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg shadow-xl w-full max-w-md p-6 relative">
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 text-gray-500 hover:text-gray-700"
                >
                    <X size={20} />
                </button>

                <h2 className="text-xl font-bold text-gray-900 mb-4">New Connection</h2>

                <form onSubmit={handleConnect}>
                    <div className="mb-4">
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                            MongoDB URI
                        </label>
                        <input
                            type="text"
                            className="input-field"
                            value={uri}
                            onChange={(e) => setUri(e.target.value)}
                            placeholder="mongodb://localhost:27017"
                        />
                        <p className="text-xs text-gray-500 mt-1">
                            Start with mongodb:// or mongodb+srv://
                        </p>
                    </div>

                    {error && (
                        <div className="mb-4 p-3 bg-red-50 text-red-700 text-sm rounded-lg border border-red-100">
                            {error}
                        </div>
                    )}

                    <div className="flex justify-end space-x-3">
                        <button
                            type="button"
                            onClick={() => {
                                onConnect('mock://localhost');
                                onClose();
                            }}
                            className="text-gray-500 hover:text-blue-600 font-medium text-sm mr-auto"
                        >
                            Mock Connect
                        </button>
                        <button
                            type="button"
                            onClick={onClose}
                            className="btn-secondary"
                        >
                            Cancel
                        </button>
                        <button
                            type="submit"
                            disabled={loading}
                            className={`btn-primary ${loading ? 'opacity-70' : ''}`}
                        >
                            {loading ? 'Connecting...' : 'Connect'}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
};

export default ConnectionManager;
