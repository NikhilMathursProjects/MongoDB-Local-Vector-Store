import React, { useState } from 'react';
import { addDocuments } from '../api';

const DataManager = ({ onAdd }) => {
    const [text, setText] = useState('');
    const [message, setMessage] = useState('');
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!text.trim()) return;

        setLoading(true);
        setMessage('');

        try {
            await addDocuments(text.trim(), { source: 'dashboard' });
            setMessage('Document added successfully!');
            setText('');
            if (onAdd) onAdd(); // Callback to refresh stats
        } catch (error) {
            setMessage('Error adding document.');
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="card mb-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Add Document</h2>
            <form onSubmit={handleSubmit}>
                <div className="mb-4">
                    <label className="block text-sm font-medium text-gray-700 mb-1">Text Content</label>
                    <textarea
                        className="input-field h-32"
                        placeholder="Enter text to embed..."
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        required
                    />
                </div>
                <div className="flex items-center justify-between">
                    <button
                        type="submit"
                        disabled={loading}
                        className={`btn-primary ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
                    >
                        {loading ? 'Processing...' : 'Add to Index'}
                    </button>
                    {message && (
                        <span className={`text-sm ${message.includes('Error') ? 'text-red-600' : 'text-green-600'}`}>
                            {message}
                        </span>
                    )}
                </div>
            </form>
        </div>
    );
};

export default DataManager;
