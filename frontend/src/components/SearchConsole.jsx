import React, { useState } from 'react';
import { searchDocuments } from '../api';

const SearchConsole = () => {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);

    const handleSearch = async (e) => {
        e.preventDefault();
        if (!query.trim()) return;

        setLoading(true);
        setResults([]);
        try {
            const data = await searchDocuments(query, 5);
            setResults(data);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="card">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Semantic Search</h2>
            <form onSubmit={handleSearch} className="mb-6">
                <div className="relative flex">
                    <input
                        type="text"
                        className="input-field rounded-r-none border-r-0"
                        placeholder="Search for something (e.g. 'fluffy animals')..."
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                    />
                    <button
                        type="submit"
                        disabled={loading}
                        className="btn-primary rounded-l-none"
                    >
                        {loading ? 'Searching...' : 'Search'}
                    </button>
                </div>
            </form>

            <div className="space-y-4">
                {results.length > 0 ? (
                    results.map((item, index) => (
                        <div key={index} className="p-4 bg-gray-50 rounded-lg border border-gray-100 hover:shadow-md transition-shadow">
                            <div className="flex justify-between items-start mb-2">
                                <span className="inline-block px-2 py-1 text-xs font-semibold bg-blue-100 text-blue-800 rounded">
                                    Score: {item.score.toFixed(4)}
                                </span>
                            </div>
                            <p className="text-gray-700 mb-2">{item.content}</p>
                            {Object.keys(item.metadata).length > 0 && (
                                <div className="text-xs text-gray-400">
                                    <pre>{JSON.stringify(item.metadata, null, 2)}</pre>
                                </div>
                            )}
                        </div>
                    ))
                ) : (
                    !loading && query && <p className="text-gray-500 italic text-center">No results found.</p>
                )}
            </div>
        </div>
    );
};

export default SearchConsole;
