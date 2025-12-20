const API_BASE = 'http://localhost:8000';

export const fetchStats = async () => {
    const res = await fetch(`${API_BASE}/stats`);
    if (!res.ok) throw new Error('Failed to fetch stats');
    return res.json();
};

export const addDocuments = async (text, metadata = {}) => {
    const res = await fetch(`${API_BASE}/documents`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts: [text], metadatas: [metadata] }),
    });
    if (!res.ok) throw new Error('Failed to add document');
    return res.json();
};

export const searchDocuments = async (query, k = 5, filter = null) => {
    const res = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, k, filter }),
    });
    if (!res.ok) throw new Error('Failed to search');
    return res.json();
};
