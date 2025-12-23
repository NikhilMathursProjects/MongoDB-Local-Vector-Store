import React, { useEffect, useState } from 'react';
import { fetchStats } from '../api';

const Dashboard = () => {
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(true);

    const refreshStats = async () => {
        try {
            const data = await fetchStats();
            setStats(data);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        refreshStats();
        // Refresh every 5 seconds or allow manual refresh
        const interval = setInterval(refreshStats, 5000);
        return () => clearInterval(interval);
    }, []);

    if (loading) return <div className="p-4">Loading stats...</div>;
    if (!stats) return <div className="p-4 text-red-500">Error loading stats. Check backend.</div>;

    return (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div className="card border-l-4 border-blue-500">
                <h3 className="text-gray-500 text-sm font-uppercase font-bold tracking-wider">Documents</h3>
                <p className="text-3xl font-bold text-gray-800 mt-1">{stats.document_count}</p>
            </div>
            <div className="card border-l-4 border-green-500">
                <h3 className="text-gray-500 text-sm font-uppercase font-bold tracking-wider">Index Type</h3>
                <p className="text-3xl font-bold text-gray-800 mt-1">{stats.index_type}</p>
            </div>
            <div className="card border-l-4 border-purple-500">
                <h3 className="text-gray-500 text-sm font-uppercase font-bold tracking-wider">Metric</h3>
                <p className="text-3xl font-bold text-gray-800 mt-1">{stats.metric}</p>
            </div>
            <div className="card border-l-4 border-indigo-500">
                <h3 className="text-gray-500 text-sm font-uppercase font-bold tracking-wider">Dimensions</h3>
                <p className="text-3xl font-bold text-gray-800 mt-1">{stats.dimensions}</p>
            </div>
        </div>
    );
};

export default Dashboard;
