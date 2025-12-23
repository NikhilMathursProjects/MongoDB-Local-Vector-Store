import React from 'react';
import { BarChart2, Activity, Zap, Database } from 'lucide-react';

const BenchmarksTab = ({ connection }) => {
    if (!connection || !connection.startsWith('mock://')) {
        return (
            <div className="flex-1 flex flex-col items-center justify-center p-8 text-gray-400 bg-white min-h-[400px]">
                <Activity size={64} className="mb-4 text-gray-200" />
                <h2 className="text-xl font-medium text-gray-600">Benchmarks Unavailable</h2>
                <p className="max-w-md text-center mt-2 text-sm">
                    Benchmarking is currently available only in Mock Mode for demonstration purposes.
                </p>
            </div>
        );
    }

    const metrics = [
        { type: 'FAISS (HNSW)', latency: 5, qps: 1200, recall: 98, color: 'bg-green-500' },
        { type: 'Lucene (HNSW)', latency: 12, qps: 850, recall: 99, color: 'bg-blue-500' },
        { type: 'Mongo Native', latency: 45, qps: 300, recall: 100, color: 'bg-gray-500' }
    ];

    return (
        <div className="max-w-6xl mx-auto w-full p-6">
            <div className="mb-8">
                <h2 className="text-xl font-bold text-gray-800 mb-2">Vector Search Comparison</h2>
                <p className="text-gray-500">Comparing latency, throughput, and recall across different indexing engines.</p>
            </div>

            {/* Scorecards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
                <div className="p-6 rounded-xl border border-gray-200 shadow-sm bg-gradient-to-br from-white to-green-50">
                    <div className="flex items-center gap-3 mb-2 text-green-700">
                        <Zap size={20} />
                        <h3 className="font-bold">Lowest Latency</h3>
                    </div>
                    <div className="text-3xl font-bold text-gray-800">5 ms</div>
                    <div className="text-sm text-gray-500 mt-1">FAISS (HNSW)</div>
                </div>
                <div className="p-6 rounded-xl border border-gray-200 shadow-sm bg-gradient-to-br from-white to-blue-50">
                    <div className="flex items-center gap-3 mb-2 text-blue-700">
                        <BarChart2 size={20} />
                        <h3 className="font-bold">Highest QPS</h3>
                    </div>
                    <div className="text-3xl font-bold text-gray-800">1,200</div>
                    <div className="text-sm text-gray-500 mt-1">FAISS (HNSW)</div>
                </div>
                <div className="p-6 rounded-xl border border-gray-200 shadow-sm bg-gradient-to-br from-white to-gray-50">
                    <div className="flex items-center gap-3 mb-2 text-gray-700">
                        <Database size={20} />
                        <h3 className="font-bold">Best Recall</h3>
                    </div>
                    <div className="text-3xl font-bold text-gray-800">100%</div>
                    <div className="text-sm text-gray-500 mt-1">Mongo Native</div>
                </div>
            </div>

            {/* Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">

                {/* Latency Chart */}
                <div className="p-6 border border-gray-200 rounded-xl bg-white shadow-sm">
                    <h3 className="font-bold text-gray-700 mb-6 flex items-center gap-2">
                        <Zap size={18} className="text-yellow-500" /> Search Latency (lower is better)
                    </h3>
                    <div className="space-y-6">
                        {metrics.map(m => (
                            <div key={m.type}>
                                <div className="flex justify-between text-sm mb-1">
                                    <span className="font-medium text-gray-700">{m.type}</span>
                                    <span className="font-mono text-gray-500">{m.latency} ms</span>
                                </div>
                                <div className="h-3 w-full bg-gray-100 rounded-full overflow-hidden">
                                    <div
                                        className={`h-full ${m.color} rounded-full`}
                                        style={{ width: `${(m.latency / 50) * 100}%` }}
                                    ></div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* QPS Chart */}
                <div className="p-6 border border-gray-200 rounded-xl bg-white shadow-sm">
                    <h3 className="font-bold text-gray-700 mb-6 flex items-center gap-2">
                        <Activity size={18} className="text-blue-500" /> Throughput (QPS) (higher is better)
                    </h3>
                    <div className="space-y-6">
                        {metrics.map(m => (
                            <div key={m.type}>
                                <div className="flex justify-between text-sm mb-1">
                                    <span className="font-medium text-gray-700">{m.type}</span>
                                    <span className="font-mono text-gray-500">{m.qps}</span>
                                </div>
                                <div className="h-3 w-full bg-gray-100 rounded-full overflow-hidden">
                                    <div
                                        className={`h-full ${m.color} rounded-full`}
                                        style={{ width: `${(m.qps / 1500) * 100}%` }}
                                    ></div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Detailed Table */}
            <div className="border border-gray-200 rounded-xl overflow-hidden shadow-sm">
                <table className="w-full text-sm text-left">
                    <thead className="bg-gray-50 text-gray-600 font-bold border-b border-gray-200">
                        <tr>
                            <th className="p-4">Engine</th>
                            <th className="p-4">Index Type</th>
                            <th className="p-4">Latency (P99)</th>
                            <th className="p-4">Throughput</th>
                            <th className="p-4">Recall @ 10</th>
                            <th className="p-4">Index Size</th>
                        </tr>
                    </thead>
                    <tbody className="bg-white">
                        <tr className="border-b border-gray-100">
                            <td className="p-4 font-bold text-green-700">FAISS</td>
                            <td className="p-4 text-gray-600">HNSW</td>
                            <td className="p-4 font-mono">5 ms</td>
                            <td className="p-4 font-mono">1,200 QPS</td>
                            <td className="p-4 font-mono">0.98</td>
                            <td className="p-4 text-gray-500">125 MB</td>
                        </tr>
                        <tr className="border-b border-gray-100">
                            <td className="p-4 font-bold text-blue-700">Lucene</td>
                            <td className="p-4 text-gray-600">HNSW (Graph)</td>
                            <td className="p-4 font-mono">12 ms</td>
                            <td className="p-4 font-mono">850 QPS</td>
                            <td className="p-4 font-mono">0.99</td>
                            <td className="p-4 text-gray-500">145 MB</td>
                        </tr>
                        <tr>
                            <td className="p-4 font-bold text-gray-700">Mongo Native</td>
                            <td className="p-4 text-gray-600">IVF (Clustered)</td>
                            <td className="p-4 font-mono">45 ms</td>
                            <td className="p-4 font-mono">300 QPS</td>
                            <td className="p-4 font-mono">1.00</td>
                            <td className="p-4 text-gray-500">32 MB</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default BenchmarksTab;
