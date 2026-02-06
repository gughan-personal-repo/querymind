import React from 'react';
import { Download } from 'lucide-react';

const ResultsTable = ({ data, onClose }) => {
    if (!data) return null;

    const rows = Array.isArray(data.rows) ? data.rows : [];
    const columns = Array.isArray(data.columns)
        ? data.columns
        : (rows[0] ? Object.keys(rows[0]) : []);

    return (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-8 animate-fade-in">
            <div className="bg-secondary border border-white/10 rounded-2xl w-full max-w-6xl max-h-[80vh] flex flex-col shadow-2xl">
                <div className="p-4 border-b border-white/10 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <h3 className="font-semibold text-lg">Query Results</h3>
                        <span className="text-sm text-gray-400">
                            {data.row_count ?? rows.length} rows • {((data.execution_time_ms || 0) / 1000).toFixed(2)}s
                        </span>
                    </div>
                    <div className="flex items-center gap-3">
                        <button className="p-2 hover:bg-white/5 rounded-lg text-gray-400 hover:text-white transition-colors">
                            <Download className="w-5 h-5" />
                        </button>
                        <button
                            onClick={onClose}
                            className="px-3 py-1 hover:bg-white/10 rounded-lg text-sm"
                        >
                            Close
                        </button>
                    </div>
                </div>

                {data.summary && (
                    <div className="px-4 py-3 border-b border-white/10 text-sm text-gray-300 whitespace-pre-line">
                        {data.summary}
                    </div>
                )}

                <div className="flex-1 overflow-auto custom-scrollbar p-4">
                    {rows.length === 0 || columns.length === 0 ? (
                        <div className="text-sm text-gray-300">No results.</div>
                    ) : (
                        <table className="w-full text-left text-sm border-collapse">
                            <thead className="sticky top-0 bg-secondary z-10">
                                <tr>
                                    {columns.map((col) => (
                                        <th key={col} className="px-4 py-3 font-medium text-gray-400 border-b border-white/10 whitespace-nowrap">
                                            {col}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-white/5">
                                {rows.map((row, i) => (
                                    <tr key={i} className="hover:bg-white/5 transition-colors group">
                                        {columns.map((col) => (
                                            <td key={col} className="px-4 py-3 text-gray-300 whitespace-nowrap">
                                                {row?.[col]?.toString?.() ?? String(row?.[col] ?? '')}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    )}
                </div>
            </div>
        </div>
    );
};

export default ResultsTable;
