import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import axios from 'axios';
import {
    Database,
    Plus,
    Search,
    Share2,
    Settings,
    Bell,
    Paperclip,
    Mic,
    ArrowUp,
    Shield,
    Lock,
    Sparkles,
    Code,
    Play,
    CheckCircle2,
    Loader2,
    ArrowLeft,
    ChevronLeft,
    ChevronRight,
    ChevronDown,
    ChevronUp,
    Table,
    GitBranch,
    Download,
    X,
    Copy,
    Check,
    ArrowUpDown,
    Sigma,
} from 'lucide-react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import ReactFlow, { useReactFlow, ReactFlowProvider, Handle, Position, MarkerType } from 'reactflow';
import 'reactflow/dist/style.css';
import rippleLogo from '../assets/ripple-report/ripple.jpg';
import waveLogo from '../assets/ripple-report/wave.jpg';
import tsunamiLogo from '../assets/ripple-report/tsunami.jpg';

const DEFAULT_PROJECT = 'project-6ab0b570-446d-448e-882';
const BILLING_PROJECT = 'project-6ab0b570-446d-448e-882';
const DEFAULT_DATASET = 'gold_layer';
const USER_NAME = 'Alex';

// Quick action cards for dashboard
const QUICK_ACTIONS = [
    {
        id: 'pii-audit',
        icon: Shield,
        title: 'Analyze PII risk in the marketing dataset',
        action: 'Run audit',
        type: 'yellow',
        query: 'Analyze PII risk in the marketing dataset',
    },
    {
        id: 'email-search',
        icon: Search,
        title: 'Find all tables containing email_address',
        action: 'Search metadata',
        type: 'blue',
        query: 'Search tables for email_address',
    },
    {
        id: 'lineage-viz',
        icon: Share2,
        title: 'Visualize lineage for the revenue_summary table',
        action: 'Generate graph',
        type: 'blue',
        query: 'Show lineage for revenue_summary',
    },
    {
        id: 'rls-audit',
        icon: Lock,
        title: 'Audit row-level security for HR schemas',
        action: 'Check permissions',
        type: 'yellow',
        query: 'Audit row-level security for HR schemas',
    },
];

// Recent analysis items for sidebar - empty on initial load
const RECENT_ANALYSIS = [];

// Time-based greeting
const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good morning';
    if (hour < 18) return 'Good afternoon';
    return 'Good evening';
};

// Action Card Component
const ActionCard = ({ icon: Icon, title, action, type, onClick }) => {
    const baseClasses = 'dashboard-card';
    const typeClasses = type === 'yellow' ? 'bg-card-yellow' : 'bg-card-blue';
    const iconBg = type === 'yellow' ? 'bg-amber-100 text-amber-600' : 'bg-blue-100 text-blue-600';

    return (
        <button className={`${baseClasses} ${typeClasses} text-left w-full`} onClick={onClick}>
            <div className={`w-12 h-12 rounded-2xl flex items-center justify-center ${iconBg}`}>
                <Icon className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold leading-snug text-[#1a1f36] flex-1 mt-2">{title}</h3>
            <div className="flex items-center gap-2 font-semibold text-sm text-[#1a1f36]/50 hover:text-[#1a1f36]/80 transition-colors">
                {action} <ChevronRight className="w-4 h-4" />
            </div>
        </button>
    );
};

// Sidebar Component
const Sidebar = ({ recentAnalysis, onNewAnalysis, activeAnalysis, onSelectAnalysis }) => {
    return (
        <aside className="w-72 sidebar flex flex-col min-h-screen h-screen shadow-2xl z-30">
            {/* Brand Header */}
            <div className="p-8 flex items-center gap-4">
                <div className="w-11 h-11 bg-white/10 rounded-2xl flex items-center justify-center border border-white/10 shadow-lg">
                    <Database className="w-6 h-6 text-yellow-300" />
                </div>
                <div>
                    <h1 className="text-xl font-bold tracking-tight text-white">QueryMind</h1>
                    <p className="text-[10px] tracking-[0.15em] uppercase font-semibold text-white/40 mt-0.5">
                        BigQuery Insights
                    </p>
                </div>
            </div>

            {/* New Analysis Button */}
            <div className="px-6 mb-8">
                <button
                    onClick={onNewAnalysis}
                    className="w-full bg-white text-[#1e296b] hover:bg-white/95 font-bold py-4 px-5 rounded-2xl flex items-center justify-between shadow-xl transition-all active:scale-[0.98] group"
                >
                    <span className="text-base">New Analysis</span>
                    <div className="w-8 h-8 rounded-xl bg-[#1e296b]/5 flex items-center justify-center group-hover:bg-[#1e296b]/10 transition-colors">
                        <Plus className="w-5 h-5 text-[#1e296b]" />
                    </div>
                </button>
            </div>

            {/* Recent Analysis Section */}
            <div className="flex-1 px-4 overflow-y-auto custom-scrollbar">
                <p className="px-4 text-[10px] tracking-[0.2em] uppercase text-white/30 font-bold mb-4">
                    Recent Analysis
                </p>
                <div className="space-y-1">
                    {recentAnalysis.map((item) => (
                        <button
                            key={item.id}
                            onClick={() => onSelectAnalysis(item)}
                            className={`w-full flex items-center gap-3 px-4 py-3.5 rounded-xl transition-all ${activeAnalysis === item.id
                                ? 'bg-white/10 text-white border border-white/10'
                                : 'text-white/50 hover:text-white hover:bg-white/5'
                                }`}
                        >
                            <item.icon className="w-4 h-4 opacity-60" />
                            <span className="font-semibold text-sm tracking-tight truncate">{item.title}</span>
                        </button>
                    ))}
                </div>
            </div>

            {/* User Profile */}
            <div className="p-6">
                <div className="bg-white/5 p-4 rounded-2xl flex items-center justify-between border border-white/5">
                    <div className="flex items-center gap-3">
                        <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-amber-200 to-amber-400 flex items-center justify-center overflow-hidden border-2 border-white/20 shadow-lg">
                            <img
                                src="https://api.dicebear.com/7.x/avataaars/svg?seed=Alex"
                                alt="Avatar"
                                className="w-full h-full"
                            />
                        </div>
                        <div className="flex flex-col">
                            <span className="text-sm font-bold text-white tracking-tight">Alex Sterling</span>
                            <span className="text-[9px] uppercase font-semibold text-white/30 tracking-widest">
                                Enterprise Admin
                            </span>
                        </div>
                    </div>
                    <button className="p-2.5 hover:bg-white/10 rounded-xl transition-all group text-white/30 hover:text-white">
                        <Settings className="w-5 h-5 transition-transform group-hover:rotate-45" />
                    </button>
                </div>
            </div>
        </aside>
    );
};

// Table Result Card Component - Matches screenshot design
const TableResultCard = ({ result, onClick }) => {
    const similarityScore = typeof result.similarity === 'number' ? result.similarity : 0;
    const similarity = `${Math.round(similarityScore * 100)}% match`;
    const datasetName = result.dataset_id || DEFAULT_DATASET;
    const tableName = result.table_name || 'unknown_table';
    const primaryLabel = result.column_name || tableName;
    const secondaryLabel = `${datasetName}.${tableName}`;

    return (
        <button
            onClick={() => onClick(result)}
            className="bg-white rounded-2xl border border-black/10 p-5 hover:border-[#1e296b]/30 hover:shadow-lg transition-all cursor-pointer text-left min-w-[200px] flex-shrink-0"
        >
            <div className="text-base font-bold text-[#1a1f36] mb-1">
                {primaryLabel}
            </div>
            <div className="text-xs text-[#94a3b8] font-mono">
                {secondaryLabel}
            </div>
            <div className="text-xs text-[#64748b] mt-2 font-semibold">
                {similarity}
            </div>
        </button>
    );
};


// Inline Schema Display Component - matches screenshot design
const InlineSchemaDisplay = ({ table, schema, loading, error, onSuggestion }) => {
    if (!table) return null;

    return (
        <div className="mt-4 animate-fade-in">
            {loading ? (
                <div className="flex items-center justify-center py-8">
                    <Loader2 className="w-6 h-6 text-[#1e296b] animate-spin" />
                </div>
            ) : error ? (
                <div className="text-center py-8 text-red-500 text-sm">{error}</div>
            ) : schema ? (
                <div className="bg-white rounded-2xl border border-black/10 overflow-hidden">
                    {/* Schema Header */}
                    <div className="px-6 py-4 flex items-center justify-between border-b border-black/5">
                        <div className="flex items-center gap-3">
                            <Table className="w-5 h-5 text-[#1e296b]" />
                            <span className="text-base font-bold text-[#1a1f36]">
                                Schema: {table.table_name}
                            </span>
                        </div>
                        <span className="text-xs text-[#94a3b8]">
                            Last synced: just now
                        </span>
                    </div>

                    {/* Schema Table - Scrollable */}
                    <div className="px-6 py-4">
                        <div className="max-h-[300px] overflow-y-auto">
                            <table className="w-full">
                                <thead className="sticky top-0 bg-white">
                                    <tr className="text-left">
                                        <th className="pb-3 text-[10px] font-semibold text-[#94a3b8] uppercase tracking-wider">
                                            Column
                                        </th>
                                        <th className="pb-3 text-[10px] font-semibold text-[#94a3b8] uppercase tracking-wider">
                                            Type
                                        </th>
                                        <th className="pb-3 text-[10px] font-semibold text-[#94a3b8] uppercase tracking-wider">
                                            Description
                                        </th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-black/5">
                                    {schema.columns?.map((col, idx) => (
                                        <tr key={idx}>
                                            <td className="py-3 pr-4">
                                                <span className="font-mono font-semibold text-[#1a1f36] text-sm">
                                                    {col.name}
                                                </span>
                                            </td>
                                            <td className="py-3 pr-4">
                                                <span className="text-sm text-[#64748b]">
                                                    {col.data_type}
                                                </span>
                                            </td>
                                            <td className="py-3">
                                                <span className="text-sm text-[#64748b] italic">
                                                    {col.description || '-'}
                                                </span>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                        {schema.columns?.length > 10 && (
                            <div className="text-center text-xs text-[#94a3b8] mt-3 border-t border-black/5 pt-3">
                                {schema.columns.length} columns total
                            </div>
                        )}
                    </div>

                    {/* Suggested Next Steps */}
                    <div className="px-6 py-4 border-t border-black/5 bg-[#fafbfc]">
                        <div className="text-[10px] font-semibold text-[#94a3b8] uppercase tracking-wider mb-3">
                            Suggested Next Steps
                        </div>
                        <div className="flex gap-2 flex-wrap">
                            <button
                                onClick={() => onSuggestion?.(`Show lineage for ${table.dataset_id || DEFAULT_DATASET}.${table.table_name}`)}
                                className="px-4 py-2 text-sm font-medium text-[#1e296b] border border-[#1e296b]/20 rounded-full hover:bg-[#1e296b]/5 transition-colors"
                            >
                                Show Lineage
                            </button>
                        </div>
                    </div>
                </div>
            ) : null}
        </div>
    );
};

const buildLineageSummary = (edges, focusTable) => {
    if (!edges?.length) {
        return { upstream: [], downstream: [] };
    }
    const upstream = new Set();
    const downstream = new Set();
    edges.forEach((edge) => {
        if (focusTable && edge.to_table === focusTable) {
            upstream.add(`${edge.dataset_id}.${edge.from_table}`);
        }
        if (focusTable && edge.from_table === focusTable) {
            downstream.add(`${edge.to_dataset_id}.${edge.to_table}`);
        }
        if (!focusTable) {
            upstream.add(`${edge.dataset_id}.${edge.from_table}`);
            downstream.add(`${edge.to_dataset_id}.${edge.to_table}`);
        }
    });
    return {
        upstream: Array.from(upstream),
        downstream: Array.from(downstream),
    };
};

const LineageSummaryCard = ({ edges, focusTable }) => {
    if (!edges?.length) return null;
    const summary = buildLineageSummary(edges, focusTable);
    const upstream = summary.upstream.slice(0, 4);
    const downstream = summary.downstream.slice(0, 4);

    return (
        <div className="bg-white rounded-2xl border border-black/10 p-5 shadow-soft">
            <div className="flex items-center justify-between mb-3">
                <div>
                    <h3 className="text-base font-bold text-[#1a1f36]">Lineage Summary</h3>
                    <p className="text-xs text-[#94a3b8]">
                        {focusTable ? `${focusTable} dependencies` : 'Dependency overview'}
                    </p>
                </div>
                <span className="text-[10px] font-semibold uppercase tracking-wider text-[#94a3b8]">
                    {summary.upstream.length} upstream • {summary.downstream.length} downstream
                </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="bg-[#f8fafc] rounded-xl border border-black/5 p-3">
                    <div className="text-[10px] font-semibold uppercase tracking-wider text-[#94a3b8] mb-2">
                        Upstream
                    </div>
                    {upstream.length ? (
                        <div className="space-y-1 text-sm text-[#1a1f36]">
                            {upstream.map((item) => (
                                <div key={item} className="font-mono text-xs text-[#1e296b]">
                                    {DEFAULT_PROJECT}.{item}
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="text-xs text-[#94a3b8]">None detected</div>
                    )}
                </div>
                <div className="bg-[#f8fafc] rounded-xl border border-black/5 p-3">
                    <div className="text-[10px] font-semibold uppercase tracking-wider text-[#94a3b8] mb-2">
                        Downstream
                    </div>
                    {downstream.length ? (
                        <div className="space-y-1 text-sm text-[#1a1f36]">
                            {downstream.map((item) => (
                                <div key={item} className="font-mono text-xs text-[#1e296b]">
                                    {DEFAULT_PROJECT}.{item}
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="text-xs text-[#94a3b8]">None detected</div>
                    )}
                </div>
            </div>
        </div>
    );
};

const CHART_POINT_LIMIT = 120;
const CHART_BAR_LIMIT = 12;

const parseNumber = (value) => {
    if (value === null || value === undefined) return null;
    if (typeof value === 'number') return Number.isFinite(value) ? value : null;
    if (typeof value === 'string') {
        const cleaned = value.replace(/[$,%\s,]/g, '');
        if (!cleaned) return null;
        const num = Number(cleaned);
        return Number.isFinite(num) ? num : null;
    }
    return null;
};

const parseDateValue = (value) => {
    if (value instanceof Date && !Number.isNaN(value.getTime())) return value;
    if (typeof value === 'number') {
        if (value > 1e12) return new Date(value);
        if (value > 1e9) return new Date(value * 1000);
        return null;
    }
    if (typeof value === 'string') {
        const trimmed = value.trim();
        const isDigitsOnly = /^[0-9]+$/.test(trimmed);
        if (isDigitsOnly) {
            const numeric = Number(trimmed);
            if (numeric > 1e12) return new Date(numeric);
            if (numeric > 1e9) return new Date(numeric * 1000);
            return null;
        }
        if (!/[-/:tTZ]/.test(trimmed)) return null;
        const parsed = new Date(trimmed);
        return Number.isNaN(parsed.getTime()) ? null : parsed;
    }
    return null;
};

const inferChartSpec = (columns, rows) => {
    if (!columns.length || rows.length < 2) return null;
    const sample = rows.slice(0, 50);
    const sampleCount = sample.length || 1;
    const dateTokens = ['date', 'time', 'timestamp', 'datetime', 'day', 'month', 'year'];

    const stats = columns.map((col) => {
        let numericCount = 0;
        let dateCount = 0;
        const uniques = new Set();
        for (const row of sample) {
            const val = row[col];
            if (val !== null && val !== undefined) {
                uniques.add(String(val));
            }
            if (parseNumber(val) !== null) numericCount += 1;
            if (parseDateValue(val)) dateCount += 1;
        }
        const name = col.toLowerCase();
        const nameScore = dateTokens.reduce((score, token) => (name.includes(token) ? score + 1 : score), 0);
        return {
            col,
            numericRatio: numericCount / sampleCount,
            dateRatio: dateCount / sampleCount,
            uniqueRatio: uniques.size / sampleCount,
            dateNameScore: nameScore,
        };
    });

    const numericCandidates = stats
        .filter((stat) => stat.numericRatio >= 0.7)
        .sort((a, b) => b.numericRatio - a.numericRatio);
    const dateCandidates = stats
        .filter((stat) => stat.dateRatio >= 0.7 || (stat.dateRatio >= 0.5 && stat.dateNameScore > 0))
        .sort((a, b) => b.dateNameScore - a.dateNameScore || b.dateRatio - a.dateRatio);

    if (dateCandidates.length && numericCandidates.length) {
        const dateCol = dateCandidates[0].col;
        const valueCol = numericCandidates.find((stat) => stat.col !== dateCol)?.col;
        if (valueCol) {
            return { type: 'line', labelCol: dateCol, valueCol };
        }
    }

    const categoryCandidates = stats
        .filter((stat) => stat.numericRatio < 0.4 && stat.dateRatio < 0.4)
        .sort((a, b) => a.uniqueRatio - b.uniqueRatio);
    if (numericCandidates.length && categoryCandidates.length) {
        return {
            type: 'bar',
            labelCol: categoryCandidates[0].col,
            valueCol: numericCandidates[0].col,
        };
    }

    return null;
};

const buildLineSeries = (rows, labelCol, valueCol) => {
    const points = [];
    for (const row of rows) {
        const dateVal = parseDateValue(row[labelCol]);
        const numVal = parseNumber(row[valueCol]);
        if (!dateVal || numVal === null) continue;
        points.push({ x: dateVal.getTime(), y: numVal });
    }
    points.sort((a, b) => a.x - b.x);
    if (points.length > CHART_POINT_LIMIT) {
        const step = Math.ceil(points.length / CHART_POINT_LIMIT);
        return points.filter((_, idx) => idx % step === 0);
    }
    return points;
};

const buildBarSeries = (rows, labelCol, valueCol) => {
    const bars = [];
    for (const row of rows) {
        const label = row[labelCol];
        const numVal = parseNumber(row[valueCol]);
        if (label === null || label === undefined || numVal === null) continue;
        bars.push({ label: String(label), value: numVal });
    }
    bars.sort((a, b) => b.value - a.value);
    return bars.slice(0, CHART_BAR_LIMIT);
};

const ChartCard = ({ spec, rows }) => {
    if (!spec) return null;
    const { type, labelCol, valueCol } = spec;
    const series = type === 'line'
        ? buildLineSeries(rows, labelCol, valueCol)
        : buildBarSeries(rows, labelCol, valueCol);

    if (!series.length) return null;

    return (
        <div className="bg-white rounded-2xl border border-black/5 shadow-soft overflow-hidden">
            <div className="px-6 py-4 flex items-center justify-between border-b border-black/5">
                <div>
                    <h3 className="text-lg font-semibold text-[#1e296b]">Chart</h3>
                    <p className="text-xs text-[#64748b] mt-1">
                        {type === 'line' ? 'Line' : 'Bar'} • {valueCol.replace(/_/g, ' ')} by {labelCol.replace(/_/g, ' ')}
                    </p>
                </div>
            </div>
            <div className="px-6 py-4">
                {type === 'line' ? (
                    <LineChart points={series} />
                ) : (
                    <BarChart bars={series} />
                )}
            </div>
        </div>
    );
};

const LineChart = ({ points }) => {
    const width = 600;
    const height = 220;
    const padding = { left: 46, right: 16, top: 16, bottom: 32 };
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;

    const scaleX = (x) =>
        padding.left + ((x - minX) / rangeX) * (width - padding.left - padding.right);
    const scaleY = (y) =>
        height - padding.bottom - ((y - minY) / rangeY) * (height - padding.top - padding.bottom);

    const path = points
        .map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${scaleX(p.x)} ${scaleY(p.y)}`)
        .join(' ');

    const formatDate = (value) => {
        const date = new Date(value);
        return `${date.getMonth() + 1}/${date.getDate()}`;
    };

    return (
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-56">
            {[0, 1, 2, 3].map((i) => {
                const y = padding.top + (i / 3) * (height - padding.top - padding.bottom);
                return (
                    <line
                        key={i}
                        x1={padding.left}
                        x2={width - padding.right}
                        y1={y}
                        y2={y}
                        stroke="#e2e8f0"
                        strokeDasharray="4 4"
                    />
                );
            })}
            <path d={path} fill="none" stroke="#1e296b" strokeWidth="2.5" />
            {points.slice(-1).map((p, idx) => (
                <circle key={idx} cx={scaleX(p.x)} cy={scaleY(p.y)} r="4" fill="#1e296b" />
            ))}
            <text x={padding.left} y={height - 8} fontSize="10" fill="#64748b" textAnchor="start">
                {formatDate(minX)}
            </text>
            <text x={width - padding.right} y={height - 8} fontSize="10" fill="#64748b" textAnchor="end">
                {formatDate(maxX)}
            </text>
            <text x={padding.left - 6} y={padding.top + 4} fontSize="10" fill="#64748b" textAnchor="end">
                {maxY.toLocaleString()}
            </text>
            <text x={padding.left - 6} y={height - padding.bottom} fontSize="10" fill="#64748b" textAnchor="end">
                {minY.toLocaleString()}
            </text>
        </svg>
    );
};

const BarChart = ({ bars }) => {
    const width = 600;
    const height = 220;
    const padding = { left: 36, right: 16, top: 16, bottom: 48 };
    const maxValue = Math.max(...bars.map((b) => b.value), 1);
    const barAreaWidth = width - padding.left - padding.right;
    const gap = 10;
    const barWidth = Math.max((barAreaWidth - gap * (bars.length - 1)) / bars.length, 10);

    return (
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-56">
            {[0, 1, 2, 3].map((i) => {
                const y = padding.top + (i / 3) * (height - padding.top - padding.bottom);
                return (
                    <line
                        key={i}
                        x1={padding.left}
                        x2={width - padding.right}
                        y1={y}
                        y2={y}
                        stroke="#e2e8f0"
                        strokeDasharray="4 4"
                    />
                );
            })}
            {bars.map((bar, idx) => {
                const x = padding.left + idx * (barWidth + gap);
                const barHeight = ((bar.value / maxValue) * (height - padding.top - padding.bottom)) || 0;
                const y = height - padding.bottom - barHeight;
                const label = bar.label.length > 12 ? `${bar.label.slice(0, 11)}…` : bar.label;
                return (
                    <g key={`${bar.label}-${idx}`}>
                        <rect x={x} y={y} width={barWidth} height={barHeight} fill="#1e296b" rx="4" />
                        <text
                            x={x + barWidth / 2}
                            y={height - 22}
                            fontSize="10"
                            fill="#64748b"
                            textAnchor="middle"
                        >
                            {label}
                        </text>
                        <title>{bar.label}</title>
                    </g>
                );
            })}
            <text x={padding.left - 6} y={padding.top + 4} fontSize="10" fill="#64748b" textAnchor="end">
                {maxValue.toLocaleString()}
            </text>
            <text x={padding.left - 6} y={height - padding.bottom} fontSize="10" fill="#64748b" textAnchor="end">
                0
            </text>
        </svg>
    );
};


// Execution Results Display Component - New Design
const ExecutionResultsDisplay = ({ result }) => {
    const [detailsOpen, setDetailsOpen] = useState(false);
    const [page, setPage] = useState(0);
    const pageSize = 10;
    const maxDisplayRows = 100;

    // Safe access to result properties with defaults
    const rows = result?.rows || [];
    const columns = result?.columns || (rows.length > 0 ? Object.keys(rows[0]) : []);
    const totalRows = result?.total_rows ?? result?.row_count ?? rows.length ?? 0;
    const chartSpec = useMemo(() => inferChartSpec(columns, rows), [columns, rows]);

    // Derived state
    const displayableRows = rows.slice(0, maxDisplayRows);
    const totalPages = Math.ceil(displayableRows.length / pageSize);
    const displayedRows = displayableRows.slice(page * pageSize, (page + 1) * pageSize);
    const shownRows = displayableRows.length;

    // Early return if no valid result to display
    if (!result) {
        return null;
    }

    const handleDownloadCSV = () => {
        if (!columns.length || !rows.length) return;

        const escape = (val) => {
            if (val === null || val === undefined) return '';
            const str = String(val).replace(/"/g, '""');
            return `"${str}"`;
        };
        const header = columns.map(escape).join(',');
        const csvRows = rows.map((row) => columns.map((col) => escape(row[col])).join(','));
        const csv = [header, ...csvRows].join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'query_results.csv';
        a.click();
        URL.revokeObjectURL(url);
    };

    // Render summary text with basic markdown-like formatting
    const renderSummary = (text) => {
        if (!text) return null;
        const paragraphs = text.split(/\n\n+/);
        return paragraphs.map((para, pIdx) => {
            // Check for bullet points
            if (para.match(/^[\-\*•]\s/m)) {
                const items = para.split(/\n/).filter(Boolean);
                return (
                    <ul key={pIdx} className="list-disc list-inside space-y-1.5 mb-4">
                        {items.map((item, i) => (
                            <li key={i} className="text-[#1a1f36]">
                                {renderInlineFormatting(item.replace(/^[\-\*•]\s*/, ''))}
                            </li>
                        ))}
                    </ul>
                );
            }
            // Check for numbered lists
            if (para.match(/^\d+\.\s/m)) {
                const items = para.split(/\n/).filter(Boolean);
                return (
                    <ol key={pIdx} className="list-decimal list-inside space-y-1.5 mb-4">
                        {items.map((item, i) => (
                            <li key={i} className="text-[#1a1f36]">
                                {renderInlineFormatting(item.replace(/^\d+\.\s*/, ''))}
                            </li>
                        ))}
                    </ol>
                );
            }
            // Regular paragraph
            return (
                <p key={pIdx} className="text-[#1a1f36] mb-4 last:mb-0 leading-relaxed">
                    {renderInlineFormatting(para)}
                </p>
            );
        });
    };

    // Render inline formatting (bold, percentages, currency)
    const renderInlineFormatting = (text) => {
        const parts = text.split(/(\*\*[^*]+\*\*|\+[\d.]+%|\-[\d.]+%|\$[\d,]+\.?\d*)/g);
        return parts.map((part, i) => {
            if (part.startsWith('**') && part.endsWith('**')) {
                return <strong key={i} className="font-semibold text-[#1a1f36]">{part.slice(2, -2)}</strong>;
            }
            if (part.match(/^\+[\d.]+%$/)) {
                return <span key={i} className="text-green-600 font-semibold">{part}</span>;
            }
            if (part.match(/^\-[\d.]+%$/)) {
                return <span key={i} className="text-red-500 font-semibold">{part}</span>;
            }
            if (part.match(/^\$[\d,]+\.?\d*$/)) {
                return <strong key={i} className="font-semibold text-[#1a1f36]">{part}</strong>;
            }
            return part;
        });
    };

    // Format cell value with color for growth percentages
    const formatCellValue = (value, colName) => {
        if (value === null || value === undefined) return '-';
        const strVal = String(value);
        const colLower = colName.toLowerCase();
        if (colLower.includes('growth') || colLower.includes('change') || colLower.includes('%')) {
            const num = parseFloat(strVal.replace(/[%,]/g, ''));
            if (!isNaN(num)) {
                if (num > 0) return <span className="text-green-600 font-medium">+{num}%</span>;
                if (num < 0) return <span className="text-red-500 font-medium">{num}%</span>;
                return <span className="text-[#64748b]">{num}%</span>;
            }
        }
        return strVal;
    };

    return (
        <div className="mt-4 space-y-6">
            {/* Summary + Execution Details */}
            {(result.summary || result.execution_time_ms !== undefined) && (
                <div className="bg-white rounded-2xl border border-black/5 shadow-soft overflow-hidden">
                    {result.summary && (
                        <div className="relative px-6 py-6">
                            <span className="absolute left-0 top-0 h-full w-1 bg-[#1e296b]" aria-hidden="true" />
                            <div className="pl-4">
                                <div className="flex items-center gap-2 mb-4">
                                    <Sparkles className="w-5 h-5 text-[#1e296b]" />
                                    <h3 className="text-lg font-semibold text-[#1a1f36]">Summary</h3>
                                </div>
                                <div className="text-base text-[#1a1f36]">
                                    {renderSummary(result.summary)}
                                </div>
                            </div>
                        </div>
                    )}
                    <button
                        type="button"
                        onClick={() => setDetailsOpen(!detailsOpen)}
                        className="w-full px-6 py-4 flex items-center justify-between border-t border-black/5 bg-[#fbfcff] hover:bg-black/[0.02] transition-colors"
                    >
                        <div className="flex items-center gap-3">
                            <Database className="w-4 h-4 text-[#64748b]" />
                            <span className="text-sm font-medium text-[#1a1f36]">Execution Details</span>
                        </div>
                        {detailsOpen ? (
                            <ChevronUp className="w-4 h-4 text-[#64748b]" />
                        ) : (
                            <ChevronDown className="w-4 h-4 text-[#64748b]" />
                        )}
                    </button>
                    {detailsOpen && (
                        <div className="px-6 py-4 border-t border-black/5 bg-[#fbfcff] text-sm">
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <span className="text-[#64748b]">Execution time:</span>
                                    <span className="ml-2 font-medium text-[#1a1f36]">{result.execution_time_ms}ms</span>
                                </div>
                                <div>
                                    <span className="text-[#64748b]">Bytes processed:</span>
                                    <span className="ml-2 font-medium text-[#1a1f36]">
                                        {((result.bytes_processed || 0) / 1024 / 1024).toFixed(2)} MB
                                    </span>
                                </div>
                                {result.cached && (
                                    <div>
                                        <span className="text-[#64748b]">Cache:</span>
                                        <span className="ml-2 font-medium text-green-600">Hit</span>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {chartSpec && (
                <ChartCard spec={chartSpec} rows={rows} />
            )}

            {/* Query Results Section */}
            <div className="bg-white rounded-2xl border border-black/5 shadow-soft overflow-hidden">
                {/* Header */}
                <div className="px-6 py-4 flex items-center justify-between border-b border-black/5">
                    <h3 className="text-lg font-semibold text-[#1e296b]">Query Results</h3>
                    <button
                        type="button"
                        onClick={handleDownloadCSV}
                        className="flex items-center gap-2 px-3 py-1.5 border border-black/10 rounded-lg text-xs font-semibold text-[#1a1f36] hover:bg-black/[0.02] transition-colors"
                    >
                        <Download className="w-4 h-4" />
                        CSV
                    </button>
                </div>

                {/* Table */}
                <div className="overflow-x-auto">
                    {displayedRows.length === 0 || columns.length === 0 ? (
                        <div className="text-center py-12 text-[#64748b]">No results found</div>
                    ) : (
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="bg-[#1e296b]">
                                    {columns.map((col) => (
                                        <th
                                            key={col}
                                            className="text-left px-6 py-3 text-xs font-bold uppercase tracking-wider text-white whitespace-nowrap"
                                        >
                                            {col.replace(/_/g, ' ')}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {displayedRows.map((row, idx) => (
                                    <tr
                                        key={idx}
                                        className="border-b border-black/5 hover:bg-black/[0.01] transition-colors"
                                    >
                                        {columns.map((col) => (
                                            <td key={col} className="px-6 py-3 text-[#1a1f36] whitespace-nowrap">
                                                {formatCellValue(row[col], col)}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    )}
                </div>

                {/* Pagination Footer */}
                {(columns.length > 0 && shownRows > 0) && (
                    <div className="px-6 py-3 flex items-center justify-between border-t border-black/5 bg-[#f8f9fc]">
                        <span className="text-sm text-[#64748b]">
                            Showing <span className="font-semibold text-[#1a1f36]">{shownRows.toLocaleString()}</span> of{' '}
                            <span className="font-semibold text-[#1a1f36]">{totalRows.toLocaleString()}</span> rows
                        </span>
                        {totalPages > 1 && (
                            <div className="flex items-center gap-2">
                                <button
                                    type="button"
                                    onClick={() => setPage((p) => Math.max(0, p - 1))}
                                    disabled={page === 0}
                                    className="w-8 h-8 flex items-center justify-center border border-black/10 rounded-lg text-[#64748b] hover:bg-black/[0.02] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                                >
                                    <ChevronLeft className="w-4 h-4" />
                                </button>
                                <button
                                    type="button"
                                    onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                                    disabled={page >= totalPages - 1}
                                    className="w-8 h-8 flex items-center justify-center border border-black/10 rounded-lg text-[#64748b] hover:bg-black/[0.02] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                                >
                                    <ChevronRight className="w-4 h-4" />
                                </button>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};


// Chat Message Component
const ChatMessage = ({ message, onExecuteSQL, isExecuting, onTableClick, onSuggestion }) => {
    const [copied, setCopied] = useState(false);
    const [lineageControls, setLineageControls] = useState(null);
    const [lineageExpanded, setLineageExpanded] = useState(true);
    const [toolProgress, setToolProgress] = useState(0);

    const toolSteps = useMemo(() => {
        const mapStep = (raw) => {
            const normalized = String(raw).toLowerCase();
            if (normalized.includes("search")) return "Search Tables";
            if (normalized.includes("generate")) return "Generate SQL";
            if (normalized.includes("validate_table")) return "Validate Table";
            if (normalized.includes("validate_query")) return "Validate Query";
            if (normalized.includes("assess")) return "Assess SQL";
            if (normalized.includes("validate")) return "Validate Query";
            if (normalized.includes("execute")) return "Execute SQL";
            if (normalized.includes("lineage")) return "Get Lineage";
            if (normalized.includes("impact")) return "Impact Assessment";
            if (normalized.includes("classify")) return "Classify Intent";
            if (normalized.includes("route")) return "Route Request";
            if (normalized.includes("call")) return "Call Tool";
            return String(raw).replace(/_/g, " ");
        };
        if (Array.isArray(message.toolTraceText) && message.toolTraceText.length > 0) {
            return message.toolTraceText.map(mapStep);
        }
        if (Array.isArray(message.toolCalls) && message.toolCalls.length > 0) {
            return message.toolCalls.map((tool) => mapStep(tool?.name || "Tool Call"));
        }
        return ["Search Tables", "Generate SQL", "Assess SQL", "Validate Query"];
    }, [message.toolTraceText, message.toolCalls]);

    useEffect(() => {
        if (!message.pending) {
            setToolProgress(0);
            return undefined;
        }
        setToolProgress(0);
        const maxActive = Math.max(toolSteps.length - 1, 0);
        const interval = setInterval(() => {
            setToolProgress((prev) => Math.min(prev + 1, maxActive));
        }, 900);
        return () => clearInterval(interval);
    }, [message.pending, toolSteps.length]);

    const handleCopySql = (sql) => {
        navigator.clipboard.writeText(sql);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    if (message.role === 'user') {
        return (
            <div className="flex justify-end w-full animate-fade-in">
                <div className="flex gap-4 max-w-[50%] flex-row-reverse">
                    <div className="w-10 h-10 rounded-xl bg-[#1e296b] flex items-center justify-center flex-shrink-0">
                        <div className="w-2.5 h-2.5 rounded-full bg-white" />
                    </div>
                    <div className="message-bubble-user">
                        <p className="text-base leading-relaxed text-left break-words">{message.content}</p>
                    </div>
                </div>
            </div>
        );
    }

        return (
            <div className="flex justify-start w-full animate-fade-in">
                <div className="flex gap-4 w-full">
                    <div className="w-10 h-10 rounded-xl bg-white border border-black/5 flex items-center justify-center flex-shrink-0 shadow-soft">
                        <Sparkles className="w-5 h-5 text-blue-500" />
                    </div>
                    <div className="flex flex-col gap-3 flex-1">
                    {/* Tooling Trace: show while thinking, collapse after final response */}
                    {message.pending && (
                        <div className="flex flex-col gap-3">
                            <div className="inline-flex items-center gap-3 px-4 py-2.5 bg-white/80 rounded-2xl border border-black/5 shadow-soft">
                                <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
                                <span className="text-sm text-[#94a3b8] font-medium">Thinking...</span>
                            </div>
                            <div className="pl-1 space-y-2">
                                {toolSteps.map((step, idx) => {
                                    const isDone = idx < toolProgress;
                                    const isActive = idx === toolProgress;
                                    return (
                                        <div key={`${step}-${idx}`} className="flex items-center gap-2 text-sm">
                                            {isDone ? (
                                                <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                                            ) : isActive ? (
                                                <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
                                            ) : (
                                                <div className="w-3.5 h-3.5 rounded-full border border-[#cbd5f5] bg-[#f1f5ff]" />
                                            )}
                                            <span
                                                className={
                                                    isDone
                                                        ? "text-[#1a1f36] font-medium"
                                                        : isActive
                                                            ? "text-[#1a1f36] font-medium"
                                                            : "text-[#94a3b8]"
                                                }
                                            >
                                                Tool: {step}
                                            </span>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    )}

                    {/* Main Message */}
                    <div className="message-bubble-assistant">
                        <p className="text-base leading-relaxed font-medium">{message.content}</p>
                    </div>

                    {/* Impact Assessment */}
                    {message.impactAssessment && !message.impactAssessment.error && (
                        <RippleReport assessment={message.impactAssessment} />
                    )}

                    {/* Table Validation */}
                    {message.tableValidation && (
                        <TableValidationReport validation={message.tableValidation} />
                    )}

                    {/* SQL Block */}
                    {message.sql && (
                        <div className="bg-[#1a1f36] rounded-2xl overflow-hidden shadow-elevated border border-white/5">
                            {/* SQL Header */}
                            <div className="px-5 py-4 bg-white/5 border-b border-white/5 flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                    <div className="w-9 h-9 rounded-lg bg-white/10 flex items-center justify-center">
                                        <Code className="w-4 h-4 text-white/80" />
                                    </div>
                                    <div>
                                        <span className="text-sm font-semibold text-white/90">BigQuery SQL</span>
                                        <span className="block text-[9px] uppercase font-semibold text-white/30 tracking-wider">
                                            Generated via Gemini
                                        </span>
                                    </div>
                                </div>
                                <div className="flex items-center gap-2">
                                    {message.confidence && (
                                        <div className="px-3 py-1.5 rounded-lg bg-green-500/10 border border-green-500/20 flex items-center gap-1.5">
                                            <CheckCircle2 className="w-3.5 h-3.5 text-green-400" />
                                            <span className="text-xs font-semibold text-green-400">
                                                {(message.confidence * 100).toFixed(0)}%
                                            </span>
                                        </div>
                                    )}
                                    <button
                                        onClick={() => handleCopySql(message.sql)}
                                        className="p-2 hover:bg-white/10 rounded-lg transition-colors"
                                        title="Copy SQL"
                                    >
                                        {copied ? (
                                            <Check className="w-4 h-4 text-green-400" />
                                        ) : (
                                            <Copy className="w-4 h-4 text-white/40" />
                                        )}
                                    </button>
                                    <button
                                        onClick={() => onExecuteSQL(message.sql, message.id)}
                                        disabled={isExecuting}
                                        className="flex items-center gap-2 px-4 py-2 bg-white text-[#1a1f36] hover:bg-white/90 rounded-xl text-sm font-bold shadow-lg transition-all active:scale-95 disabled:opacity-50"
                                    >
                                        {isExecuting ? (
                                            <Loader2 className="w-4 h-4 animate-spin" />
                                        ) : (
                                            <Play className="w-4 h-4 fill-current" />
                                        )}
                                        {isExecuting ? 'Running' : 'Run'}
                                    </button>
                                </div>
                            </div>

                            {/* SQL Code - with proper word wrap */}
                            <div className="overflow-hidden">
                                <SyntaxHighlighter
                                    language="sql"
                                    style={vscDarkPlus}
                                    customStyle={{
                                        margin: 0,
                                        borderRadius: 0,
                                        padding: '1.25rem 1.5rem',
                                        background: 'transparent',
                                        fontFamily: '"Söhne Mono", "SF Mono", Menlo, Monaco, Consolas, "Liberation Mono", monospace',
                                        fontSize: '0.875rem',
                                        lineHeight: '1.6',
                                        whiteSpace: 'pre-wrap',
                                        wordBreak: 'break-word',
                                    }}
                                    wrapLines
                                    wrapLongLines
                                >
                                    {message.sql}
                                </SyntaxHighlighter>
                            </div>

                            {/* Tables Used */}
                            {message.tablesUsed?.length > 0 && (
                                <div className="px-5 py-3 bg-black/20 border-t border-white/5 flex items-center gap-4 overflow-x-auto no-scrollbar">
                                    <span className="text-[9px] font-semibold text-white/30 uppercase tracking-widest whitespace-nowrap">
                                        Tables
                                    </span>
                                    <div className="flex gap-1.5">
                                        {message.tablesUsed.map((t) => (
                                            <span
                                                key={t}
                                                className="px-3 py-1 bg-white/5 border border-white/5 rounded-full text-[10px] font-semibold text-white/60 whitespace-nowrap"
                                            >
                                                {t}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {/* Query Validation */}
                    {message.queryValidation && (
                        <QueryValidationCard validation={message.queryValidation} />
                    )}

                    {/* Inline Execution Results - New Design */}
                    {message.executionResult && (
                        <ExecutionResultsDisplay result={message.executionResult} />
                    )}

                    {/* Execution Error */}
                    {message.executionError && (
                        <div className="mt-3 bg-red-50 rounded-2xl border border-red-200 px-5 py-3">
                            <div className="flex items-center gap-2 text-red-600">
                                <X className="w-4 h-4" />
                                <span className="text-sm font-medium">Execution failed: {message.executionError}</span>
                            </div>
                        </div>
                    )}

                    {/* Search Results - Horizontal Cards matching screenshot */}
                    {(message.intent === 'get_matching_tables'
                        || message.intent === 'get_matching_columns'
                        || (message.needsSelection && message.searchResults?.length > 0)) && (
                            <div className="mt-2">
                                {(message.intent === 'get_matching_tables' || message.intent === 'get_matching_columns') && (
                                    <div className="flex items-center gap-2 mb-4">
                                        <div className="w-6 h-6 rounded-lg bg-purple-100 flex items-center justify-center">
                                            <Sparkles className="w-3.5 h-3.5 text-purple-600" />
                                        </div>
                                        <span className="text-xs font-semibold uppercase tracking-wider text-[#64748b]">
                                            {(() => {
                                                const count = Math.min(message.searchResults?.length || 0, 5);
                                                if (count === 0) return 'No Matches found';
                                                if (message.intent === 'get_matching_tables') {
                                                    return `Below are top ${count} tables matched`;
                                                }
                                                return `Below are top ${count} tables matched with relevant columns`;
                                            })()}
                                        </span>
                                    </div>
                                )}

                                {message.searchResults?.length > 0 && (
                                    <div className="flex gap-4 overflow-x-auto pb-2 no-scrollbar">
                                        {message.searchResults.slice(0, 5).map((result, idx) => (
                                            <TableResultCard
                                                key={idx}
                                                result={result}
                                                onClick={onTableClick}
                                            />
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}

                    {/* Lineage Graph - Visual Graph design matching screenshot */}
                    {message.lineageEdges?.length > 0 && (
                        <LineageSummaryCard
                            edges={message.lineageEdges}
                            focusTable={message.lineageTable}
                        />
                    )}

                    {/* Lineage Graph - Visual Graph design matching screenshot */}
                    {message.lineageEdges?.length > 0 && (
                        <div className="bg-white rounded-2xl border border-black/10 overflow-hidden shadow-soft">
                            {/* Visual Graph Header */}
                            <div className="px-6 py-4 flex items-center justify-between border-b border-black/5">
                                <div>
                                    <h3 className="text-lg font-bold text-[#1a1f36]">Lineage Graph</h3>
                                    <p className="text-xs text-[#94a3b8] flex items-center gap-2">
                                        <span>Data flow</span>
                                        <span className="inline-flex items-center gap-1 text-[#1e296b] font-semibold">
                                            <span>Upstream</span>
                                            <span>→</span>
                                            <span>Downstream</span>
                                        </span>
                                    </p>
                                </div>
                                <div className="flex items-center gap-1">
                                    {lineageExpanded && (
                                        <>
                                            <button
                                                onClick={() => lineageControls?.zoomIn()}
                                                className="w-8 h-8 rounded-lg border border-black/10 flex items-center justify-center hover:bg-black/5 transition-colors"
                                                title="Zoom In"
                                            >
                                                <Plus className="w-4 h-4 text-[#64748b]" />
                                            </button>
                                            <button
                                                onClick={() => lineageControls?.zoomOut()}
                                                className="w-8 h-8 rounded-lg border border-black/10 flex items-center justify-center hover:bg-black/5 transition-colors"
                                                title="Zoom Out"
                                            >
                                                <span className="text-[#64748b] font-bold text-lg leading-none">−</span>
                                            </button>
                                            <button
                                                onClick={() => lineageControls?.fitView()}
                                                className="w-8 h-8 rounded-lg border border-black/10 flex items-center justify-center hover:bg-black/5 transition-colors"
                                                title="Fit to View"
                                            >
                                                <svg className="w-4 h-4 text-[#64748b]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                                                </svg>
                                            </button>
                                        </>
                                    )}
                                    <button
                                        onClick={() => setLineageExpanded(!lineageExpanded)}
                                        className="w-8 h-8 rounded-lg border border-black/10 flex items-center justify-center hover:bg-black/5 transition-colors"
                                        title={lineageExpanded ? "Minimize" : "Expand"}
                                    >
                                        {lineageExpanded ? (
                                            <ChevronUp className="w-4 h-4 text-[#64748b]" />
                                        ) : (
                                            <ChevronDown className="w-4 h-4 text-[#64748b]" />
                                        )}
                                    </button>
                                </div>
                            </div>
                            {/* Graph Container */}
                            {lineageExpanded && (
                                <div className="h-80 bg-gradient-to-br from-[#f7f8ff] via-[#f5f6ff] to-[#eef0ff]">
                                    <LineageGraphWithProvider
                                        edges={message.lineageEdges}
                                        focusTable={message.lineageTable}
                                        onControlsReady={setLineageControls}
                                    />
                                </div>
                            )}
                        </div>
                    )}

                    {/* Inline Schema Display */}
                    {message.schemaData && (
                        <InlineSchemaDisplay
                            table={message.schemaData.table}
                            schema={message.schemaData.schema}
                            loading={false}
                            error={null}
                            onSuggestion={onSuggestion}
                        />
                    )}
                </div>
            </div>
        </div >
    );
};

const RippleReport = ({ assessment }) => {
    if (!assessment) return null;
    const level = (assessment.impact_level || 'MEDIUM').toUpperCase();
    const levelLabel = level.charAt(0) + level.slice(1).toLowerCase();
    const tone = {
        HIGH: { badge: 'bg-red-50 text-red-600 border-red-200' },
        MEDIUM: { badge: 'bg-amber-50 text-amber-600 border-amber-200' },
        LOW: { badge: 'bg-emerald-50 text-emerald-600 border-emerald-200' },
    }[level] || { badge: 'bg-slate-50 text-slate-600 border-slate-200' };

    const logo = level === 'LOW' ? rippleLogo : level === 'MEDIUM' ? waveLogo : tsunamiLogo;
    const impactLabel = level === 'HIGH' ? 'Critical Impact' : level === 'MEDIUM' ? 'Moderate Impact' : 'Low Impact';

    const keyFindings = Array.isArray(assessment.impact_reasoning) && assessment.impact_reasoning.length > 0
        ? assessment.impact_reasoning
        : [
            assessment.downstream_tables?.length
                ? `Downstream dependencies: ${assessment.downstream_tables.length} tables affected.`
                : null,
            assessment.has_upstream !== undefined
                ? `Upstream dependencies: ${assessment.has_upstream ? 'Yes' : 'None detected'}.`
                : null,
            assessment.column
                ? `Column impacted: ${assessment.column}.`
                : null,
            assessment.table
                ? `Target table: ${assessment.table}.`
                : null,
        ].filter(Boolean);

    const recommendations = [
        ...(Array.isArray(assessment.recommendation_reasoning) ? assessment.recommendation_reasoning : []),
        ...(Array.isArray(assessment.actions) ? assessment.actions : []),
    ];
    if (recommendations.length === 0 && assessment.recommendation) {
        recommendations.push(assessment.recommendation);
    }

    return (
        <div className="bg-[#f8f9fc] rounded-3xl border border-black/5 shadow-soft p-6">
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold text-[#1a1f36]">Ripple Report</h3>
                <span className="px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider bg-white border border-black/10 text-[#64748b]">
                    V1.2 Analysis
                </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-[200px_1fr] gap-4">
                <div className="bg-white rounded-2xl border border-black/5 p-5 flex flex-col items-center text-center">
                    <img
                        src={logo}
                        alt={`${levelLabel} impact`}
                        className="w-20 h-20 object-contain"
                    />
                    <div className="mt-3 text-[10px] font-black uppercase tracking-widest text-[#94a3b8]">Risk Level</div>
                    <div className="mt-2 text-lg font-bold text-[#1a1f36]">{levelLabel}</div>
                    <div className={`mt-3 px-3 py-1 rounded-full text-xs font-semibold border ${tone.badge}`}>
                        {impactLabel}
                    </div>
                </div>

                <div className="bg-white rounded-2xl border border-black/5 p-5">
                    <div className="flex items-center gap-2 mb-3">
                        <div className="w-6 h-6 rounded-lg bg-orange-100 flex items-center justify-center text-orange-600 text-xs font-bold">
                            !
                        </div>
                        <span className="text-xs font-semibold uppercase tracking-wider text-[#64748b]">Key Findings</span>
                    </div>
                    <div className="space-y-3 text-sm text-[#1a1f36]">
                        {keyFindings.slice(0, 3).map((item, idx) => (
                            <div key={idx} className="flex gap-3">
                                <div className="w-5 h-5 rounded-full bg-[#eef2ff] text-[#1e296b] text-xs font-bold flex items-center justify-center">
                                    {idx + 1}
                                </div>
                                <div className="leading-relaxed">{item}</div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <div className="mt-4 bg-white rounded-2xl border border-black/5 p-5">
                <div className="flex items-center gap-2 mb-3">
                    <div className="w-6 h-6 rounded-lg bg-[#eef2ff] flex items-center justify-center">
                        <Sparkles className="w-3.5 h-3.5 text-[#1e296b]" />
                    </div>
                    <span className="text-xs font-semibold uppercase tracking-wider text-[#64748b]">Recommendations</span>
                </div>
                <div className="space-y-3 text-sm text-[#1a1f36]">
                    {recommendations.length > 0 ? (
                        recommendations.slice(0, 3).map((item, idx) => (
                            <div key={idx} className="flex gap-3">
                                <div className="w-5 h-5 rounded-full bg-[#eef2ff] text-[#1e296b] text-xs font-bold flex items-center justify-center">
                                    {idx + 1}
                                </div>
                                <div className="leading-relaxed">{item}</div>
                            </div>
                        ))
                    ) : (
                        <div className="text-[#64748b]">No recommendations provided.</div>
                    )}
                </div>
            </div>
        </div>
    );
};

const QueryValidationCard = ({ validation }) => {
    if (!validation) return null;
    const recommendationRaw = String(validation.recommendation || "UNKNOWN");
    const recommendation = recommendationRaw.toUpperCase();
    const tone = {
        APPROVE: { badge: 'bg-emerald-50 text-emerald-600 border-emerald-200' },
        WARN: { badge: 'bg-amber-50 text-amber-600 border-amber-200' },
        REJECT: { badge: 'bg-red-50 text-red-600 border-red-200' },
        'NEEDS REVIEW': { badge: 'bg-amber-50 text-amber-600 border-amber-200' },
        UNKNOWN: { badge: 'bg-slate-50 text-slate-600 border-slate-200' },
    }[recommendation] || { badge: 'bg-slate-50 text-slate-600 border-slate-200' };

    const cost = validation.cost || {};
    const efficiency = validation.efficiency || {};
    const issues = Array.isArray(validation.issues) ? validation.issues : [];
    const suggestions = Array.isArray(validation.suggestions) ? validation.suggestions : [];

    return (
        <div className="bg-white rounded-2xl border border-black/10 p-5 shadow-soft">
            <div className="flex items-center justify-between mb-3">
                <div>
                    <h3 className="text-base font-bold text-[#1a1f36]">Query Validation</h3>
                    <p className="text-xs text-[#94a3b8]">Cost + efficiency gate</p>
                </div>
                <span
                    className={`px-3 py-1 rounded-full text-[10px] font-bold tracking-wider border ${tone.badge} ${recommendationRaw.toLowerCase() === 'needs review' ? '' : 'uppercase'}`}
                >
                    {recommendationRaw.toLowerCase() === 'needs review' ? 'needs review' : recommendation}
                </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div className="bg-[#f8fafc] rounded-xl border border-black/5 p-3">
                    <div className="text-[10px] font-semibold uppercase tracking-wider text-[#94a3b8]">Estimated Cost</div>
                    <div className="text-sm font-semibold text-[#1a1f36]">
                        {typeof cost.estimated_cost_usd === 'number' ? `$${cost.estimated_cost_usd.toFixed(2)}` : 'unknown'}
                    </div>
                    <div className="text-[10px] text-[#94a3b8]">
                        Budget {typeof cost.budget_usd === 'number' ? `$${cost.budget_usd.toFixed(2)}` : 'unknown'}
                    </div>
                </div>
                <div className="bg-[#f8fafc] rounded-xl border border-black/5 p-3">
                    <div className="text-[10px] font-semibold uppercase tracking-wider text-[#94a3b8]">Bytes Processed</div>
                    <div className="text-sm font-semibold text-[#1a1f36]">
                        {typeof cost.bytes_processed_gb === 'number' ? `${cost.bytes_processed_gb.toFixed(2)} GB` : 'unknown'}
                    </div>
                    <div className="text-[10px] text-[#94a3b8]">
                        {cost.referenced_tables?.length || 0} referenced tables
                    </div>
                </div>
                <div className="bg-[#f8fafc] rounded-xl border border-black/5 p-3">
                    <div className="text-[10px] font-semibold uppercase tracking-wider text-[#94a3b8]">Efficiency</div>
                    <div className="text-sm font-semibold text-[#1a1f36]">
                        {efficiency.is_efficient ? 'Efficient' : 'Needs Review'}
                    </div>
                    <div className="text-[10px] text-[#94a3b8]">
                        {issues.length} issues detected
                    </div>
                </div>
            </div>

            <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="bg-white rounded-xl border border-black/5 p-4">
                    <div className="text-[10px] font-semibold uppercase tracking-wider text-[#94a3b8] mb-2">Issues</div>
                    {issues.length ? (
                        <div className="space-y-2 text-sm text-[#1a1f36]">
                            {issues.slice(0, 4).map((item, idx) => (
                                <div key={idx} className="flex gap-2">
                                    <span className="text-[#94a3b8]">{idx + 1}.</span>
                                    <span>{item}</span>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="text-sm text-[#64748b]">No issues detected.</div>
                    )}
                </div>
                <div className="bg-white rounded-xl border border-black/5 p-4">
                    <div className="text-[10px] font-semibold uppercase tracking-wider text-[#94a3b8] mb-2">Suggestions</div>
                    {suggestions.length ? (
                        <div className="space-y-2 text-sm text-[#1a1f36]">
                            {suggestions.slice(0, 4).map((item, idx) => (
                                <div key={idx} className="flex gap-2">
                                    <span className="text-[#94a3b8]">{idx + 1}.</span>
                                    <span>{item}</span>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="text-sm text-[#64748b]">No suggestions.</div>
                    )}
                </div>
            </div>
        </div>
    );
};

const TableValidationReport = ({ validation }) => {
    if (!validation) return null;
    const [showAllViolations, setShowAllViolations] = useState(false);
    const cliOutput = typeof validation.cli_output === 'string' ? validation.cli_output.trim() : '';
    if (cliOutput) {
        return (
            <div className="bg-white rounded-3xl border border-black/5 shadow-soft overflow-hidden">
                <div className="px-6 pt-6 pb-4 border-b border-black/5">
                    <div className="text-[10px] font-semibold uppercase tracking-wider text-[#64748b]">
                        BigQuery • Validation
                    </div>
                    <div className="mt-2 text-lg font-bold text-[#1a1f36]">CLI Validation Report</div>
                    <div className="text-xs text-[#94a3b8]">
                        Rendered from `data-valid-ai validate` output.
                    </div>
                </div>
                <div className="px-6 py-5">
                    <div className="bg-[#0b1220] text-[#e2e8f0] rounded-2xl border border-black/10 p-4 overflow-x-auto">
                        <pre className="m-0 font-mono text-xs leading-5 whitespace-pre">{cliOutput}</pre>
                    </div>
                </div>
            </div>
        );
    }
    const result = validation.validation_result || {};
    const summary = validation.llm_summary || {};
    const violations = Array.isArray(result.violations) ? result.violations : [];
    const passRate = typeof result.pass_rate === 'number'
        ? result.pass_rate.toFixed(1)
        : null;
    const tableRef = result.table || '';
    const tableParts = tableRef.split('.').filter(Boolean);
    const datasetName = tableParts.length >= 2 ? tableParts[tableParts.length - 2] : tableRef || 'dataset';
    const tableName = tableParts.length ? tableParts[tableParts.length - 1] : 'table';
    const datasetTable = tableParts.length >= 2 ? `${tableParts[tableParts.length - 2]}.${tableParts[tableParts.length - 1]}` : tableRef || 'dataset.table';
    const projectName = tableParts.length >= 3 ? tableParts[0] : DEFAULT_PROJECT;
    const passedRulesList = Array.isArray(result.passed_rules)
        ? result.passed_rules
        : Array.isArray(result.passed_rule_names)
            ? result.passed_rule_names
            : [];

    const summaryText = summary?.summary && summary.summary !== 'unknown'
        ? summary.summary
        : violations.length
            ? `The validation suite identified ${violations.length} critical violation${violations.length === 1 ? '' : 's'}. ${passRate ? `While ${passRate}% of rules passed,` : 'Review'} remediation is required.`
            : `No critical violations detected${passRate ? `. ${passRate}% of rules passed.` : '.'}`;

    const remediationFallback = violations
        .map((item) => item.remediation_sql ? `SQL: ${item.remediation_sql}` : item.remediation_suggestion)
        .filter(Boolean);
    const remediation = Array.isArray(summary?.recommendations) && summary.recommendations.length
        ? summary.recommendations
        : remediationFallback;

    const totalRules = result.rules_executed ?? ((result.rules_passed ?? 0) + (result.rules_failed ?? 0) + (result.rules_skipped ?? 0));
    const failedRules = result.rules_failed ?? violations.length;
    const passRateValue = passRate
        ? Number(passRate)
        : totalRules
            ? (100 * (result.rules_passed ?? 0)) / totalRules
            : 0;
    const durationSeconds = result.duration_seconds ?? result.duration ?? result.metadata?.duration_seconds;
    const durationLabel = typeof durationSeconds === 'number' ? `${durationSeconds.toFixed(2)}s` : 'unknown';
    const qualityScore = Number.isFinite(passRateValue) ? `${Math.round(passRateValue)}%` : 'N/A';

    const remediationItems = remediation.map((item) => item);
    const normalizedRecommendations = remediationItems.map((item) => {
        if (typeof item === 'string') {
            return {
                rule: 'Recommendation',
                recommendation: item,
                sql: item.toLowerCase().startsWith('sql:') ? item.slice(4).trim() : 'unknown',
            };
        }
        return {
            rule: item?.rule || 'Recommendation',
            recommendation: item?.recommendation || 'unknown',
            sql: item?.sql || 'unknown',
        };
    });
    const risks = Array.isArray(summary?.risks) && summary.risks.length
        ? summary.risks
        : violations.map((item) => item.message || item.rule_name || 'Issue detected').filter(Boolean);
    const nextSteps = Array.isArray(summary?.next_steps) ? summary.next_steps : [];

    return (
        <div className="bg-white rounded-3xl border border-black/5 shadow-soft overflow-hidden">
            <div className="px-6 pt-6 pb-4 border-b border-black/5">
                <div className="flex items-center justify-between">
                    <div>
                        <div className="text-[10px] font-semibold uppercase tracking-wider text-[#64748b]">
                            BigQuery • {result.layer ? `${result.layer.charAt(0).toUpperCase()}${result.layer.slice(1)} Layer` : 'Validation'}
                        </div>
                        <div className="mt-2 text-lg font-bold text-[#1a1f36]">{tableName}</div>
                        <div className="text-xs text-[#94a3b8] font-mono">
                            {projectName}.{datasetTable}
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        <button className="px-3 py-1.5 rounded-full border border-black/10 text-xs font-semibold text-[#475569] hover:bg-black/[0.02]">
                            Re-run
                        </button>
                        <button className="px-3 py-1.5 rounded-full border border-black/10 text-xs font-semibold text-[#1e296b] bg-[#eef2ff] hover:bg-[#e0e7ff]">
                            Edit Rules
                        </button>
                    </div>
                </div>

                <div className="mt-5 grid grid-cols-1 md:grid-cols-4 gap-3">
                    <div className="bg-[#f8f9fc] rounded-2xl border border-black/5 p-4">
                        <div className="text-[10px] font-semibold uppercase tracking-wider text-[#94a3b8]">Rules Executed</div>
                        <div className="text-xl font-bold text-[#1a1f36]">{totalRules ?? 0}</div>
                    </div>
                    <div className="bg-[#fff5f5] rounded-2xl border border-[#ffd7d7] p-4">
                        <div className="text-[10px] font-semibold uppercase tracking-wider text-[#e11d48]">Rules Failed</div>
                        <div className="text-xl font-bold text-[#e11d48]">{failedRules ?? 0}</div>
                    </div>
                    <div className="bg-[#f8f9fc] rounded-2xl border border-black/5 p-4">
                        <div className="text-[10px] font-semibold uppercase tracking-wider text-[#94a3b8]">Pass Rate</div>
                        <div className="text-xl font-bold text-[#1a1f36]">{qualityScore}</div>
                        <div className="text-[10px] text-[#94a3b8]">Quality Score</div>
                    </div>
                    <div className="bg-[#f8f9fc] rounded-2xl border border-black/5 p-4">
                        <div className="text-[10px] font-semibold uppercase tracking-wider text-[#94a3b8]">Duration</div>
                        <div className="text-xl font-bold text-[#1a1f36]">{durationLabel}</div>
                    </div>
                </div>
            </div>

            <div className="px-6 py-5 border-b border-black/5">
                <div className="text-xs font-semibold uppercase tracking-wider text-[#94a3b8] mb-2">
                    Passed Rules ({result.rules_passed ?? 0})
                </div>
                <div className="space-y-2">
                    {passedRulesList.length ? (
                        passedRulesList.slice(0, 3).map((item, idx) => {
                            const label = typeof item === 'string'
                                ? item
                                : item?.rule_name || item?.rule_id || JSON.stringify(item);
                            return (
                                <div key={idx} className="flex items-center gap-3 bg-white border border-black/5 rounded-2xl px-4 py-2">
                                    <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                                    <span className="text-sm text-[#1a1f36]">{label}</span>
                                </div>
                            );
                        })
                    ) : (
                        <div className="text-sm text-[#64748b]">
                            Rules passed: {result.rules_passed ?? 0}. Details not provided for {datasetTable}.
                        </div>
                    )}
                </div>
            </div>

            <div className="px-6 py-5 border-b border-black/5">
                <div className="flex items-center justify-between mb-3">
                    <div className="text-sm font-semibold text-[#1a1f36]">Rule Assessment</div>
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-red-500">
                        {violations.length} Violations
                    </span>
                </div>
                <div className="space-y-2">
                    <div className="grid grid-cols-[80px_90px_1fr] text-[10px] font-semibold uppercase tracking-wider text-[#94a3b8] px-3">
                        <div>Status</div>
                        <div>Rule ID</div>
                        <div>Rule Name</div>
                    </div>
                    {violations.length ? (
                        (showAllViolations ? violations : violations.slice(0, 5)).map((item, idx) => {
                            const ruleId = item.rule_id || item.rule_name || `RULE-${idx + 1}`;
                            const ruleName = item.rule_name || item.rule_id || 'Validation rule failed';
                            const message = item.message || 'Rule failed';
                            const column = item.column || item.column_name || item.field;
                            const detail = column ? `Column: ${column}` : message;
                            return (
                                <div key={idx} className="bg-[#fff5f5] border border-[#ffe1e1] rounded-2xl px-3 py-3">
                                    <div className="grid grid-cols-[80px_90px_1fr] items-start gap-2">
                                        <div className="px-2 py-1 rounded-full bg-red-600 text-white text-[10px] font-bold w-fit">
                                            Failed
                                        </div>
                                        <div className="text-[11px] text-[#9ca3af] font-mono">{ruleId}</div>
                                        <div>
                                            <div className="text-sm font-semibold text-[#1a1f36]">{ruleName}</div>
                                            <div className="text-xs text-[#94a3b8] mt-1">{detail}</div>
                                        </div>
                                    </div>
                                </div>
                            );
                        })
                    ) : (
                        <div className="text-sm text-[#64748b]">No critical violations detected.</div>
                    )}
                </div>
                {violations.length > 5 && (
                    <button
                        type="button"
                        onClick={() => setShowAllViolations((prev) => !prev)}
                        className="mt-3 text-xs text-[#1e296b] font-semibold hover:underline"
                    >
                        {showAllViolations
                            ? 'Hide extra violations'
                            : `View ${violations.length - 5} More Violations`}
                    </button>
                )}
            </div>

            <div className="px-6 py-5">
                <div className="bg-[#f8f9fc] rounded-2xl border border-black/5 p-5">
                    <div className="flex items-center gap-3 mb-4">
                        <div className="w-9 h-9 rounded-2xl bg-white border border-black/5 flex items-center justify-center text-xs font-bold text-[#1e296b]">
                            D3
                        </div>
                        <div>
                            <div className="text-sm font-semibold text-[#1a1f36]">D3 Report</div>
                            <div className="text-[10px] text-[#94a3b8]">AI-Powered Analysis & Remediation</div>
                        </div>
                    </div>

                    <div className="text-[10px] font-semibold uppercase tracking-wider text-[#94a3b8] mb-2">
                        Executive Summary
                    </div>
                    <div className="text-sm text-[#1a1f36] mb-4 whitespace-pre-line">{summaryText}</div>

                    <div className="text-[10px] font-semibold uppercase tracking-wider text-[#94a3b8] mb-2">
                        Root Cause
                    </div>
                    <div className="space-y-2 text-sm text-[#1a1f36] mb-4">
                        {risks.map((item, idx) => (
                            <div key={idx} className="flex gap-2">
                                <span className="text-[#94a3b8]">{idx + 1}.</span>
                                <span>{item}</span>
                            </div>
                        ))}
                    </div>

                    <div className="text-[10px] font-semibold uppercase tracking-wider text-[#94a3b8] mb-2">
                        Recommended Fixes
                    </div>
                    <div className="space-y-4">
                        {normalizedRecommendations.length ? (
                            normalizedRecommendations.map((item, idx) => (
                                <div key={idx} className="bg-white rounded-2xl border border-black/5 p-4">
                                    <div className="flex items-start gap-3">
                                        <div className="w-6 h-6 rounded-full bg-[#eef2ff] text-[#1e296b] text-xs font-bold flex items-center justify-center">
                                            {idx + 1}
                                        </div>
                                        <div className="flex-1">
                                            <div className="text-xs font-semibold uppercase tracking-wider text-[#94a3b8]">
                                                {item.rule}
                                            </div>
                                            <div className="text-sm text-[#1a1f36] mt-1">
                                                {item.recommendation}
                                            </div>
                                            {item.sql && item.sql !== 'unknown' && (
                                                <div className="mt-3 bg-[#0f172a] text-white rounded-2xl p-4 text-xs font-mono whitespace-pre-wrap">
                                                    {item.sql}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            ))
                        ) : (
                            <div className="text-sm text-[#64748b]">No recommendations available.</div>
                        )}
                    </div>

                    {nextSteps.length > 0 && (
                        <div className="mt-5">
                            <div className="text-[10px] font-semibold uppercase tracking-wider text-[#94a3b8] mb-2">
                                Priority
                            </div>
                            <div className="space-y-2 text-sm text-[#1a1f36]">
                                {nextSteps.map((item, idx) => (
                                    <div key={idx} className="flex gap-2">
                                        <span className="text-[#94a3b8]">{idx + 1}.</span>
                                        <span>{item}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

// Helper function to get node category and styling
const getNodeCategory = (tableName, datasetId) => {
    const name = `${datasetId || ''} ${tableName || ''}`.toLowerCase();
    if (name.includes('raw_') || name.includes('crm_') || name.includes('src_')) {
        return {
            category: 'SOURCE LAYER',
            icon: Database,
            bgColor: '#eef0ff',
            textColor: '#2b2f55',
            categoryColor: '#7a80b0',
            borderColor: '#d8d9ee',
            iconBg: '#ffffff',
            iconColor: '#5b5f86',
            shadow: '0 16px 30px rgba(45, 55, 90, 0.12)',
        };
    }
    if (name.includes('stg_') || name.includes('staging')) {
        return {
            category: 'STAGING',
            icon: ArrowUpDown,
            bgColor: '#eef0ff',
            textColor: '#2b2f55',
            categoryColor: '#7a80b0',
            borderColor: '#d8d9ee',
            iconBg: '#ffffff',
            iconColor: '#5b5f86',
            shadow: '0 16px 30px rgba(45, 55, 90, 0.12)',
        };
    }
    if (name.includes('int_') || name.includes('integration')) {
        return {
            category: 'INTEGRATION',
            icon: Sigma,
            bgColor: '#eef0ff',
            textColor: '#2b2f55',
            categoryColor: '#7a80b0',
            borderColor: '#d8d9ee',
            iconBg: '#ffffff',
            iconColor: '#5b5f86',
            shadow: '0 16px 30px rgba(45, 55, 90, 0.12)',
        };
    }
    if (name.includes('dwh_') || name.includes('fact_') || name.includes('dim_')) {
        return {
            category: 'CORE ENTITY',
            icon: Shield,
            bgColor: 'linear-gradient(135deg, #2d2a64 0%, #5a5a92 100%)',
            textColor: '#ffffff',
            categoryColor: 'rgba(255,255,255,0.65)',
            borderColor: 'transparent',
            iconBg: 'rgba(255,255,255,0.18)',
            iconColor: '#ffffff',
            shadow: '0 22px 40px rgba(46, 46, 96, 0.35)',
        };
    }
    if (name.includes('rpt_') || name.includes('report') || name.includes('agg_')) {
        return {
            category: 'REPORTING',
            icon: Table,
            bgColor: '#eef0ff',
            textColor: '#2b2f55',
            categoryColor: '#7a80b0',
            borderColor: '#d8d9ee',
            iconBg: '#ffffff',
            iconColor: '#5b5f86',
            shadow: '0 16px 30px rgba(45, 55, 90, 0.12)',
        };
    }
    return {
        category: 'TABLE',
        icon: Table,
        bgColor: '#eef0ff',
        textColor: '#2b2f55',
        categoryColor: '#7a80b0',
        borderColor: '#d8d9ee',
        iconBg: '#ffffff',
        iconColor: '#5b5f86',
        shadow: '0 16px 30px rgba(45, 55, 90, 0.12)',
    };
};

const LINEAGE_NODE_WIDTH = 260;
const LINEAGE_NODE_X_GAP = 300;
const LINEAGE_NODE_Y_GAP = 190;
const LINEAGE_BASE_X = 60;
const LINEAGE_BASE_Y = 40;

// Custom Node Component for styled lineage nodes
const LineageNode = ({ data }) => {
    const {
        category,
        icon: Icon,
        bgColor,
        textColor,
        borderColor,
        categoryColor,
        iconBg,
        iconColor,
        shadow,
    } = data.nodeStyle;

    return (
        <div
            style={{
                background: bgColor,
                color: textColor,
                border: `1px solid ${borderColor}`,
                borderRadius: 18,
                padding: '16px 20px',
                width: LINEAGE_NODE_WIDTH,
                maxWidth: LINEAGE_NODE_WIDTH,
                boxShadow: shadow,
                position: 'relative',
            }}
        >
            <Handle
                type="target"
                position={Position.Left}
                style={{
                    width: 10,
                    height: 10,
                    opacity: 0,
                    border: 'none',
                    background: 'transparent',
                }}
            />
            <Handle
                type="source"
                position={Position.Right}
                style={{
                    width: 10,
                    height: 10,
                    opacity: 0,
                    border: 'none',
                    background: 'transparent',
                }}
            />
            <div
                style={{
                    width: 30,
                    height: 30,
                    borderRadius: 10,
                    background: iconBg,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: '0 8px 16px rgba(15, 23, 42, 0.08)',
                }}
            >
                {Icon && <Icon style={{ width: 16, height: 16, color: iconColor }} />}
            </div>
            <div
                style={{
                    marginTop: 12,
                    fontSize: 10,
                    fontWeight: 700,
                    letterSpacing: '0.08em',
                    textTransform: 'uppercase',
                    color: categoryColor,
                }}
            >
                {category}
            </div>
            <div
                style={{
                    marginTop: 6,
                    fontSize: 12,
                    fontWeight: 600,
                    color: textColor,
                    opacity: 0.75,
                    lineHeight: 1.3,
                    whiteSpace: 'normal',
                    overflowWrap: 'anywhere',
                }}
            >
                {data.dataset}
            </div>
            <div
                style={{
                    fontSize: 15,
                    fontWeight: 700,
                    marginTop: 6,
                    lineHeight: 1.35,
                    whiteSpace: 'normal',
                    overflowWrap: 'anywhere',
                    wordBreak: 'break-word',
                }}
            >
                {data.table}
            </div>
        </div>
    );
};

const nodeTypes = { lineageNode: LineageNode };

// Lineage Graph Component - Matches screenshot design
const LineageGraph = ({ edges, focusTable, onControlsReady }) => {
    const { nodes, graphEdges } = useMemo(() => {
        if (!edges?.length) return { nodes: [], graphEdges: [] };

        const nodeMap = new Map();
        const gEdges = [];

        // Organize nodes by category for proper layer positioning
        const layers = { source: [], staging: [], integration: [], core: [], reporting: [], other: [] };

        edges.forEach((edge) => {
            const fromId = `${edge.project_id}.${edge.dataset_id}.${edge.from_table}`;
            const toId = `${edge.to_project_id}.${edge.to_dataset_id}.${edge.to_table}`;

            if (!nodeMap.has(fromId)) {
                const nodeStyle = getNodeCategory(edge.from_table, edge.dataset_id);
                nodeMap.set(fromId, {
                    id: fromId,
                    type: 'lineageNode',
                    data: {
                        label: `${edge.dataset_id}.${edge.from_table}`,
                        dataset: edge.dataset_id,
                        table: edge.from_table,
                        nodeStyle,
                    },
                    position: { x: 0, y: 0 },
                });

                // Categorize for positioning
                if (nodeStyle.category === 'SOURCE LAYER') layers.source.push(fromId);
                else if (nodeStyle.category === 'STAGING') layers.staging.push(fromId);
                else if (nodeStyle.category === 'INTEGRATION') layers.integration.push(fromId);
                else if (nodeStyle.category === 'CORE ENTITY') layers.core.push(fromId);
                else if (nodeStyle.category === 'REPORTING') layers.reporting.push(fromId);
                else layers.other.push(fromId);
            }

            if (!nodeMap.has(toId)) {
                const nodeStyle = getNodeCategory(edge.to_table, edge.to_dataset_id);
                nodeMap.set(toId, {
                    id: toId,
                    type: 'lineageNode',
                    data: {
                        label: `${edge.to_dataset_id}.${edge.to_table}`,
                        dataset: edge.to_dataset_id,
                        table: edge.to_table,
                        nodeStyle,
                    },
                    position: { x: 0, y: 0 },
                });

                if (nodeStyle.category === 'SOURCE LAYER') layers.source.push(toId);
                else if (nodeStyle.category === 'STAGING') layers.staging.push(toId);
                else if (nodeStyle.category === 'INTEGRATION') layers.integration.push(toId);
                else if (nodeStyle.category === 'CORE ENTITY') layers.core.push(toId);
                else if (nodeStyle.category === 'REPORTING') layers.reporting.push(toId);
                else layers.other.push(toId);
            }
        });

        const focusId = focusTable
            ? Array.from(nodeMap.keys()).find((id) => id.endsWith(`.${focusTable}`))
            : null;

        if (focusId) {
            const adjacency = new Map();
            const reverse = new Map();
            nodeMap.forEach((_, id) => {
                adjacency.set(id, []);
                reverse.set(id, []);
            });
            edges.forEach((edge) => {
                const fromId = `${edge.project_id}.${edge.dataset_id}.${edge.from_table}`;
                const toId = `${edge.to_project_id}.${edge.to_dataset_id}.${edge.to_table}`;
                adjacency.get(fromId)?.push(toId);
                reverse.get(toId)?.push(fromId);
            });

            const upstreamLevels = new Map([[focusId, 0]]);
            const downstreamLevels = new Map([[focusId, 0]]);
            const queueUp = [focusId];
            const queueDown = [focusId];

            while (queueUp.length) {
                const current = queueUp.shift();
                const level = upstreamLevels.get(current);
                reverse.get(current)?.forEach((prev) => {
                    if (!upstreamLevels.has(prev)) {
                        upstreamLevels.set(prev, level - 1);
                        queueUp.push(prev);
                    }
                });
            }

            while (queueDown.length) {
                const current = queueDown.shift();
                const level = downstreamLevels.get(current);
                adjacency.get(current)?.forEach((next) => {
                    if (!downstreamLevels.has(next)) {
                        downstreamLevels.set(next, level + 1);
                        queueDown.push(next);
                    }
                });
            }

            const mergedLevels = new Map();
            nodeMap.forEach((_, id) => {
                const up = upstreamLevels.get(id);
                const down = downstreamLevels.get(id);
                const level = down !== undefined ? down : up !== undefined ? up : 0;
                mergedLevels.set(id, level);
            });

            const levelBuckets = new Map();
            mergedLevels.forEach((level, id) => {
                if (!levelBuckets.has(level)) levelBuckets.set(level, []);
                levelBuckets.get(level).push(id);
            });

            levelBuckets.forEach((ids, level) => {
                ids.forEach((id, idx) => {
                    const node = nodeMap.get(id);
                    if (!node) return;
                    node.position = {
                        x: LINEAGE_BASE_X + level * LINEAGE_NODE_X_GAP,
                        y: LINEAGE_BASE_Y + idx * LINEAGE_NODE_Y_GAP,
                    };
                });
            });
        } else {
            // Position nodes in layers (left to right flow)
            const xPositions = {
                source: LINEAGE_BASE_X,
                staging: LINEAGE_BASE_X + LINEAGE_NODE_X_GAP,
                integration: LINEAGE_BASE_X + LINEAGE_NODE_X_GAP,
                core: LINEAGE_BASE_X + LINEAGE_NODE_X_GAP * 2,
                reporting: LINEAGE_BASE_X + LINEAGE_NODE_X_GAP * 2,
                other: LINEAGE_BASE_X + LINEAGE_NODE_X_GAP,
            };
            const layerCounts = { source: 0, staging: 0, integration: 0, core: 0, reporting: 0, other: 0 };

            nodeMap.forEach((node) => {
                const cat = node.data.nodeStyle.category;
                let layer = 'other';
                if (cat === 'SOURCE LAYER') layer = 'source';
                else if (cat === 'STAGING') layer = 'staging';
                else if (cat === 'INTEGRATION') layer = 'integration';
                else if (cat === 'CORE ENTITY') layer = 'core';
                else if (cat === 'REPORTING') layer = 'reporting';

                // Offset staging and integration vertically
                const yOffset = layer === 'staging' ? 0 : layer === 'integration' ? LINEAGE_NODE_Y_GAP / 2 : 0;

                node.position = {
                    x: xPositions[layer],
                    y: layerCounts[layer] * LINEAGE_NODE_Y_GAP + LINEAGE_BASE_Y + yOffset,
                };
                layerCounts[layer]++;
            });
        }

        // Create curved edges
        edges.forEach((edge, idx) => {
            const fromId = `${edge.project_id}.${edge.dataset_id}.${edge.from_table}`;
            const toId = `${edge.to_project_id}.${edge.to_dataset_id}.${edge.to_table}`;

            gEdges.push({
                id: `${fromId}-${toId}-${idx}`,
                source: fromId,
                target: toId,
                type: 'bezier',
                animated: false,
                style: {
                    stroke: '#c7cbdf',
                    strokeWidth: 2.5,
                },
                pathOptions: {
                    curvature: 0.5,
                },
                markerEnd: {
                    type: MarkerType.ArrowClosed,
                    width: 22,
                    height: 22,
                    color: '#c7cbdf',
                },
            });
        });

        return { nodes: Array.from(nodeMap.values()), graphEdges: gEdges };
    }, [edges, focusTable]);

    if (!nodes.length) return null;

    return (
        <ReactFlow
            nodes={nodes}
            edges={graphEdges}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.2 }}
            proOptions={{ hideAttribution: true }}
            nodesDraggable={false}
            nodesConnectable={false}
            elementsSelectable={false}
        >
            <LineageControls onControlsReady={onControlsReady} />
        </ReactFlow>
    );
};

// Inner component to access React Flow hooks for zoom controls
const LineageControls = ({ onControlsReady }) => {
    const { zoomIn, zoomOut, fitView } = useReactFlow();

    useEffect(() => {
        if (onControlsReady) {
            onControlsReady({ zoomIn, zoomOut, fitView });
        }
    }, [onControlsReady, zoomIn, zoomOut, fitView]);

    return null;
};

// Wrapper component with ReactFlowProvider
const LineageGraphWithProvider = ({ edges, focusTable, onControlsReady }) => {
    return (
        <ReactFlowProvider>
            <LineageGraph edges={edges} focusTable={focusTable} onControlsReady={onControlsReady} />
        </ReactFlowProvider>
    );
};


// Execution Results Modal
const ExecutionResultsModal = ({ data, onClose }) => {
    if (!data) return null;
    const rows = Array.isArray(data.rows) ? data.rows : [];
    const columns = Array.isArray(data.columns)
        ? data.columns
        : (rows[0] ? Object.keys(rows[0]) : []);

    const handleDownloadCSV = () => {
        if (!columns.length || !rows.length) return;
        const escape = (val) => {
            if (val === null || val === undefined) return '';
            const str = String(val).replace(/"/g, '""');
            return `"${str}"`;
        };
        const header = columns.map(escape).join(',');
        const csvRows = rows.map((row) => columns.map((col) => escape(row?.[col])).join(','));
        const csv = [header, ...csvRows].join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'query_results.csv';
        a.click();
        URL.revokeObjectURL(url);
    };

    return (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-8">
            <div className="bg-white rounded-3xl shadow-2xl w-full max-w-5xl max-h-[80vh] flex flex-col overflow-hidden animate-fade-in">
                {/* Header */}
                <div className="px-6 py-5 border-b border-black/5 flex items-center justify-between">
                    <div>
                        <h3 className="text-xl font-bold text-[#1a1f36]">Execution Results</h3>
                        <p className="text-sm text-[#64748b] mt-0.5">
                            {(data.row_count ?? rows.length)?.toLocaleString()} rows returned
                            {data.cached && ' (cached)'}
                        </p>
                        {data.summary && (
                            <p className="text-sm text-[#1a1f36]/80 mt-2 max-w-3xl whitespace-pre-line">
                                {data.summary}
                            </p>
                        )}
                    </div>
                    <div className="flex items-center gap-3">
                        <button
                            onClick={handleDownloadCSV}
                            className="flex items-center gap-2 px-4 py-2 bg-[#1e296b]/5 hover:bg-[#1e296b]/10 rounded-xl text-sm font-semibold text-[#1e296b] transition-colors"
                        >
                            <Download className="w-4 h-4" />
                            Export CSV
                        </button>
                        <button
                            onClick={onClose}
                            className="p-2 hover:bg-black/5 rounded-xl transition-colors"
                        >
                            <X className="w-5 h-5 text-[#64748b]" />
                        </button>
                    </div>
                </div>

                {/* Table */}
                <div className="flex-1 overflow-auto p-6">
                    {rows.length === 0 || columns.length === 0 ? (
                        <div className="text-center py-12 text-[#64748b]">No results found</div>
                    ) : (
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="border-b border-black/5">
                                    {columns.map((col) => (
                                        <th
                                            key={col}
                                            className="text-left px-4 py-3 text-xs font-bold uppercase tracking-wider text-[#64748b]"
                                        >
                                            {col}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {rows.slice(0, 100).map((row, idx) => (
                                    <tr key={idx} className="border-b border-black/5 hover:bg-black/[0.02]">
                                        {columns.map((col) => (
                                            <td key={col} className="px-4 py-3 text-[#1a1f36]">
                                                {String(row?.[col] ?? '')}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    )}
                </div>

                {/* Footer */}
                <div className="px-6 py-4 border-t border-black/5 bg-black/[0.02] flex items-center justify-between text-xs text-[#64748b]">
                    <span>Execution time: {data.execution_time_ms}ms</span>
                    <span>Bytes processed: {(data.bytes_processed / 1024 / 1024).toFixed(2)} MB</span>
                </div>
            </div>
        </div>
    );
};

// Main MetadataBotUI Component
const MetadataBotUI = () => {
    const [view, setView] = useState('chat'); // 'dashboard' | 'chat'
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isExecuting, setIsExecuting] = useState(false);
    const [executionResult, setExecutionResult] = useState(null);
    const [selectedTable, setSelectedTable] = useState(null);
    const [recentAnalysis, setRecentAnalysis] = useState(RECENT_ANALYSIS);
    const [activeAnalysis, setActiveAnalysis] = useState(null);
    const [sessionId] = useState(() => `session-${Date.now()}`);
    const [pendingLineageSelection, setPendingLineageSelection] = useState(false);
    const messagesEndRef = useRef(null);
    const inputRef = useRef(null);
    const inputContainerRef = useRef(null);
    const maxInputHeight = 88;

    const scrollToBottom = useCallback(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, []);

    useEffect(() => {
        scrollToBottom();
    }, [messages, scrollToBottom]);

    const inferToolPlan = (text) => {
        const q = (text || "").toLowerCase();
        if (q.match(/^\s*(hi|hello|hey|yo|hola|help|capabilities|what can you do|how can you help)\b/)) {
            return ["chat"];
        }
        if (q.includes("data quality") || q.includes("validate table") || q.includes("table validation") || q.includes("quality check")) {
            return ["validate_table"];
        }
        if (q.includes("lineage") || q.includes("upstream") || q.includes("downstream")) {
            return ["get_lineage"];
        }
        if (q.includes("impact") || q.includes("schema change") || q.includes("ripple")) {
            return ["ripple_report"];
        }
        if (q.includes("execute") || q.includes("run sql") || q.includes("run query") || q.match(/```sql|\\bselect\\b.+\\bfrom\\b/)) {
            return ["assess_sql", "execute_sql"];
        }
        if (
            q.includes("sql")
            || q.includes("query")
            || q.includes("trend")
            || q.includes("by date")
            || q.match(/\\b(sum|avg|average|count|min|max|median|distinct)\\b/)
        ) {
            return ["search_tables", "generate_sql", "assess_sql", "validate_query"];
        }
        if (q.match(/\\b(schema|metadata|table|tables|column|columns|field|fields)\\b/)) {
            return ["search_tables"];
        }
        return ["search_tables", "generate_sql"];
    };

    const isLikelySQL = (text) => {
        return /\\bselect\\b[\\s\\S]+\\bfrom\\b/i.test(text.trim());
    };

    const mapTableResults = (items = []) =>
        items.map((item) => ({
            table_name: item.table_name,
            column_name: null,
            dataset_id: item.dataset_id,
            full_name: `${DEFAULT_PROJECT}.${item.dataset_id}.${item.table_name}`,
            similarity: item.hybrid_score ?? item.semantic_score ?? 0,
        }));

    const mapColumnResults = (items = []) =>
        items.map((item) => ({
            table_name: item.table_name,
            column_name: item.column_name,
            dataset_id: item.dataset_id,
            full_name: `${DEFAULT_PROJECT}.${item.dataset_id}.${item.table_name}`,
            similarity: item.hybrid_score ?? item.semantic_score ?? 0,
        }));

    const mapLineageEdges = (edges = []) =>
        edges.map((edge) => ({
            project_id: DEFAULT_PROJECT,
            dataset_id: edge.source_dataset,
            from_table: edge.source_table,
            to_project_id: DEFAULT_PROJECT,
            to_dataset_id: edge.target_dataset,
            to_table: edge.target_table,
        }));

    const resolveImpactTarget = (text, entities) => {
        const triple = text.match(/([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)/);
        if (triple) {
            return { dataset_id: triple[1], table_name: triple[2], column_name: triple[3] };
        }
        const columns = Array.isArray(entities?.columns) ? entities.columns : [];
        if (entities?.dataset_id && entities?.table_name && columns.length > 0) {
            return {
                dataset_id: entities.dataset_id,
                table_name: entities.table_name,
                column_name: columns[0],
            };
        }
        const double = text.match(/([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)/);
        if (double && columns.length > 0) {
            return {
                dataset_id: entities?.dataset_id || DEFAULT_DATASET,
                table_name: double[1],
                column_name: double[2],
            };
        }
        if (entities?.table_name && columns.length > 0) {
            return {
                dataset_id: entities?.dataset_id || DEFAULT_DATASET,
                table_name: entities.table_name,
                column_name: columns[0],
            };
        }
        return null;
    };

    const resolveTableTarget = (text) => {
        const triple = text.match(/([a-zA-Z0-9_-]+)\\.([a-zA-Z0-9_]+)\\.([a-zA-Z0-9_]+)/);
        if (triple) {
            return { dataset_id: triple[2], table_name: triple[3] };
        }
        const double = text.match(/([a-zA-Z0-9_]+)\\.([a-zA-Z0-9_]+)/);
        if (double) {
            return { dataset_id: double[1], table_name: double[2] };
        }
        const named = text.match(/table\\s+([a-zA-Z0-9_]+)/i);
        if (named) {
            return { dataset_id: DEFAULT_DATASET, table_name: named[1] };
        }
        return null;
    };

    const mapRippleToImpactAssessment = (report) => {
        if (!report) return null;
        const severity = report.executive_summary?.severity_distribution || {};
        const tsunami = severity.Tsunami || 0;
        const wave = severity.Wave || 0;
        const ripple = severity.Ripple || 0;
        const impactLevel = tsunami > 0 ? 'HIGH' : wave > 0 ? 'MEDIUM' : 'LOW';
        const upstreamCount = report.upstream_analysis?.sources?.length || 0;
        const downstreamCount = report.downstream_analysis?.impacted_tables?.length || 0;
        const recs = report.recommendations || {};

        return {
            impact_level: impactLevel,
            impact_reasoning: [
                report.executive_summary?.summary,
                `Risk score: ${report.executive_summary?.risk_score ?? 'unknown'}`,
                `Upstream sources: ${upstreamCount}`,
                `Downstream tables: ${downstreamCount}`,
            ].filter(Boolean),
            downstream_tables: report.downstream_analysis?.impacted_tables || [],
            has_upstream: upstreamCount > 0,
            column: `${report.target?.dataset_id}.${report.target?.table_name}.${report.target?.column_name}`,
            table: `${report.target?.dataset_id}.${report.target?.table_name}`,
            recommendation: report.executive_summary?.recommended_action || 'review',
            recommendation_reasoning: [
                ...(recs.breaking_change_warnings || []),
                ...(recs.migration_steps || []),
            ],
            actions: [
                ...(recs.testing_checklist || []),
                ...(recs.rollback_procedures || []),
            ],
            ripple_summary: report.executive_summary,
        };
    };

    const sendMessage = async (text) => {
        if (!text.trim()) return;
        setPendingLineageSelection(false);

        const normalized = text.toLowerCase();
        const wantsTableValidation = normalized.includes("data quality")
            || normalized.includes("validate table")
            || normalized.includes("table validation")
            || normalized.includes("quality check");
        const tableTarget = wantsTableValidation ? resolveTableTarget(text) : null;

        const pendingId = `assistant-${Date.now()}`;
        const userMessage = {
            id: `user-${Date.now()}`,
            role: 'user',
            content: text,
        };
        const pendingAssistant = {
            id: pendingId,
            role: 'assistant',
            content: 'Working...',
            pending: true,
            toolTraceText: inferToolPlan(text),
        };
        setMessages((prev) => [...prev, userMessage, pendingAssistant]);
        setIsLoading(true);
        setView('chat');

        try {
            let assistantMessage = {
                id: pendingId,
                role: 'assistant',
                content: 'Done.',
                pending: false,
                toolCalls: [],
                intent: null,
                needsSelection: false,
                sql: null,
                confidence: null,
                tablesUsed: [],
                searchResults: [],
                lineageEdges: [],
                lineageTable: null,
                executionResult: null,
                impactAssessment: null,
                queryValidation: null,
                tableValidation: null,
            };
            if (tableTarget) {
                const tableResp = await axios.post('/api/validate_table', {
                    dataset_id: tableTarget.dataset_id,
                    table_name: tableTarget.table_name,
                    layer: tableTarget.dataset_id?.includes("gold")
                        ? "gold"
                        : tableTarget.dataset_id?.includes("bronze")
                            ? "bronze"
                            : "silver",
                    include_llm_summary: true,
                });
                const data = tableResp.data || {};
                assistantMessage.content = data.validation_result?.table
                    ? `Data quality results for ${data.validation_result.table}.`
                    : 'Data quality results ready.';
                assistantMessage.tableValidation = data;
            } else {
                const chatResp = await axios.post('/api/chat', { query: text, session_id: sessionId });
                const data = chatResp.data || {};
                assistantMessage.content = data.reply || 'Done.';
                assistantMessage.confidence = data.confidence ?? null;
                assistantMessage.intent = data.intent || null;
                assistantMessage.searchResults = data.search_results || [];
                assistantMessage.lineageEdges = data.lineage?.edges || [];
                assistantMessage.lineageTable = data.lineage?.table || null;
                assistantMessage.sql = data.sql || null;
                assistantMessage.tablesUsed = data.tables_used || [];
                assistantMessage.queryValidation = data.query_validation || null;
                assistantMessage.tableValidation = data.table_validation || null;
                assistantMessage.executionResult = data.execution_result || null;
                assistantMessage.impactAssessment = data.impact_assessment || null;
                assistantMessage.needsSelection = Boolean(data.needs_selection);
                if (assistantMessage.needsSelection) {
                    setPendingLineageSelection(true);
                } else {
                    setPendingLineageSelection(false);
                }
                if (assistantMessage.executionResult && !assistantMessage.executionResult.columns) {
                    const rows = assistantMessage.executionResult.rows || [];
                    assistantMessage.executionResult = {
                        ...assistantMessage.executionResult,
                        columns: rows.length ? Object.keys(rows[0]) : [],
                    };
                }
            }

            setMessages((prev) => prev.map((m) => (m.id === pendingId ? assistantMessage : m)));

            // Update recent analysis
            const newAnalysis = {
                id: `r-${Date.now()}`,
                icon: assistantMessage.lineageEdges?.length ? GitBranch : assistantMessage.searchResults?.length ? Search : Table,
                title: text.slice(0, 25) + (text.length > 25 ? '...' : ''),
                active: false,
            };
            setRecentAnalysis((prev) => [newAnalysis, ...prev.slice(0, 4)]);
        } catch (error) {
            console.error('Chat error:', error);
            const errorMessage = {
                id: pendingId,
                role: 'assistant',
                content: `Sorry, I encountered an error: ${error.response?.data?.detail || error.message}. Please try again.`,
                pending: false,
            };
            setMessages((prev) => prev.map((m) => (m.id === pendingId ? errorMessage : m)));
        } finally {
            setIsLoading(false);
        }
    };

    // Handle table card click - shows schema inline in chat
    const handleTableClick = async (table) => {
        setIsLoading(true);

        try {
            if (pendingLineageSelection) {
                const response = await axios.post('/api/get_lineage', {
                    table_name: table.table_name,
                    dataset_id: table.dataset_id,
                });
                const data = response.data || {};
                const lineageMessage = {
                    id: `lineage-${Date.now()}`,
                    role: 'assistant',
                    content: `Lineage for ${table.table_name}.`,
                    lineageEdges: mapLineageEdges(data.edges || []),
                    lineageTable: table.table_name,
                };
                setPendingLineageSelection(false);
                setMessages((prev) => [...prev, lineageMessage]);
                return;
            }

            const response = await axios.post('/api/get_table_schema', {
                dataset_id: table.dataset_id || DEFAULT_DATASET,
                table_name: table.table_name,
            });
            const schema = response.data || null;
            const schemaMessage = {
                id: `schema-${Date.now()}`,
                role: 'assistant',
                content: `Schema for **${table.table_name}**.`,
                schemaData: {
                    table,
                    schema,
                },
            };

            setMessages((prev) => [...prev, schemaMessage]);
        } catch (error) {
            console.error('Table click error:', error);
            const errorMessage = {
                id: `error-${Date.now()}`,
                role: 'assistant',
                content: `Sorry, I couldn't retrieve details for ${table.table_name}: ${error.response?.data?.detail || error.message}`,
            };
            setMessages((prev) => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        if (input.trim() && !isLoading) {
            sendMessage(input);
            setInput('');
            if (inputRef.current) {
                inputRef.current.style.height = 'auto';
                inputRef.current.style.overflowY = 'hidden';
            }
        }
    };

    const handleInputChange = (e) => {
        setInput(e.target.value);
        if (!inputRef.current) return;
        inputRef.current.style.height = 'auto';
        const nextHeight = Math.min(inputRef.current.scrollHeight, maxInputHeight);
        inputRef.current.style.height = `${nextHeight}px`;
        inputRef.current.style.overflowY = inputRef.current.scrollHeight > maxInputHeight ? 'auto' : 'hidden';
    };

    const handleInputKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    };

    const handleQuickAction = (action) => {
        sendMessage(action.query);
    };

    const handleExecuteSQL = async (sql, messageId) => {
        setIsExecuting(true);
        try {
            const response = await axios.post('/api/execute_sql', {
                sql: sql,
                max_rows: 100,
            });
            // Store execution result in the specific message for inline display
            setMessages((prev) =>
                prev.map((msg) =>
                    msg.id === messageId
                        ? { ...msg, executionResult: response.data }
                        : msg
                )
            );
        } catch (error) {
            console.error('Execution error:', error);
            // Store error in the message
            setMessages((prev) =>
                prev.map((msg) =>
                    msg.id === messageId
                        ? { ...msg, executionError: error.response?.data?.detail || error.message }
                        : msg
                )
            );
        } finally {
            setIsExecuting(false);
        }
    };

    const handleNewAnalysis = () => {
        setMessages([]);
        setView('dashboard');
        setActiveAnalysis(null);
    };

    const handleSelectAnalysis = (item) => {
        setActiveAnalysis(item.id);
        // Could load historical analysis here
    };

    const handleSuggestion = (query) => {
        if (query) {
            sendMessage(query);
        }
    };

    return (
        <div className="min-h-screen flex bg-background">
            {/* Sidebar */}
            <Sidebar
                recentAnalysis={recentAnalysis}
                onNewAnalysis={handleNewAnalysis}
                activeAnalysis={activeAnalysis}
                onSelectAnalysis={handleSelectAnalysis}
            />

            {/* Main Content */}
            <main className="flex-1 flex flex-col h-screen overflow-hidden">
                {/* Header */}
                <header className="px-8 py-5 flex items-center justify-between border-b border-black/5 bg-white">
                    <div className="flex items-center gap-4">
                        {view === 'chat' && (
                            <button
                                onClick={() => setView('dashboard')}
                                className="p-2 hover:bg-black/5 rounded-xl transition-colors mr-2"
                            >
                                <ArrowLeft className="w-5 h-5 text-[#1a1f36]/60" />
                            </button>
                        )}
                        <div className="flex items-center gap-3 bg-white px-4 py-2 rounded-full shadow-sm border border-black/5">
                            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse-soft" />
                            <span className="text-xs font-semibold uppercase tracking-wider text-[#1a1f36]/50">
                                Data Status: Connected
                            </span>
                            <div className="w-px h-4 bg-black/10" />
                            <span className="text-xs font-semibold uppercase tracking-wider text-[#1a1f36]/50">
                                BigQuery Prod
                            </span>
                        </div>
                    </div>
                    <button className="w-11 h-11 bg-white rounded-full flex items-center justify-center shadow-sm border border-black/5 hover:bg-black/[0.02] transition-colors">
                        <Bell className="w-5 h-5 text-[#1a1f36]/50" />
                    </button>
                </header>

                {/* Dashboard View */}
                {view === 'dashboard' && (
                    <div className="flex-1 overflow-y-auto custom-scrollbar">
                        <div className="max-w-5xl mx-auto px-12 py-16">
                            {/* Greeting */}
                            <h2 className="text-5xl font-black text-[#1a1f36] mb-3">
                                {getGreeting()}, {USER_NAME}.
                            </h2>
                            <p className="text-xl text-[#64748b] font-medium mb-16">
                                How can I help with your BigQuery metadata today?
                            </p>

                            {/* Quick Actions Grid */}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-20">
                                {QUICK_ACTIONS.map((action) => (
                                    <ActionCard
                                        key={action.id}
                                        icon={action.icon}
                                        title={action.title}
                                        action={action.action}
                                        type={action.type}
                                        onClick={() => handleQuickAction(action)}
                                    />
                                ))}
                            </div>
                        </div>
                    </div>
                )}

                {/* Chat View */}
                {view === 'chat' && (
                    <div className="flex-1 overflow-y-auto custom-scrollbar">
                        <div className="px-8 py-10">
                            <div className="flex flex-col gap-8">
                                {messages.map((msg) => (
                                    <ChatMessage
                                        key={msg.id}
                                        message={msg}
                                        onExecuteSQL={handleExecuteSQL}
                                        isExecuting={isExecuting}
                                        onTableClick={handleTableClick}
                                        onSuggestion={handleSuggestion}
                                    />
                                ))}
                                <div ref={messagesEndRef} className="h-4" />
                            </div>
                        </div>
                    </div>
                )}

                {/* Input Bar */}
                <div className="px-8 pb-8 pt-4 bg-background">
                    <div className="w-full">
                        <form
                            onSubmit={handleSubmit}
                            className="bg-white rounded-[2rem] p-3 flex items-center gap-3 shadow-elevated border border-black/5 transition-all focus-within:border-[#1e296b]/20 focus-within:shadow-[0_0_0_4px_rgba(30,41,107,0.05)]"
                        >
                            <button
                                type="button"
                                className="p-3 hover:bg-black/5 rounded-full transition-colors flex-shrink-0"
                            >
                                <Paperclip className="w-5 h-5 text-[#1a1f36]/30 rotate-45" />
                            </button>
                            <textarea
                                ref={inputRef}
                                rows={1}
                                value={input}
                                onChange={handleInputChange}
                                onKeyDown={handleInputKeyDown}
                                placeholder="Ask anything about your BigQuery metadata..."
                                className="flex-1 input-text text-[#1a1f36] placeholder:text-[#1a1f36]/30 bg-transparent outline-none py-2 min-w-0 resize-none leading-6 max-h-[88px] overflow-y-hidden"
                                disabled={isLoading}
                            />
                            <button
                                type="button"
                                className="p-3 hover:bg-black/5 rounded-full transition-colors flex-shrink-0"
                            >
                                <Mic className="w-5 h-5 text-[#1a1f36]/30" />
                            </button>
                            <button
                                type="submit"
                                disabled={!input.trim() || isLoading}
                                className="w-12 h-12 bg-[#1e296b] hover:bg-[#161d4d] disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-xl flex items-center justify-center shadow-lg transition-all active:scale-95 flex-shrink-0"
                            >
                                <ArrowUp className="w-6 h-6" />
                            </button>
                        </form>
                    </div>
                </div>
            </main>

            {/* Execution Results Modal */}
            {executionResult && (
                <ExecutionResultsModal
                    data={executionResult}
                    onClose={() => setExecutionResult(null)}
                />
            )}
        </div>
    );
};

export default MetadataBotUI;
