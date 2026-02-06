import React, { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, Code, Play, CheckCircle2, Loader2, ArrowLeft, Paperclip, Mic, ArrowUp } from 'lucide-react';
import axios from 'axios';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import ResultsTable from './ResultsTable';

const QueryValidationCard = ({ validation }) => {
    if (!validation) return null;
    const recommendation = String(validation.recommendation || "UNKNOWN").toUpperCase();
    const tone = {
        APPROVE: 'bg-emerald-50 text-emerald-600 border-emerald-200',
        WARN: 'bg-amber-50 text-amber-600 border-amber-200',
        'NEEDS REVIEW': 'bg-amber-50 text-amber-600 border-amber-200',
        REJECT: 'bg-red-50 text-red-600 border-red-200',
        UNKNOWN: 'bg-slate-50 text-slate-600 border-slate-200',
    }[recommendation] || 'bg-slate-50 text-slate-600 border-slate-200';
    const cost = validation.cost || {};
    const issues = Array.isArray(validation.issues) ? validation.issues : [];
    const suggestions = Array.isArray(validation.suggestions) ? validation.suggestions : [];

    return (
        <div className="w-full bg-white border border-black/10 rounded-[1.75rem] p-6 shadow-soft">
            <div className="flex items-center justify-between mb-3">
                <div>
                    <div className="text-sm font-bold text-[#1a1f36]">Query Validation</div>
                    <div className="text-xs text-[#94a3b8]">Cost + efficiency check</div>
                </div>
                <span className={`px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider border ${tone}`}>
                    {recommendation}
                </span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div className="bg-[#f8fafc] rounded-xl border border-black/5 p-3">
                    <div className="text-[10px] font-semibold uppercase tracking-wider text-[#94a3b8]">Estimated Cost</div>
                    <div className="text-sm font-semibold text-[#1a1f36]">
                        {typeof cost.estimated_cost_usd === 'number' ? `$${cost.estimated_cost_usd.toFixed(4)}` : 'unknown'}
                    </div>
                </div>
                <div className="bg-[#f8fafc] rounded-xl border border-black/5 p-3">
                    <div className="text-[10px] font-semibold uppercase tracking-wider text-[#94a3b8]">Bytes</div>
                    <div className="text-sm font-semibold text-[#1a1f36]">
                        {typeof cost.bytes_processed_gb === 'number' ? `${cost.bytes_processed_gb.toFixed(2)} GB` : 'unknown'}
                    </div>
                </div>
                <div className="bg-[#f8fafc] rounded-xl border border-black/5 p-3">
                    <div className="text-[10px] font-semibold uppercase tracking-wider text-[#94a3b8]">Issues</div>
                    <div className="text-sm font-semibold text-[#1a1f36]">{issues.length}</div>
                </div>
            </div>
            {(issues.length > 0 || suggestions.length > 0) && (
                <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div className="bg-white rounded-xl border border-black/5 p-4">
                        <div className="text-[10px] font-semibold uppercase tracking-wider text-[#94a3b8] mb-2">Issues</div>
                        {issues.length ? (
                            <div className="space-y-2 text-sm text-[#1a1f36]">
                                {issues.slice(0, 3).map((item, idx) => (
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
                                {suggestions.slice(0, 3).map((item, idx) => (
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
            )}
        </div>
    );
};

const ChatInterface = ({ initialQuestion, onBack }) => {
    const defaultProjectId = 'project-6ab0b570-446d-448e-882';
    const billingProjectId = 'project-6ab0b570-446d-448e-882';
    const defaultDatasetId = 'gold_layer';
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [executionResult, setExecutionResult] = useState(null);
    const [isExecuting, setIsExecuting] = useState(false);
    const messagesEndRef = useRef(null);
    const hasPerformedInitialQuery = useRef(false);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    // Handle initial question from dashboard ONLY ONCE
    useEffect(() => {
        if (initialQuestion && !hasPerformedInitialQuery.current) {
            hasPerformedInitialQuery.current = true;
            performQuery(initialQuestion);
        }
    }, [initialQuestion]);

    const performQuery = async (question) => {
        const userMessage = {
            id: "u-" + Date.now().toString(),
            role: 'user',
            content: question,
            type: 'text'
        };

        setMessages(prev => [...prev, userMessage]);
        setIsLoading(true);

        try {
            const tableResp = await axios.post('/api/get_matching_tables', {
                query: question,
                top_k: 5,
            });
            const topTables = (tableResp.data || []).slice(0, 3).map((item) => ({
                dataset_id: item.dataset_id,
                table_name: item.table_name,
            }));
            const response = await axios.post('/api/generate_sql', {
                user_query: question,
                tables: topTables,
            });

            const data = response.data || {};
            const tablesUsed = data.tables_used || topTables.map((t) => `${t.dataset_id}.${t.table_name}`);
            let queryValidation = data.query_validation || null;
            if (!queryValidation && data.sql) {
                try {
                    const validationResp = await axios.post('/api/validate_query', {
                        sql: data.sql,
                    });
                    queryValidation = validationResp.data;
                } catch (validationError) {
                    console.error(validationError);
                }
            }
            let content = data.sql ? 'Generated SQL.' : 'Unable to generate SQL.';
            if (queryValidation && Array.isArray(queryValidation.issues) && queryValidation.issues.length > 0) {
                const issuePreview = queryValidation.issues.slice(0, 2).join('; ');
                content += `\nQuery check: ${queryValidation.recommendation}. ${issuePreview}`;
            }

            const aiMessage = {
                id: "a-" + (Date.now() + 1).toString(),
                role: 'assistant',
                content,
                sql: data.sql,
                confidence: data.sql ? 0.8 : null,
                tables_used: tablesUsed,
                queryValidation,
                type: 'response'
            };

            setMessages(prev => [...prev, aiMessage]);
        } catch (error) {
            console.error(error);
            const errorMessage = {
                id: "e-" + (Date.now() + 1).toString(),
                role: 'assistant',
                content: `Analysis failed: ${error.response?.data?.detail || error.message}`,
                type: 'error'
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;
        const q = input;
        setInput('');
        performQuery(q);
    };

    const handleExecute = async (sql, options = {}) => {
        const { autoFix = false } = options;
        setIsExecuting(true);
        try {
            const response = await axios.post('/api/execute_sql', {
                sql: sql,
                max_rows: 100,
            });
            setExecutionResult(response.data);
        } catch (error) {
            console.error(error);
            const data = error.response?.data;
            if (data?.approved === false) {
                const validationMessage = {
                    id: "v-" + (Date.now() + 1).toString(),
                    role: 'assistant',
                    type: 'validation',
                    content: 'SQL needs improvements before execution.',
                    issues: data.issues || [],
                    suggestedSql: data.suggested_sql || null,
                    originalSql: sql
                };
                setMessages(prev => [...prev, validationMessage]);
            } else {
                alert(`Execution failed: ${data?.detail || error.message}`);
            }
        } finally {
            setIsExecuting(false);
        }
    };

    return (
        <div className="flex-1 flex flex-col h-full bg-[#f8f9fc] relative overflow-hidden">
            {executionResult && (
                <ResultsTable
                    data={executionResult}
                    onClose={() => setExecutionResult(null)}
                />
            )}

            {/* Header - Fixed contrast & alignment */}
            <header className="px-8 py-6 border-b border-black/5 bg-white flex items-center justify-between z-20">
                <div className="flex items-center gap-4">
                    <button
                        onClick={onBack}
                        className="p-3 hover:bg-gray-100 rounded-2xl transition-all active:scale-95 group"
                    >
                        <ArrowLeft className="w-6 h-6 text-[#1a1f36] opacity-60 group-hover:opacity-100" />
                    </button>
                    <div className="flex flex-col">
                        <h2 className="font-extrabold text-[#1a1f36] text-xl">Analysis Session</h2>
                        <p className="text-xs font-bold text-[#1a1f36]/40 uppercase tracking-widest">Dataset: thelook_ecommerce</p>
                    </div>
                </div>
                <div className="flex items-center gap-3">
                    <div className="px-4 py-1.5 bg-[#1e296b]/5 text-[#1e296b] rounded-full text-[10px] font-black uppercase tracking-widest border border-[#1e296b]/10">
                        Prod Connected
                    </div>
                </div>
            </header>

            {/* Messages Area - Improved spacing and centering */}
            <div className="flex-1 overflow-y-auto custom-scrollbar bg-background">
                <div className="max-w-5xl mx-auto px-6 py-12 md:px-12">
                    <div className="flex flex-col gap-10">
                        {messages.map((msg) => (
                            <div
                                key={msg.id}
                                className={`flex w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                            >
                                <div className={`flex gap-5 max-w-[90%] md:max-w-[85%] ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>

                                    {/* Avatar */}
                                    <div className={`w-11 h-11 rounded-2xl flex items-center justify-center flex-shrink-0 shadow-soft mt-1 ${msg.role === 'user'
                                            ? 'bg-[#1e296b] text-white'
                                            : 'bg-white text-[#1e296b] border border-black/5'
                                        }`}>
                                        {msg.role === 'user' ? <div className="w-2.5 h-2.5 rounded-full bg-white" /> : <Sparkles className="w-6 h-6 text-[#3b82f6]" />}
                                    </div>

                                    {/* Content Bubble container */}
                                    <div className={`flex flex-col gap-4 flex-1 ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                                        {/* Text Content */}
                                        <div className={`p-6 md:p-8 rounded-[2rem] shadow-soft ${msg.role === 'user'
                                                ? 'bg-[#1e296b] text-white rounded-tr-none'
                                                : 'bg-white border border-black/5 text-[#1a1f36] rounded-tl-none font-medium'
                                            }`}>
                                            <p className="text-lg leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                                        </div>

                                        {/* SQL Result Card (Assistant Only) */}
                                        {msg.type === 'response' && msg.sql && (
                                            <div className="bg-[#1a1f36] rounded-[2.5rem] overflow-hidden shadow-2xl animate-fade-in w-full border border-white/5">
                                                {/* Header */}
                                                <div className="px-6 md:px-8 py-5 bg-white/5 border-b border-white/5 flex flex-col sm:flex-row sm:items-center justify-between gap-5">
                                                    <div className="flex items-center gap-4">
                                                        <div className="w-10 h-10 rounded-xl bg-white/10 flex items-center justify-center">
                                                            <Code className="w-5 h-5 text-white/80" />
                                                        </div>
                                                        <div className="flex flex-col">
                                                            <span className="text-sm font-bold text-white/90">BigQuery SQL</span>
                                                            <span className="text-[10px] uppercase font-bold text-white/30 tracking-widest">Generated via Gemini</span>
                                                        </div>
                                                    </div>
                                                    <div className="flex items-center gap-4">
                                                        {msg.confidence && (
                                                            <div className="px-4 py-2 rounded-xl bg-green-500/10 border border-green-500/20 flex items-center gap-2">
                                                                <CheckCircle2 className="w-4 h-4 text-green-400" />
                                                                <span className="text-xs font-bold text-green-400">
                                                                    {(msg.confidence * 100).toFixed(0)}% Match
                                                                </span>
                                                            </div>
                                                        )}
                                                        <button
                                                            onClick={() => handleExecute(msg.sql)}
                                                            disabled={isExecuting}
                                                            className="flex items-center gap-2 px-6 py-3 bg-white text-[#1a1f36] hover:bg-white/90 rounded-2xl text-sm font-black shadow-xl transition-all active:scale-95 disabled:opacity-50"
                                                        >
                                                            {isExecuting ? <Loader2 className="w-5 h-5 animate-spin" /> : <Play className="w-5 h-5 fill-current" />}
                                                            {isExecuting ? 'Running' : 'Run Query'}
                                                        </button>
                                                    </div>
                                                </div>

                                                {/* SQL Code */}
                                                <div className="overflow-x-auto">
                                                    <SyntaxHighlighter
                                                        language="sql"
                                                        style={vscDarkPlus}
                                                        customStyle={{
                                                            margin: 0,
                                                            borderRadius: 0,
                                                            padding: '2rem md:2.5rem',
                                                            background: 'transparent',
                                                            fontFamily: '"Söhne Mono", "SF Mono", Menlo, Monaco, Consolas, "Liberation Mono", monospace',
                                                            fontSize: '0.875rem',
                                                            lineHeight: '1.7',
                                                        }}
                                                        wrapLines={true}
                                                    >
                                                        {msg.sql}
                                                    </SyntaxHighlighter>
                                                </div>

                                                {/* Footer / Context */}
                                                {msg.tables_used?.length > 0 && (
                                                    <div className="px-8 py-5 bg-black/20 border-t border-white/5 flex items-center gap-5 overflow-x-auto no-scrollbar">
                                                        <span className="text-[10px] font-bold text-white/30 uppercase tracking-widest whitespace-nowrap">Schema Access</span>
                                                        <div className="flex gap-2">
                                                            {msg.tables_used.map(t => (
                                                                <span key={t} className="px-4 py-1.5 bg-white/5 border border-white/5 rounded-full text-[10px] font-bold text-white/60 tracking-wider whitespace-nowrap">{t}</span>
                                                            ))}
                                                        </div>
                                                    </div>
                                                )}
                                            </div>
                                        )}

                                        {msg.queryValidation && (
                                            <QueryValidationCard validation={msg.queryValidation} />
                                        )}

                                        {msg.type === 'validation' && (
                                            <div className="w-full bg-white border border-red-200 rounded-[2rem] p-6 shadow-soft">
                                                <div className="text-sm font-bold text-red-600 mb-3">Validation Required</div>
                                                <ul className="space-y-2 text-sm text-[#1a1f36]">
                                                    {(msg.issues || []).map((issue, idx) => (
                                                        <li key={`${msg.id}-issue-${idx}`} className="leading-relaxed">
                                                            • {issue.message}
                                                            {issue.suggestion && (
                                                                <div className="text-xs text-[#1a1f36]/60 mt-1">
                                                                    Suggestion: {issue.suggestion}
                                                                </div>
                                                            )}
                                                        </li>
                                                    ))}
                                                </ul>

                                                {msg.suggestedSql && (
                                                    <div className="mt-4 rounded-xl overflow-hidden border border-black/10">
                                                        <div className="px-4 py-2 bg-black/5 text-xs font-bold uppercase tracking-widest text-[#1a1f36]/60">
                                                            Suggested SQL
                                                        </div>
                                                        <SyntaxHighlighter
                                                            language="sql"
                                                            style={vscDarkPlus}
                                                            customStyle={{
                                                                margin: 0,
                                                                borderRadius: 0,
                                                                padding: '1.25rem',
                                                                background: '#111827',
                                                                fontFamily: '"Söhne Mono", "SF Mono", Menlo, Monaco, Consolas, "Liberation Mono", monospace',
                                                                fontSize: '0.875rem',
                                                                lineHeight: '1.6',
                                                            }}
                                                            wrapLines={true}
                                                        >
                                                            {msg.suggestedSql}
                                                        </SyntaxHighlighter>
                                                    </div>
                                                )}

                                                <div className="mt-4 flex flex-wrap gap-3">
                                                    {!msg.suggestedSql && (
                                                        <button
                                                            onClick={() => handleExecute(msg.originalSql, { autoFix: true })}
                                                            disabled={isExecuting}
                                                            className="px-4 py-2 rounded-xl bg-[#1e296b] text-white text-sm font-bold shadow-soft disabled:opacity-50"
                                                        >
                                                            {isExecuting ? 'Working...' : 'Auto-fix with Validator'}
                                                        </button>
                                                    )}
                                                    {msg.suggestedSql && (
                                                        <button
                                                            onClick={() => handleExecute(msg.suggestedSql)}
                                                            disabled={isExecuting}
                                                            className="px-4 py-2 rounded-xl bg-green-600 text-white text-sm font-bold shadow-soft disabled:opacity-50"
                                                        >
                                                            {isExecuting ? 'Running...' : 'Run Suggested Query'}
                                                        </button>
                                                    )}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        ))}
                        {isLoading && (
                            <div className="flex justify-start w-full">
                                <div className="flex gap-5">
                                    <div className="w-11 h-11 rounded-2xl bg-white border border-black/5 flex items-center justify-center flex-shrink-0 shadow-soft">
                                        <Loader2 className="w-6 h-6 text-[#3b82f6] animate-spin" />
                                    </div>
                                    <div className="bg-white border border-black/5 px-8 py-6 rounded-3xl rounded-tl-none shadow-soft flex items-center gap-4">
                                        <div className="w-2 h-2 rounded-full bg-[#3b82f6] animate-bounce" />
                                        <span className="text-lg text-gray-400 font-bold tracking-tight italic">Analyzing metadata connections...</span>
                                    </div>
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} className="h-4" />
                    </div>
                </div>
            </div>

            {/* Input Area - Redesigned for production feel */}
            <div className="px-8 pb-12 pt-6 bg-background relative z-10">
                <div className="max-w-4xl mx-auto">
                    <div className="relative group">
                        <div className="absolute inset-0 bg-[#1e296b]/5 blur-3xl rounded-[3rem] -z-10 opacity-0 group-focus-within:opacity-100 transition-opacity" />
                        <form onSubmit={handleSubmit} className="bg-white rounded-[2.5rem] p-4 flex items-center gap-4 shadow-2xl border border-black/10 transition-all focus-within:border-[#1e296b]/30">
                            <button type="button" className="p-4 hover:bg-gray-50 rounded-full transition-colors flex-shrink-0">
                                <Paperclip className="w-7 h-7 text-[#1a1f36]/30 rotate-45" />
                            </button>
                            <input
                                type="text"
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                placeholder="Ask a follow-up or a new data question..."
                                className="flex-1 input-text text-[#1a1f36] placeholder:text-gray-300 bg-transparent outline-none py-2 min-w-0"
                                disabled={isLoading}
                            />
                            <button type="button" className="hidden sm:block p-4 hover:bg-gray-50 rounded-full transition-colors flex-shrink-0">
                                <Mic className="w-7 h-7 text-[#1a1f36]/30" />
                            </button>
                            <button
                                type="submit"
                                disabled={!input.trim() || isLoading}
                                className="w-16 h-16 bg-[#1e296b] hover:bg-[#1a235c] disabled:opacity-50 text-white rounded-2xl flex items-center justify-center shadow-xl transition-all active:scale-95 flex-shrink-0"
                            >
                                <ArrowUp className="w-9 h-9" />
                            </button>
                        </form>
                    </div>
                    <div className="mt-4 flex justify-center gap-8 text-[11px] font-black uppercase tracking-[0.2em] text-[#1a1f36]/20">
                        <span className="flex items-center gap-2">Enterprise Ready</span>
                        <span className="flex items-center gap-2">Data Privacy Secure</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ChatInterface;
