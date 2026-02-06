import React from 'react';
import { Shield, Search, Share2, Lock, Bell, Mic, Paperclip, ArrowUp } from 'lucide-react';

const ActionCard = ({ title, linkText, icon: Icon, type }) => {
    const styles = {
        yellow: "bg-card-yellow",
        blue: "bg-card-blue"
    };

    const iconColors = {
        yellow: "text-[#b45309] bg-[#fef3c7]",
        blue: "text-[#3b82f6] bg-[#dbeafe]"
    };

    return (
        <div className={`dashboard-card ${styles[type]}`}>
            <div className={`w-12 h-12 rounded-2xl flex items-center justify-center ${iconColors[type]}`}>
                <Icon className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold leading-snug flex-1 mt-2">{title}</h3>
            <div className="flex items-center gap-2 font-bold text-sm opacity-60 hover:opacity-100 transition-opacity">
                {linkText} <span className="text-lg">→</span>
            </div>
        </div>
    );
};

const Dashboard = ({ onSearch }) => {
    const [input, setInput] = React.useState('');

    const handleSearch = (e) => {
        e.preventDefault();
        if (input.trim()) {
            onSearch(input);
        }
    };

    const handleQuickAction = (text) => {
        onSearch(text);
    };

    return (
        <div className="flex-1 flex flex-col h-full bg-background overflow-hidden relative">
            {/* Header */}
            <header className="p-8 flex items-center justify-between">
                <div className="flex items-center gap-4 bg-white px-5 py-2.5 rounded-full shadow-sm border border-black/5">
                    <div className="w-2 h-2 rounded-full bg-sidebar animate-pulse" />
                    <span className="text-xs font-bold uppercase tracking-wider opacity-60 border-r border-black/10 pr-4">Data Status: Connected</span>
                    <span className="text-xs font-bold uppercase tracking-wider opacity-60 pl-2">BigQuery Prod</span>
                </div>
                <button className="w-12 h-12 bg-white rounded-full flex items-center justify-center shadow-sm border border-black/5 hover:bg-gray-50 transition-colors">
                    <Bell className="w-5 h-5 opacity-60" />
                </button>
            </header>

            {/* Main Content Scrollable Area */}
            <div className="flex-1 overflow-y-auto px-16 py-12 custom-scrollbar">
                <div className="max-w-5xl">
                    <h2 className="text-6xl font-black text-[#1a1f36] mb-4">Good morning, Alex.</h2>
                    <p className="text-2xl text-gray-400 font-medium mb-16">How can I help with your BigQuery metadata today?</p>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-24">
                        <div onClick={() => handleQuickAction("Analyze PII risk in the marketing dataset")}>
                            <ActionCard
                                type="yellow"
                                title="Analyze PII risk in the marketing dataset"
                                linkText="Run audit"
                                icon={Shield}
                            />
                        </div>
                        <div onClick={() => handleQuickAction("Find all tables containing email_address")}>
                            <ActionCard
                                type="blue"
                                title="Find all tables containing email_address"
                                linkText="Search metadata"
                                icon={Search}
                            />
                        </div>
                        <div onClick={() => handleQuickAction("Visualize lineage for the revenue_summary table")}>
                            <ActionCard
                                type="blue"
                                title="Visualize lineage for the revenue_summary table"
                                linkText="Generate graph"
                                icon={Share2}
                            />
                        </div>
                        <div onClick={() => handleQuickAction("Audit row-level security for HR schemas")}>
                            <ActionCard
                                type="yellow"
                                title="Audit row-level security for HR schemas"
                                linkText="Check permissions"
                                icon={Lock}
                            />
                        </div>
                    </div>
                </div>
            </div>

            {/* Floating Input Bar */}
            <div className="px-16 pb-12">
                <div className="max-w-5xl mx-auto relative">
                    <form onSubmit={handleSearch} className="bg-white rounded-[2.5rem] p-4 flex items-center gap-4 shadow-2xl border border-black/5">
                        <button type="button" className="p-4 hover:bg-gray-100 rounded-full transition-colors">
                            <Paperclip className="w-6 h-6 opacity-40 rotate-45" />
                        </button>
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Ask anything about your BigQuery metadata..."
                            className="flex-1 input-text placeholder:text-gray-300 bg-transparent outline-none py-2"
                        />
                        <button type="button" className="p-4 hover:bg-gray-100 rounded-full transition-colors mr-2">
                            <Mic className="w-6 h-6 opacity-40" />
                        </button>
                        <button
                            type="submit"
                            className="w-14 h-14 bg-[#1e296b] hover:bg-[#1a235c] text-white rounded-2xl flex items-center justify-center shadow-lg transition-transform active:scale-95"
                        >
                            <ArrowUp className="w-8 h-8" />
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
