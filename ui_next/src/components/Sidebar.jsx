import React from 'react';
import { Database, Plus, Search, Share2, Settings, User } from 'lucide-react';

const Sidebar = ({ onNewAnalysis }) => {
    return (
        <aside className="w-80 bg-[#1e296b] flex flex-col h-full text-white shadow-2xl z-30">
            {/* Brand */}
            <div className="p-10 flex items-center gap-5">
                <div className="w-11 h-11 bg-white/15 rounded-2xl flex items-center justify-center border border-white/10 shadow-lg backdrop-blur-sm">
                    <Database className="w-6 h-6 text-white" />
                </div>
                <div>
                    <h1 className="text-2xl font-black tracking-tight leading-none text-white">Metadata Bot</h1>
                    <p className="text-[10px] tracking-[0.2em] uppercase font-black opacity-40 mt-1">BigQuery Explorer</p>
                </div>
            </div>

            {/* New Analysis Button */}
            <div className="px-8 mb-10">
                <button
                    onClick={onNewAnalysis}
                    className="w-full bg-white text-[#1e296b] hover:bg-white/95 font-black py-5 px-6 rounded-[1.5rem] flex items-center justify-between shadow-2xl transition-all active:scale-[0.97] group"
                >
                    <span className="text-lg">New Analysis</span>
                    <div className="w-8 h-8 rounded-xl bg-[#1e296b]/5 flex items-center justify-center group-hover:bg-[#1e296b]/10 transition-colors">
                        <Plus className="w-5 h-5 text-[#1e296b]" />
                    </div>
                </button>
            </div>

            {/* Recent Analysis Section */}
            <div className="flex-1 px-4 space-y-10 custom-scrollbar overflow-y-auto">
                <div>
                    <p className="px-6 text-[11px] tracking-[0.2em] uppercase opacity-30 font-black mb-6">Recent Activity</p>
                    <div className="space-y-2 px-2">
                        <div className="flex items-center gap-4 px-4 py-4 bg-white/10 text-white rounded-[1.25rem] border border-white/10 shadow-lg cursor-pointer transition-all hover:bg-white/15">
                            <Database className="w-5 h-5 opacity-60" />
                            <span className="font-bold text-sm tracking-tight opacity-90">marketing_v2_logs</span>
                        </div>
                        <div className="flex items-center gap-4 px-4 py-4 text-white/50 hover:text-white hover:bg-white/5 rounded-[1.25rem] cursor-pointer transition-all">
                            <Search className="w-5 h-5 opacity-40" />
                            <span className="font-bold text-sm tracking-tight">customer_id search</span>
                        </div>
                        <div className="flex items-center gap-4 px-4 py-4 text-white/50 hover:text-white hover:bg-white/5 rounded-[1.25rem] cursor-pointer transition-all">
                            <Share2 className="w-5 h-5 opacity-40" />
                            <span className="font-bold text-sm tracking-tight">revenue_summary lineage</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* User Profile */}
            <div className="p-8">
                <div className="bg-white/5 p-5 rounded-[2rem] flex items-center justify-between border border-white/5 backdrop-blur-md shadow-inner">
                    <div className="flex items-center gap-4">
                        <div className="w-12 h-12 rounded-2xl bg-[#ffd5cc] flex items-center justify-center overflow-hidden border-2 border-white/10 shadow-lg">
                            <img src="https://api.dicebear.com/7.x/avataaars/svg?seed=Alex" alt="Avatar" className="w-full h-full" />
                        </div>
                        <div className="flex flex-col">
                            <span className="text-sm font-black text-white tracking-tight">Alex Sterling</span>
                            <span className="text-[10px] uppercase font-black opacity-30 tracking-widest leading-none">Enterprise</span>
                        </div>
                    </div>
                    <button className="p-3 hover:bg-white/10 rounded-2xl transition-all active:scale-90 group text-white/30 hover:text-white">
                        <Settings className="w-6 h-6 transition-transform group-hover:rotate-45" />
                    </button>
                </div>
            </div>
        </aside>
    );
};

export default Sidebar;
