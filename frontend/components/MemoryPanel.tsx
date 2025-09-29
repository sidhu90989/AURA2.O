"use client";
import React, { useState, useEffect } from 'react';
import { fetchJson, apiBase } from '../lib/api';

interface MemoryItem { id:number; text:string; }

export default function MemoryPanel() {
  const [items, setItems] = useState<MemoryItem[]>([]);
  const [q,setQ] = useState('AURA');
  const [loading,setLoading] = useState(false);

  async function search(): Promise<void> {
    setLoading(true);
    try {
      const res = await fetchJson(`${apiBase()}/memory/search?q=${encodeURIComponent(q)}`);
      // results is list of [embedding_ref, distance]; we just display
      setItems(res.results.map((r:any, i:number) => ({ id:i, text:`${r[0]} (d=${r[1].toFixed?.(2)})` })));
    } catch(e) {
      setItems([{id:0,text:'error'}]);
    } finally { setLoading(false); }
  }

  useEffect(() => { search(); }, []);

  return (
    <div className="flex flex-col h-full">
      <h2 className="font-display text-xl mb-2 glow-text">Memory</h2>
      <div className="flex gap-2 mb-2">
  <input value={q} onChange={(e: React.ChangeEvent<HTMLInputElement>)=>setQ(e.target.value)} className="flex-1 bg-black/30 border border-white/10 rounded px-2 py-1 text-xs" />
        <button onClick={search} className="px-3 py-1 text-xs bg-aura-accent/30 border border-aura-accent rounded">GO</button>
      </div>
      <div className="flex-1 overflow-y-auto text-[11px] font-mono space-y-1 pr-1">
        {loading && <div className="animate-pulse">Loading...</div>}
        {!loading && items.map((m: MemoryItem) => (
          <div key={m.id} className="truncate border border-white/5 bg-white/5 rounded px-2 py-1">{m.text}</div>
        ))}
      </div>
    </div>
  );
}
