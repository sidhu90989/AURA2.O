"use client";
import React, { useState } from 'react';
import { postJson, fetchJson, apiBase } from '../lib/api';

export default function GraphPanel() {
  const [text,setText] = useState('Testing graph node');
  const [emotion,setEmotion] = useState('neutral');
  const [result,setResult] = useState<any>(null);
  const [related,setRelated] = useState<any[]>([]);

  async function store(): Promise<void> {
    const r = await postJson(`${apiBase()}/graph/store`, { text, emotion });
    setResult(r);
  }
  async function query(): Promise<void> {
    const r = await fetchJson(`${apiBase()}/graph/related?q=${encodeURIComponent(text)}`);
    setRelated(r.results || []);
  }

  return (
    <div className="flex flex-col h-full">
      <h2 className="font-display text-xl mb-2 glow-text">Knowledge Graph</h2>
      <div className="flex gap-2 mb-2 text-xs">
  <input className="flex-1 bg-black/30 border border-white/10 rounded px-2 py-1" value={text} onChange={(e: React.ChangeEvent<HTMLInputElement>)=>setText(e.target.value)} />
  <select className="bg-black/30 border border-white/10 rounded px-2" value={emotion} onChange={(e: React.ChangeEvent<HTMLSelectElement>)=>setEmotion(e.target.value)}>
          <option value="neutral">neutral</option>
          <option value="joy">joy</option>
          <option value="sadness">sadness</option>
          <option value="anger">anger</option>
        </select>
        <button onClick={store} className="px-3 py-1 border border-aura-accent bg-aura-accent/20 rounded">STORE</button>
        <button onClick={query} className="px-3 py-1 border border-aura-accent2 bg-aura-accent2/20 rounded">RELATED</button>
      </div>
      <div className="text-[11px] font-mono space-y-1 overflow-y-auto flex-1 pr-1">
        {result && <div className="border border-white/10 rounded px-2 py-1">Stored: {result.id}</div>}
        {related.map((r: any, i: number) => (
          <div key={i} className="border border-white/5 rounded px-2 py-1">{JSON.stringify(r)}</div>
        ))}
      </div>
    </div>
  );
}
