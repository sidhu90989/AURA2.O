"use client";
import React, { useEffect, useState } from 'react';
import { apiBase } from '../lib/api';

interface StatusInfo { label: string; status: string; }

function StatusBadge({ label, status }: StatusInfo) {
  return (
    <div className="flex items-center gap-2">
      <div className="status-pulse"><span /><i /></div>
      <span>{label}: {status}</span>
    </div>
  );
}

export default function StatusBar() {
  const [api, setApi] = useState('checking');
  const [mem, setMem] = useState('idle');
  const [nlp, setNlp] = useState('idle');

  async function ping() {
    try {
      const r = await fetch(`${apiBase()}/health`);
      if (r.ok) setApi('online'); else setApi('error');
    } catch { setApi('error'); }
    try {
      const r2 = await fetch(`${apiBase()}/system/snapshot`);
      if (r2.ok) setMem('ready');
    } catch { /* ignore */ }
  }

  useEffect(() => {
    ping();
    const id = setInterval(ping, 8000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="flex items-center gap-6 text-xs font-mono">
      <StatusBadge label="API" status={api} />
      <StatusBadge label="MEM" status={mem} />
      <StatusBadge label="NLP" status={nlp} />
    </div>
  );
}
