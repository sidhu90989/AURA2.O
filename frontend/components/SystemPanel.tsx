"use client";
import React from 'react';
import useSWR from 'swr';
import { fetchJson, apiBase } from '../lib/api';

interface SystemSnapshot {
  cpu?: number;
  memory?: number;
  disk?: number;
  [k: string]: any; // allow backend to add fields without breaking typing
}

export default function SystemPanel() {
  const { data, error, isLoading } = useSWR<SystemSnapshot>(`${apiBase()}/system/snapshot`, fetchJson, { refreshInterval: 6000 });
  return (
    <div className="flex flex-col h-full">
      <h2 className="font-display text-xl mb-2 glow-text">System Metrics</h2>
      <div className="text-xs font-mono space-y-2 flex-1 overflow-y-auto">
        {isLoading && <div className="animate-pulse opacity-60">Loading...</div>}
        {error && <div className="text-red-400">Error: {(error as Error).message}</div>}
        {data && (
          <ul className="space-y-1">
            <li>CPU: {data.cpu?.toFixed?.(1)}%</li>
            <li>Memory: {data.memory?.toFixed?.(1)}%</li>
            <li>Disk: {data.disk?.toFixed?.(1)}%</li>
          </ul>
        )}
        {!data && !isLoading && !error && <div className="opacity-50">No data</div>}
      </div>
    </div>
  );
}
