"use client";
import React, { useState } from 'react';

export default function VoicePanel() {
  const [status, setStatus] = useState<'idle' | 'recording'>('idle');
  return (
    <div className="flex flex-col h-full">
      <h2 className="font-display text-xl mb-2 glow-text">Voice</h2>
      <div className="flex-1 flex flex-col items-center justify-center gap-4">
        <div className={`h-24 w-24 rounded-full border-4 ${status==='recording' ? 'border-aura-accent animate-pulse' : 'border-white/20'} flex items-center justify-center font-mono text-xs`}>{status.toUpperCase()}</div>
        <button
          onClick={() => setStatus((s: 'idle' | 'recording') => s === 'idle' ? 'recording' : 'idle')}
          className="px-4 py-2 bg-aura-accent/20 border border-aura-accent rounded-md hover:bg-aura-accent/30 transition-colors text-xs font-mono"
        >{status === 'idle' ? 'START' : 'STOP'}</button>
        <p className="text-[10px] opacity-60 text-center px-4">(Placeholder UI — integrate real STT / TTS streaming later)</p>
      </div>
    </div>
  );
}
