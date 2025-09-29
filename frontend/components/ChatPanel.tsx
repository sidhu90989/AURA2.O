"use client";
import React, { useState, useRef, ChangeEvent, KeyboardEvent } from 'react';
import { apiBase, postJson } from '../lib/api';

interface ChatMessage { role: 'user' | 'assistant' | 'system'; content: string }

export default function ChatPanel() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: 'system', content: 'Hello AURA' }
  ]);
  const [input, setInput] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const viewportRef = useRef<HTMLDivElement | null>(null);

  async function send() {
    if (!input.trim()) return;
    const userMsg: ChatMessage = { role: 'user', content: input.trim() };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput("");
    setLoading(true);
    try {
      const payload = { messages: newMessages.map(m => ({ role: m.role, content: m.content })), store: true };
      const res = await postJson(`${apiBase()}/nlp/chat`, payload);
      if (res?.response) {
        setMessages([...newMessages, { role: 'assistant', content: res.response }]);
      }
    } catch (e) {
      setMessages([...newMessages, { role: 'assistant', content: `[error] ${(e as Error).message}` }]);
    } finally {
      setLoading(false);
      requestAnimationFrame(() => viewportRef.current?.scrollTo({ top: viewportRef.current.scrollHeight, behavior: 'smooth' }));
    }
  }

  return (
    <div className="flex flex-col h-full">
      <h2 className="font-display text-xl mb-2 glow-text">Chat Interface</h2>
      <div ref={viewportRef} className="flex-1 overflow-y-auto pr-2 space-y-2 text-sm leading-relaxed">
        {messages.map((m: ChatMessage, i: number) => (
          <div key={i} className={`rounded-md px-3 py-2 bg-white/5 border border-white/10 whitespace-pre-wrap ${m.role === 'user' ? 'ml-6' : 'mr-6'}`}>
            <span className="font-semibold mr-1 text-aura-accent/80">{m.role.toUpperCase()}:</span>{m.content}
          </div>
        ))}
        {loading && <div className="text-xs opacity-70 animate-pulse">Thinking...</div>}
      </div>
      <div className="mt-3 flex gap-2 items-center">
        <input
          className="flex-1 bg-black/30 border border-white/10 rounded-md px-3 py-2 text-sm focus:outline-none focus:border-aura-accent focus:ring-1 focus:ring-aura-accent"
          placeholder="Type a message..."
          value={input}
          onChange={(e: ChangeEvent<HTMLInputElement>) => setInput(e.target.value)}
          onKeyDown={(e: KeyboardEvent<HTMLInputElement>) => { if (e.key === 'Enter') send(); }}
        />
        <button onClick={send} disabled={loading} className="px-4 py-2 text-sm font-mono bg-gradient-to-r from-aura-accent to-aura-accent2 rounded-md shadow-aura-glow disabled:opacity-50">SEND</button>
      </div>
    </div>
  );
}
