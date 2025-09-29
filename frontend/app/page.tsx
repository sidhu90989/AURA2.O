import React from 'react';
import ChatPanel from '../components/ChatPanel';
import VoicePanel from '../components/VoicePanel';
import SystemPanel from '../components/SystemPanel';
import MemoryPanel from '../components/MemoryPanel';
import GraphPanel from '../components/GraphPanel';

export default function Dashboard() {
  return (
    <div className="grid gap-6 md:grid-cols-12 auto-rows-[minmax(180px,auto)]">
      <section className="aura-panel md:col-span-5 row-span-3"><ChatPanel /></section>
      <section className="aura-panel md:col-span-3 row-span-2"><VoicePanel /></section>
      <section className="aura-panel md:col-span-4 row-span-2"><SystemPanel /></section>
      <section className="aura-panel md:col-span-4 row-span-2"><MemoryPanel /></section>
      <section className="aura-panel md:col-span-8 row-span-2"><GraphPanel /></section>
    </div>
  );
}
