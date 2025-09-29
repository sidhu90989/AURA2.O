import './globals.css';
import React from 'react';
import type { ReactNode } from 'react';
import StatusBar from '../components/StatusBar';

export const metadata = {
  title: 'AURA2.O HUD',
  description: 'Jarvis-style AI operations dashboard'
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body>
        <div className="scanline" />
        <main className="relative max-w-7xl mx-auto px-6 py-8 space-y-6">
          <Header />
          {children}
        </main>
      </body>
    </html>
  );
}

function Header() {
  return (
    <header className="flex items-center justify-between mb-4">
      <h1 className="text-3xl font-display glow-text tracking-widest">AURA2.O</h1>
      <StatusBar />
    </header>
  );
}
