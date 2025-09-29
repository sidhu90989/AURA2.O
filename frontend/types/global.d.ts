/// <reference types="react" />
/// <reference types="react-dom" />

// Augment the JSX namespace if needed (prevents implicit any for intrinsic elements)
import React from 'react';
declare global {
  namespace JSX {
    // allow any intrinsic element (fallback) while strict types come from React definitions
    interface IntrinsicElements {
      [elemName: string]: any;
    }
  }
}
