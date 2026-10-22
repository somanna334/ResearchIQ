const KEY = "researchiq.token";

let memoryToken: string | null = null;
let onUnauthorized: (() => void) | null = null;

export function getToken(): string | null {
  if (memoryToken) return memoryToken;
  try {
    return localStorage.getItem(KEY);
  } catch {
    return null; // storage can be blocked; the in-memory token still works for this tab
  }
}

export function setToken(token: string): void {
  memoryToken = token;
  try {
    localStorage.setItem(KEY, token);
  } catch {
    // Not persisted: the user will need to sign in again after a reload.
  }
}

export function clearToken(): void {
  memoryToken = null;
  try {
    localStorage.removeItem(KEY);
  } catch {
    // Nothing to clear.
  }
}

/** The app registers one handler that returns to the sign-in screen. */
export function setUnauthorizedHandler(handler: (() => void) | null): void {
  onUnauthorized = handler;
}

export function notifyUnauthorized(): void {
  onUnauthorized?.();
}