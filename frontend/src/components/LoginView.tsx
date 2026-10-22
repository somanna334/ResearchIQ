import { useState } from "react";
import type { FormEvent } from "react";
import { login, register } from "../auth";
import type { User } from "../auth";
import { MIN_PASSWORD_LENGTH, validateRegistration } from "../authForm";
import { setToken } from "../session";
import "../Auth.css";

interface Props {
  onAuthenticated: (user: User) => void;
  notice?: string | null;
}

export default function LoginView({ onAuthenticated, notice }: Props) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    if (mode === "register") {
      const problem = validateRegistration(email, password, confirm);
      if (problem) {
        setError(problem);
        return;
      }
    }
    setPending(true);
    try {
      const result =
        mode === "login"
          ? await login(email.trim(), password)
          : await register(email.trim(), password);
      setToken(result.access_token);
      onAuthenticated(result.user);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setPending(false);
    }
  }

  const registering = mode === "register";
  return (
    <div className="auth">
      <div className="auth-card">
        <h1>ResearchIQ</h1>
        <p className="muted small">Answers grounded in your documents</p>
        {notice && <div className="banner bad small">{notice}</div>}
        <form onSubmit={(e) => void onSubmit(e)}>
          <label>
            Email
            <input
              type="email"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </label>
          <label>
            Password
            <input
              type="password"
              autoComplete={registering ? "new-password" : "current-password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </label>
          {registering && (
            <>
              <label>
                Confirm password
                <input
                  type="password"
                  autoComplete="new-password"
                  value={confirm}
                  onChange={(e) => setConfirm(e.target.value)}
                  required
                />
              </label>
              <p className="muted small">
                Use at least {MIN_PASSWORD_LENGTH} characters. A few random words work well.
              </p>
            </>
          )}
          {error && <div className="banner bad small">{error}</div>}
          <button type="submit" className="primary" disabled={pending}>
            {pending ? "Please wait…" : registering ? "Create account" : "Sign in"}
          </button>
        </form>
        <p className="auth-switch small">
          {registering ? "Already have an account? " : "New here? "}
          <button
            className="link-btn"
            onClick={() => {
              setMode(registering ? "login" : "register");
              setError(null);
            }}
          >
            {registering ? "Sign in" : "Create an account"}
          </button>
        </p>
      </div>
    </div>
  );
}