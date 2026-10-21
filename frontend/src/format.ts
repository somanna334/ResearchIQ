export interface CiLike {
  mean: number;
  low: number;
  high: number;
  n?: number;
}

export function pct(value: number, digits = 0): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function pctCi(c: CiLike | null): string {
  if (!c) return "–";
  return `${pct(c.mean)} [${pct(c.low)}, ${pct(c.high)}]`;
}

export function ciText(c: CiLike | null, digits = 2): string {
  if (!c) return "–";
  return `${c.mean.toFixed(digits)} [${c.low.toFixed(digits)}, ${c.high.toFixed(digits)}]`;
}

export interface DeltaLike extends CiLike {
  significant: boolean;
}

export function deltaText(d: DeltaLike | null, digits = 2): string {
  if (!d) return "–";
  const sign = (value: number) => (value >= 0 ? "+" : "") + value.toFixed(digits);
  return `${sign(d.mean)} [${sign(d.low)}, ${sign(d.high)}]${d.significant ? " *" : ""}`;
}

export function money(value: number, currency: string): string {
  return `${currency} ${value.toFixed(value < 0.01 ? 4 : 2)}`;
}