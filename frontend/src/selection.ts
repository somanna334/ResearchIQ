/** Toggle an id in a selection that may hold at most `max` items. */
export function toggleLimited(selected: string[], id: string, max: number): string[] {
  if (selected.includes(id)) return selected.filter((x) => x !== id);
  if (selected.length >= max) return selected;
  return [...selected, id];
}