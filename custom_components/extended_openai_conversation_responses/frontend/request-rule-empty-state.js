export function requestRuleEmptyContent(query, groupFilter) {
  return !String(query || "").trim() && groupFilter && groupFilter !== "all"
    ? '<h2>No rules in this group</h2>'
    : '<h2>No rules match your search</h2><p>Try a different phrase or rule name.</p>';
}

export function updateRequestRuleEmptyState(empty, query, groupFilter) {
  if (!empty) return;
  const content = requestRuleEmptyContent(query, groupFilter);
  if (empty.innerHTML !== content) empty.innerHTML = content;
}
