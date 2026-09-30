const listeners = new WeakMap();

// Retained elements can be rebound by more than one lazy route implementation.
// Replace that owner's binding while preserving listeners owned by HA or others.
export function bindOwnedEvent(element, type, owner, listener) {
  if (!element) return;
  let bindings = listeners.get(element);
  if (!bindings) listeners.set(element, bindings = new Map());
  const key = `${owner}:${type}`;
  bindings.get(key)?.abort();
  const controller = new AbortController();
  bindings.set(key, controller);
  element.addEventListener(type, listener, {signal:controller.signal});
}
