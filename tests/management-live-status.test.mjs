import assert from "node:assert/strict";
import {watchLiveStatus, stopLiveStatus} from "../custom_components/extended_openai_conversation_responses/frontend/management-live-status.js";
import {bindOwnedEvent} from "../custom_components/extended_openai_conversation_responses/frontend/management-owned-events.js";
import {hasPendingBroadcast} from "../custom_components/extended_openai_conversation_responses/frontend/overview-broadcast.js";
import {ensureRequestRuleTranslations} from "../custom_components/extended_openai_conversation_responses/frontend/request-rules-ui-impl.js";

const target = new EventTarget();
let calls = 0;
for (let i=0;i<10;i++) bindOwnedEvent(target, "click", "assistant-actions", () => calls++);
target.dispatchEvent(new Event("click"));
assert.equal(calls,1);
target.addEventListener("click",()=>calls++);
bindOwnedEvent(target,"click","assistant-actions",()=>calls+=10);
target.dispatchEvent(new Event("click"));
assert.equal(calls,12);

const native = {hass:null};
let loads=0;
const translated = key => `Translated ${key}`;
const translationPanel = {_hass:{language:"en",connection:{},loadFragmentTranslation:async fragment => {assert.equal(fragment,"config");loads++;return translated;}},shadowRoot:{querySelectorAll:()=>[native]}};
await Promise.all([ensureRequestRuleTranslations(translationPanel),ensureRequestRuleTranslations(translationPanel)]);
assert.equal(loads,1);
assert.equal(native.hass.localize,translated);
translationPanel._hass={...translationPanel._hass,language:"de"};
await ensureRequestRuleTranslations(translationPanel);
assert.equal(loads,2);

const timers=new Map();let serial=0;
const realSet=globalThis.setTimeout, realClear=globalThis.clearTimeout;
globalThis.setTimeout=callback=>{timers.set(++serial,callback);return serial;};
globalThis.clearTimeout=id=>timers.delete(id);
const panel={_agentId:"a",_data:{entry_id:"e"},isConnected:true,_viewKey:()=>"overview"};
let reads=0, release;
try {
  const options={view:"overview",delay:1000,refresh:async current=>{reads++;await new Promise(resolve=>release=()=>{assert.equal(current(),false);resolve();});}};
  watchLiveStatus(panel,"broadcast",options);
  watchLiveStatus(panel,"broadcast",options);
  assert.equal(timers.size,1);
  const callback=timers.values().next().value;timers.clear();
  const pending=callback();
  assert.equal(reads,1);
  stopLiveStatus(panel);
  watchLiveStatus(panel,"broadcast",{view:"overview",delay:1000,refresh:async()=>reads++});
  release();await pending;
  assert.equal(timers.size,1);
  stopLiveStatus(panel);
  assert.equal(timers.size,0);
  watchLiveStatus(panel,"bounded",{view:"overview",delay:1000,maxRefreshes:2,refresh:async()=>reads++});
  for(let i=0;i<2;i++){const next=timers.values().next().value;timers.clear();await next();}
  assert.equal(timers.size,0);
  watchLiveStatus(panel,"route",{view:"overview",delay:1000,refresh:async()=>reads++});
  panel._agentId="b";
  const next=timers.values().next().value;timers.clear();await next();
  assert.equal(reads,3);
  assert.equal(timers.size,0);
} finally {stopLiveStatus(panel);globalThis.setTimeout=realSet;globalThis.clearTimeout=realClear;}
assert.equal(hasPendingBroadcast({history:[{deliveries:{speaker:{status:"waiting_idle"}}}]}),true);
assert.equal(hasPendingBroadcast({history:[{deliveries:{speaker:{status:"delivered"}}}]}),false);
assert.equal(hasPendingBroadcast({history:[{deliveries:{speaker:{status:"failed"}}}]}),false);
