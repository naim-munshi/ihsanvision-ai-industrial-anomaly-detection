const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('demo/auth.js', 'utf8');
async function mount(user) {
  const calls = [], navigation = [], elements = new Map();
  const element = () => ({style:{}, listeners:{}, children:[], value:'', appendChild(x){this.children.push(x);}, addEventListener(k,f){this.listeners[k]=f;}, click(){return this.listeners.click({preventDefault(){}});}});
  elements.set('api-key',element()); elements.set('lock',element());
  let loaded, bar;
  const window = {fetch:async (input,options={})=>{
    calls.push({input,options});
    if(input==='/auth/me')return {ok:true,status:200,json:async()=>({enabled:true,user})};
    return {ok:input!=='/expired',status:input==='/expired'?401:200};
  }};
  vm.runInNewContext(source,{window,URL,Headers,location:{href:'http://127.0.0.1:8004/history',origin:'http://127.0.0.1:8004',assign:x=>navigation.push(x),replace:x=>navigation.push(x)},document:{addEventListener:(event,callback)=>{loaded=callback;},createElement:element,getElementById:id=>elements.get(id),querySelector:()=>({prepend:x=>{bar=x;}})}});
  await loaded();
  return {window,calls,elements,navigation,get bar(){return bar;}};
}
const user={username:'operator-one',role:'operator',csrf:'test-csrf'};
test('session controls attach CSRF only to same-origin mutations and keep key empty',async()=>{
  const ui=await mount(user);
  assert.equal(ui.window.ihsanSession,true);
  assert.equal(ui.elements.get('api-key').value,'');
  assert.equal(ui.elements.get('api-key').disabled,true);
  await ui.window.fetch('/predict',{method:'POST'});
  assert.equal(ui.calls.at(-1).options.headers.get('X-IhsanVision-CSRF'),'test-csrf');
  await ui.window.fetch('/inspections');
  assert.equal(ui.calls.at(-1).options.headers.get('X-IhsanVision-CSRF'),null);
  await ui.window.fetch(new URL('https://external.example/'),{method:'POST'});
  assert.equal(ui.calls.at(-1).options.headers,undefined);
});
test('signout and history lock revoke server session before navigating',async()=>{
  const ui=await mount(user);
  await ui.elements.get('lock').click();
  assert.equal(ui.calls.at(-1).input,'/auth/logout');
  assert.equal(ui.calls.at(-1).options.headers.get('X-IhsanVision-CSRF'),'test-csrf');
  assert.deepEqual(ui.navigation,['/login']);
});
test('expired session clears local authenticated state and navigates to login',async()=>{
  const ui=await mount(user);
  await ui.window.fetch('/expired');
  assert.equal(ui.window.ihsanSession,false);
  assert.deepEqual(ui.navigation,['/login']);
});
test('logged-out users get login link with no session authorization',async()=>{
  const ui=await mount(null);
  assert.equal(ui.window.ihsanSession,false);
  assert.equal(ui.bar.children[1].href,'/login');
  await ui.window.fetch('/predict',{method:'POST'});
  assert.equal(ui.calls.at(-1).options.headers.get('X-IhsanVision-CSRF'),null);
});
