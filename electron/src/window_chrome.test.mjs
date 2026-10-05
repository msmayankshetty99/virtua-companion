import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import fs from 'node:fs';
import {simpleHelp} from './plain_language.mjs';
const {windowAction}=createRequire(import.meta.url)('../window_actions.cjs');

test('custom controls minimize, toggle maximize and close through existing window handling',()=>{
  const calls=[];let maximized=false;
  const w={minimize:()=>calls.push('minimize'),maximize:()=>{maximized=true;calls.push('maximize');},unmaximize:()=>{maximized=false;calls.push('restore');},close:()=>calls.push('close'),isMaximized:()=>maximized,isDestroyed:()=>false};
  windowAction(w,'minimize');assert.equal(windowAction(w,'maximize').maximized,true);assert.equal(windowAction(w,'maximize').maximized,false);windowAction(w,'close');
  assert.deepEqual(calls,['minimize','maximize','restore','close']);assert.throws(()=>windowAction(w,'quit'));
});
test('window IPC is scoped to interactive windows and hidden windows are recoverable',()=>{
  const host=fs.readFileSync(new URL('../main.cjs',import.meta.url),'utf8');
  assert.match(host,/function chromeWindow\(event\).*\[control,whiteboard\]/);
  assert.match(host,/new Tray/);assert.match(host,/if\(control.isMinimized\(\)\)control.restore\(\)/);
  assert.match(host,/frame:false, transparent:true, backgroundColor:'#00000000', skipTaskbar:true/);
});
test('simple help explains effects without exposing the full technical note',()=>{
  const field={path:'runtime.n_ctx',help:'KV pool demand etc.'};
  assert.match(simpleHelp(field),/Larger values use more memory/);
  assert.equal(field.help,'KV pool demand etc.');
  assert.match(simpleHelp({path:'custom',kind:'boolean',help:'JSON override.'}),/Turn this feature on or off/);
});
test('chat modes use one native surface, retaining the overlay no-focus boundary',()=>{
 const host=fs.readFileSync(new URL('../main.cjs',import.meta.url),'utf8');
 assert.match(host,/overlayPointer.hover\(key,!!enabled\)/);
 assert.match(host,/overlayPointer.drag\(value.source,value.active\)/);
 assert.doesNotMatch(host,/dockInteractive|open-compact-chat|dock-interactive/);
 assert.match(host,/fullControlBounds=control.getNormalBounds\(\)/);
 assert.match(host,/control.setMinimumSize\(88,88\)/);
 assert.match(host,/control.setAspectRatio\(0\)/);
 assert.match(host,/control.setFocusable\(mode!=='collapsed'\)/);
 assert.match(host,/tweenBounds\(control,target,'chat'/);
 assert.match(host,/mode==='full'\?fullControlBounds/);
});
test('authored branding images come from the root assets folder',()=>{
 const branding=fs.readFileSync(new URL('./branding.mjs',import.meta.url),'utf8');
 const host=fs.readFileSync(new URL('../main.cjs',import.meta.url),'utf8');
 assert.match(branding,/\.\.\/\.\.\/assets\/logo.svg/);assert.match(host,/path.join\(root,'assets','tray.png'\)/);
 assert.ok(fs.statSync(new URL('../../assets/tray.png',import.meta.url)).size>0);
});
test('quitting never stops at a silent Settings unload veto, and asks before the backend stops',()=>{
 const host=fs.readFileSync(new URL('../main.cjs',import.meta.url),'utf8'),preload=fs.readFileSync(new URL('../preload.cjs',import.meta.url),'utf8');
 assert.match(host,/control\.webContents\.on\('will-prevent-unload',event=>\{if\(app\.isQuitting\|\|dialog\.showMessageBoxSync\(control,/);
 assert.match(host,/ipcMain\.on\('settings-dirty',\(event,dirty\)=>\{if\(control&&!control\.isDestroyed\(\)&&event\.sender\.id===control\.webContents\.id\)/);
 const quit=host.slice(host.indexOf("app.on('before-quit'"));
 assert.ok(quit.indexOf('settingsDirty')<quit.indexOf('app.isQuitting = true')&&quit.indexOf('app.isQuitting = true')<quit.indexOf('stopBackend'));
 assert.match(preload,/settingsDirty: dirty => ipcRenderer\.send\('settings-dirty', dirty === true\)/);
});
test('standalone setup and training-data windows paint the dark page background',()=>{
 const host=fs.readFileSync(new URL('../main.cjs',import.meta.url),'utf8'),css=fs.readFileSync(new URL('./style.css',import.meta.url),'utf8');
 const bg=css.match(/--bg:(#[0-9a-f]{6})/)[1];
 for(const name of ['setupWindow','neuralDataWindow'])assert.match(host,new RegExp(name+"=new BrowserWindow\\(\\{[^}]*backgroundColor:'"+bg+"'"),name);
});
