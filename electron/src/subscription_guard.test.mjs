import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
test('status consumers and Electron host have no periodic polling timers',()=>{
  for(const file of ['animation_library.jsx','initiative_settings.jsx','voice_input.jsx','tool_approvals.jsx','gpu_resources.jsx','stream_chat.jsx','../main.cjs','../overlay_input.cjs']){
    assert.doesNotMatch(readFileSync(new URL(file,import.meta.url),'utf8'),/\bsetInterval\s*\(/,file);
  }
});
test('the only host cursor poll stands in for click-through forwarding, which Linux lacks',()=>{
  const host=readFileSync(new URL('../main.cjs',import.meta.url),'utf8');
  assert.match(host,/const linuxPointer=process\.platform==='linux'\?pointerForwarder\(/);
  assert.equal((host.match(/pointerForwarder\(/g)||[]).length,1);
  // Like Electron's forwarding, only windows ignoring the mouse get forwarded moves.
  assert.match(host,/targets:\(\)=>\[!overlayPointer\.interactive&&!coveredByWindow\(screen\.getCursorScreenPoint\(\)\)&&overlay,controlIgnoring&&!windowGesture&&control\]/);
  assert.match(host,/setIgnoreMouseEvents\(controlIgnoring=mode!=='full'/);assert.match(host,/setIgnoreMouseEvents\(controlIgnoring=!enabled/);
});
