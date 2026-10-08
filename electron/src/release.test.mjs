import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createRequire} from 'node:module';
import {EventEmitter} from 'node:events';
const require=createRequire(import.meta.url);
const {hardware,configuration,saveSetup,installRoot,askMicrophone,startSovits,stopBackend,watchListening,finderPath,LISTENING}=require('../release.cjs');
const {MANIFEST}=require('../native_backends.cjs');
const YAML=require('yaml');
// A packaged app's resources with one native bundle. The tests pass the OS explicitly, so every host checks the same one.
function fixture(backend='cuda',library='libriko-native.so'){const root=fs.mkdtempSync(path.join(os.tmpdir(),'riko-release-'));const folder=path.join(root,'native',backend);fs.mkdirSync(folder,{recursive:true});fs.writeFileSync(path.join(folder,library),'test');return root;}
const form={backend:'cuda',repo:'owner/model',filename:'model.gguf',context:8192,output:1024,threads:4,memories:'I like tea.\nMy name is Alex.',sovitsAuto:false};
test('setup separates persistent data from binaries and refuses overwrite',()=>{const root=fixture();const directory=fs.mkdtempSync(path.join(os.tmpdir(),'riko-data-'));try{saveSetup(directory,form,root,root,'linux');const config=YAML.parse(fs.readFileSync(path.join(directory,'character_config.yaml'),'utf8'));assert.equal(config.runtime.native_library,'bundled:cuda');assert.equal(config.memory.default_memories.length,2);assert.equal(config.tools.require_approval,true);assert.equal(config.emotion.probe.enabled,false);assert.deepEqual(config.voice,{wake_word:'Riko',asr_device:'cpu',asr_compute_type:'int8'});assert.ok(fs.existsSync(path.join(directory,'models')));assert.throws(()=>saveSetup(directory,form,root,root,'linux'),/EEXIST/);assert.throws(()=>saveSetup(root,form,root,root,'linux'),/outside/);}finally{fs.rmSync(root,{recursive:true,force:true});fs.rmSync(directory,{recursive:true,force:true});}});
test('setup wakes on the first word of any companion name and writes YAML the backend reads as text',()=>{
  const root=fixture(),directory=fs.mkdtempSync(path.join(os.tmpdir(),'riko-data-'));
  try{
    for(const [name,saved,wake] of [['Riko Chan','Riko Chan','Riko'],['  Ai  Hoshino ','Ai  Hoshino','Ai'],['','Riko','Riko'],[undefined,'Riko','Riko'],['Yes','Yes','Yes']]){
      const config=configuration({...form,name},root,'linux');
      assert.equal(config.presets.default.name,saved);assert.equal(config.voice.wake_word,wake);
    }
    // The backend's PyYAML reads YAML 1.1, where an unquoted Yes, on or 42 is not text.
    saveSetup(directory,{...form,name:'Yes',memories:'on\n42'},root,root,'linux');
    const config=YAML.parse(fs.readFileSync(path.join(directory,'character_config.yaml'),'utf8'),{version:'1.1'});
    assert.equal(config.presets.default.name,'Yes');assert.equal(config.voice.wake_word,'Yes');
    assert.deepEqual(config.memory.default_memories.map(memory=>memory.text),['on','42']);
  }finally{fs.rmSync(root,{recursive:true,force:true});fs.rmSync(directory,{recursive:true,force:true});}
});
test('setup validates budgets, model and explicit speech consent',()=>{const root=fixture();try{assert.throws(()=>configuration({...form,output:8192},root,'linux'),/budget/);assert.throws(()=>configuration({...form,filename:'../secret.gguf'},root,'linux'),/GGUF/);assert.throws(()=>configuration({...form,sovitsAuto:true},root,'linux'),/GPT-SoVITS/);assert.equal(startSovits({auto_start:false}),null);assert.throws(()=>startSovits({auto_start:true,executable:'relative.exe'}),/missing/);}finally{fs.rmSync(root,{recursive:true,force:true});}});
test('setup leaves room for the reply inside the live context, as Settings requires',()=>{const root=fixture();try{for(const [context,output] of [[8192,1024],[2048,2047],[131072,64]]){const {runtime,memory}=configuration({...form,context,output},root,'linux');assert.equal(runtime.n_ctx,context);assert.equal(runtime.max_output_tokens,output);assert.ok(memory.context_window_tokens>=1);assert.ok(memory.context_window_tokens+runtime.max_output_tokens<=runtime.n_ctx);}}finally{fs.rmSync(root,{recursive:true,force:true});}});
test('setup writes bundled:<backend> only for a bundle this OS ships, under its own library name',()=>{
  for(const [platform,backend,library] of [['win32','cuda','riko-native.dll'],['win32','vulkan','riko-native.dll'],['linux','vulkan','libriko-native.so'],['darwin','metal','libriko-native.dylib']]){
    const root=fixture(backend,library);
    try{assert.equal(configuration({...form,backend},root,platform).runtime.native_library,'bundled:'+backend);}finally{fs.rmSync(root,{recursive:true,force:true});}
  }
  const windows=fixture('cuda','riko-native.dll'),mac=fixture('cuda','libriko-native.so');
  try{
    assert.throws(()=>configuration(form,windows,'linux'),/Packaged native backend is missing: .*libriko-native\.so/);
    assert.throws(()=>configuration(form,mac,'darwin'),/^Error: Choose Metal$/);
    assert.throws(()=>configuration({...form,backend:'metal'},mac,'win32'),/^Error: Choose CUDA or Vulkan$/);
    assert.throws(()=>configuration({...form,backend:'../native'},mac,'linux'),/Choose CUDA or Vulkan/);
    assert.throws(()=>configuration(form,mac,'freebsd'),/No native backend ships for freebsd/);
  }finally{fs.rmSync(windows,{recursive:true,force:true});fs.rmSync(mac,{recursive:true,force:true});}
});
test('hardware detection runs only the tools for this OS\'s bundles and preselects from them',async()=>{
  const probe=outputs=>{const calls=[];return {calls,run:(file,args)=>{calls.push([file,...args]);return Promise.resolve(outputs[file]||'');}};};
  const mac=probe({'nvidia-smi':'NVIDIA','vulkaninfo':'MoltenVK'}),macHw=await hardware('darwin',mac.run);
  assert.deepEqual(mac.calls,[]);
  assert.deepEqual(macHw.backends,[{id:'metal',label:'Metal',description:'Apple silicon GPU'}]);
  assert.deepEqual(macHw.detections,[]);assert.equal(macHw.preferred,'metal');assert.equal(macHw.platform,'darwin');
  const linux=probe({'vulkaninfo':'GPU0: AMD Radeon RX 7900'}),linuxHw=await hardware('linux',linux.run);
  assert.deepEqual(linux.calls.map(call=>call[0]),['nvidia-smi','vulkaninfo']);
  assert.equal(linuxHw.preferred,'vulkan');
  assert.deepEqual(linuxHw.detections.map(item=>[item.backend,item.required,item.output]),[['cuda',true,'No NVIDIA GPU detected via nvidia-smi'],['vulkan',false,'GPU0: AMD Radeon RX 7900']]);
  const windows=await hardware('win32',probe({'nvidia-smi':'NVIDIA GeForce RTX 4090, 24564, 551.86'}).run);
  assert.equal(windows.preferred,'cuda');assert.deepEqual(windows.backends.map(backend=>backend.id),['cuda','vulkan']);
});
test('the setup page offers the backends hardware() reports instead of a fixed list',()=>{
  const page=fs.readFileSync(new URL('./first_setup.jsx',import.meta.url),'utf8');
  assert.doesNotMatch(page,/value="(cuda|vulkan|metal)"|backend:'(cuda|vulkan|metal)'|hw\.(nvidia|vulkan)\b/);
  assert.match(page,/\(hw\?\.backends\|\|\[\]\)\.map\(backend=><option key=\{backend\.id\} value=\{backend\.id\}>/);
  assert.match(page,/backend:h\.preferred/);
});
test('setup refuses a data folder that an update or uninstall would delete',()=>{
  const install=fs.mkdtempSync(path.join(os.tmpdir(),'riko-install-')),resources=path.join(install,'resources'),sibling=install+'-data';
  fs.mkdirSync(path.join(resources,'native','cuda'),{recursive:true});fs.writeFileSync(path.join(resources,'native','cuda','libriko-native.so'),'test');
  try{
    for(const directory of [install,path.join(install,'data'),path.join(install,'data','nested'),resources])assert.throws(()=>saveSetup(directory,form,resources,install,'linux'),/outside the application folder/);
    assert.ok(!fs.existsSync(path.join(install,'data')));
    assert.equal(saveSetup(sibling,form,resources,install,'linux'),sibling);assert.ok(fs.existsSync(path.join(sibling,'character_config.yaml')));
  }finally{fs.rmSync(install,{recursive:true,force:true});fs.rmSync(sibling,{recursive:true,force:true});}
  assert.equal(installRoot('/Applications/Riko.app/Contents/MacOS/Riko','darwin'),'/Applications/Riko.app');
  assert.equal(installRoot('/opt/Riko/riko-companion','linux'),'/opt/Riko');
  assert.match(fs.readFileSync(new URL('../main.cjs',import.meta.url),'utf8'),/release\.saveSetup\(values\.directory,values\.settings,process\.resourcesPath,release\.installRoot\(app\.getPath\('exe'\)\)\)/);
});
test('a second packaged launch exits before it can spawn a backend, GPT-SoVITS or windows',()=>{
  const host=fs.readFileSync(new URL('../main.cjs',import.meta.url),'utf8');
  const lock=host.indexOf("if(app.isPackaged&&!app.requestSingleInstanceLock()){app.quit();return;}");
  assert.ok(lock>0,'the losing instance must stop evaluating main.cjs');
  for(const later of ['app.whenReady(','ipcMain.','app.on(',"require('./release.cjs')",'new BrowserWindow('])assert.ok(host.indexOf(later)>lock,later+' must come after the single-instance exit');
});
test('main learns the backend holds its port only from a complete marker line',()=>{let calls=0;const watch=watchListening(()=>calls++);watch('Riko AI server: http://127.0.0.1:8765\nRIKO_BACKEND_');assert.equal(calls,0);watch('LISTENING_NOT\n');assert.equal(calls,0);watch(LISTENING+'\r\nINFO later\n');assert.equal(calls,1);watch(LISTENING+'\n');assert.equal(calls,1);const split=watchListening(()=>calls++);for(const part of ['RIKO_BACK','END_LISTEN','ING','\n'])split(Buffer.from(part));assert.equal(calls,2);});
function fakeChild({exitOnShutdown=false,signalCode=null}={}){const child=Object.assign(new EventEmitter(),{exitCode:null,signalCode,signals:[],written:'',kill(signal='SIGTERM'){child.signals.push(signal);return true;}});child.stdin={end(text){child.written+=text;if(exitOnShutdown)setImmediate(()=>child.emit('exit',0,null));}};return child;}
test('stopBackend asks for shutdown, then escalates to SIGTERM and SIGKILL',async()=>{const hung=fakeChild();await stopBackend(hung,20,20);assert.equal(hung.written,'shutdown\n');assert.deepEqual(hung.signals,['SIGTERM','SIGKILL']);const quick=fakeChild({exitOnShutdown:true});await stopBackend(quick,20,20);assert.deepEqual(quick.signals,[]);const crashed=fakeChild({signalCode:'SIGABRT'}),started=Date.now();await stopBackend(crashed,1000,1000);assert.ok(Date.now()-started<500);assert.equal(crashed.written,'');});
test('a Finder-launched macOS backend still finds Homebrew tools for MCP servers and FFmpeg',()=>{
  assert.equal(finderPath('/usr/bin:/bin:/usr/sbin:/sbin'),'/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin');
  assert.equal(finderPath('/usr/local/bin:/opt/homebrew/bin:/usr/bin'),'/opt/homebrew/sbin:/usr/local/bin:/opt/homebrew/bin:/usr/bin');
  assert.equal(finderPath(undefined),'/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin');
  assert.match(fs.readFileSync(new URL('../release.cjs',import.meta.url),'utf8'),/process\.platform==='darwin'\?\{PATH:finderPath\(process\.env\.PATH\)\}/);
});
test('Linux launches keep the AppImage sandbox flag and run under XWayland',()=>{const builder=YAML.parse(fs.readFileSync(new URL('../electron-builder.yml',import.meta.url),'utf8'));assert.deepEqual(builder.appImage.executableArgs,['--no-sandbox','--ozone-platform=x11']);assert.ok(builder.linux.executableArgs.includes('--ozone-platform=x11'));const main=fs.readFileSync(new URL('../main.cjs',import.meta.url),'utf8');const relaunch=main.indexOf("app.commandLine.hasSwitch('ozone-platform')");assert.ok(relaunch>0&&relaunch<main.indexOf('requestSingleInstanceLock'));assert.match(main,/execPath:process\.env\.APPIMAGE/);});
test('installers carry every local file main-process code requires, including the native-backend manifest',()=>{
  const builder=YAML.parse(fs.readFileSync(new URL('../electron-builder.yml',import.meta.url),'utf8')),dir=new URL('..',import.meta.url);
  const shipped=file=>builder.files.some(pattern=>pattern===file||(pattern.startsWith('*.')&&!file.includes('/')&&file.endsWith(pattern.slice(1)))||(pattern.endsWith('/**')&&file.startsWith(pattern.slice(0,-2))));
  const local=fs.readdirSync(dir).filter(f=>f.endsWith('.cjs')).flatMap(f=>[...fs.readFileSync(new URL(f,dir),'utf8').matchAll(/require\(\s*['"](\.\/[^'"]+)['"]\s*\)/g)].map(m=>path.posix.normalize(m[1])));
  assert.ok(local.includes('native_backends.json')&&local.includes('native_backends.cjs'));
  for(const file of local)assert.ok(shipped(file),file+' is required by main-process code but not packaged');
});
test('installers ship only what main-process files require, and every npm version is exact',()=>{
  // electron-builder packs each production dependency into app.asar and refuses electron or itself there; the renderer is bundled into dist/.
  const pkg=JSON.parse(fs.readFileSync(new URL('../package.json',import.meta.url),'utf8')),dir=new URL('..',import.meta.url),{builtinModules}=require('node:module');
  const required=new Set(fs.readdirSync(dir).filter(f=>f.endsWith('.cjs')).flatMap(f=>[...fs.readFileSync(new URL(f,dir),'utf8').matchAll(/require\(\s*['"]([^'"]+)['"]\s*\)/g)].map(m=>m[1])).filter(n=>!n.startsWith('.')&&n!=='electron'&&!n.startsWith('node:')&&!builtinModules.includes(n)));
  assert.deepEqual(Object.keys(pkg.dependencies).sort(),[...required].sort());
  for(const name of ['electron','electron-builder','vite'])assert.ok(pkg.devDependencies[name],name);
  for(const [name,version] of Object.entries({...pkg.dependencies,...pkg.devDependencies}))assert.match(version,/^\d+\.\d+\.\d+$/,name);
  assert.equal(pkg.devDependencies['electron-builder'],'26.0.12');
});
test('macOS asks for the microphone before the packaged backend starts, without holding it up',async()=>{
  const preferences=(status,answer=()=>Promise.resolve(true))=>{const calls=[];return {calls,getMediaAccessStatus:media=>{calls.push(['status',media]);return status;},askForMediaAccess:media=>{calls.push(['ask',media]);return answer();}};};
  const fresh=preferences('not-determined');assert.equal(await askMicrophone(fresh,'darwin'),true);assert.deepEqual(fresh.calls,[['status','microphone'],['ask','microphone']]);
  const granted=preferences('granted');assert.equal(await askMicrophone(granted,'darwin'),null);assert.deepEqual(granted.calls,[['status','microphone']]);
  for(const status of ['denied','restricted']){const known=preferences(status);assert.equal(await askMicrophone(known,'darwin'),'blocked');assert.deepEqual(known.calls,[['status','microphone']]);}
  for(const platform of ['win32','linux']){const other=preferences('not-determined');assert.equal(await askMicrophone(other,platform),null);assert.deepEqual(other.calls,[]);}
  assert.equal(await askMicrophone(preferences('not-determined',()=>Promise.resolve(false)),'darwin'),false);
  for(const fails of [()=>{throw new Error('no usage description');},()=>Promise.reject(new Error('denied'))])assert.equal(await askMicrophone(preferences('not-determined',fails),'darwin'),false);
  const host=fs.readFileSync(new URL('../main.cjs',import.meta.url),'utf8');
  assert.match(host.split('\n')[0],/nativeImage, systemPreferences \} = require\('electron'\);$/);
  const packaged=host.indexOf("const locator=path.join(app.getPath('userData'),'data-location.json');"),ask=host.indexOf('release.askMicrophone(systemPreferences).then(state=>microphoneNotice('),start=host.indexOf('backendProcess=release.startBackend(');
  assert.ok(packaged>0&&packaged<ask&&ask<start,'asked in the packaged branch, before the backend starts');
  assert.doesNotMatch(host,/await release\.askMicrophone/);
});
test('the macOS app is a hardened arm64 DMG whose processes may use the microphone and load the backend libraries',()=>{
  const builder=YAML.parse(fs.readFileSync(new URL('../electron-builder.yml',import.meta.url),'utf8')),mac=builder.mac;
  assert.deepEqual(mac.target,[{target:'dmg',arch:['arm64']}]);assert.equal(mac.hardenedRuntime,true);
  // One file for the app and everything inside it: the backend child opens the microphone and loads ad hoc-signed libraries.
  assert.equal(mac.entitlements,'build/entitlements.mac.plist');assert.equal(mac.entitlementsInherit,mac.entitlements);
  const entitlements=fs.readFileSync(new URL('../'+mac.entitlements,import.meta.url),'utf8');
  assert.deepEqual([...entitlements.matchAll(/<key>([^<]+)<\/key>\s*<true\/>/g)].map(match=>match[1]).sort(),['com.apple.security.cs.allow-jit','com.apple.security.cs.disable-library-validation','com.apple.security.device.audio-input']);
  assert.equal([...entitlements.matchAll(/<key>/g)].length,3);
  assert.match(mac.extendInfo.NSMicrophoneUsageDescription,/microphone/);
  assert.equal(mac.sign,'./build/mac_sign.cjs');assert.ok(fs.existsSync(new URL('.'+mac.sign.slice(1),new URL('../',import.meta.url))));
  assert.equal(mac.icon,'../release-stage/mac/icon.icns');
  assert.ok(MANIFEST.backends.find(backend=>backend.id==='metal').cmake_flags.includes('-DCMAKE_OSX_DEPLOYMENT_TARGET='+mac.minimumSystemVersion));
  assert.ok(!builder.files.some(pattern=>pattern.startsWith('build')),'build resources stay out of app.asar');
});
test('packaged apps cannot be relaunched as plain Node to borrow their microphone permission',()=>{const builder=YAML.parse(fs.readFileSync(new URL('../electron-builder.yml',import.meta.url),'utf8'));assert.deepEqual(builder.electronFuses,{runAsNode:false,enableNodeOptionsEnvironmentVariable:false,enableNodeCliInspectArguments:false});});
