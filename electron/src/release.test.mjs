import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createRequire} from 'node:module';
import {EventEmitter} from 'node:events';
const require=createRequire(import.meta.url);
const {configuration,saveSetup,installRoot,startSovits,stopBackend,watchListening,finderPath,LISTENING}=require('../release.cjs');
const YAML=require('yaml');
function fixture(){const root=fs.mkdtempSync(path.join(os.tmpdir(),'riko-release-'));const folder=path.join(root,'native','cuda');fs.mkdirSync(folder,{recursive:true});fs.writeFileSync(path.join(folder,process.platform==='win32'?'riko-native.dll':'libriko-native.so'),'test');return root;}
const form={backend:'cuda',repo:'owner/model',filename:'model.gguf',context:8192,output:1024,threads:4,memories:'I like tea.\nMy name is Alex.',sovitsAuto:false};
test('setup separates persistent data from binaries and refuses overwrite',()=>{const root=fixture();const directory=fs.mkdtempSync(path.join(os.tmpdir(),'riko-data-'));try{saveSetup(directory,form,root);const config=YAML.parse(fs.readFileSync(path.join(directory,'character_config.yaml'),'utf8'));assert.equal(config.runtime.native_library,'bundled:cuda');assert.equal(config.memory.default_memories.length,2);assert.equal(config.tools.require_approval,true);assert.equal(config.emotion.probe.enabled,false);assert.deepEqual(config.voice,{asr_device:'cpu',asr_compute_type:'int8'});assert.ok(fs.existsSync(path.join(directory,'models')));assert.throws(()=>saveSetup(directory,form,root),/EEXIST/);assert.throws(()=>saveSetup(root,form,root),/outside/);}finally{fs.rmSync(root,{recursive:true,force:true});fs.rmSync(directory,{recursive:true,force:true});}});
test('setup validates budgets, model and explicit speech consent',()=>{const root=fixture();try{assert.throws(()=>configuration({...form,output:8192},root),/budget/);assert.throws(()=>configuration({...form,filename:'../secret.gguf'},root),/GGUF/);assert.throws(()=>configuration({...form,sovitsAuto:true},root),/GPT-SoVITS/);assert.equal(startSovits({auto_start:false}),null);assert.throws(()=>startSovits({auto_start:true,executable:'relative.exe'}),/missing/);}finally{fs.rmSync(root,{recursive:true,force:true});}});
test('setup leaves room for the reply inside the live context, as Settings requires',()=>{const root=fixture();try{for(const [context,output] of [[8192,1024],[2048,2047],[131072,64]]){const {runtime,memory}=configuration({...form,context,output},root);assert.equal(runtime.n_ctx,context);assert.equal(runtime.max_output_tokens,output);assert.ok(memory.context_window_tokens>=1);assert.ok(memory.context_window_tokens+runtime.max_output_tokens<=runtime.n_ctx);}}finally{fs.rmSync(root,{recursive:true,force:true});}});
test('setup refuses a data folder that an update or uninstall would delete',()=>{
  const install=fs.mkdtempSync(path.join(os.tmpdir(),'riko-install-')),resources=path.join(install,'resources'),sibling=install+'-data';
  fs.mkdirSync(path.join(resources,'native','cuda'),{recursive:true});fs.writeFileSync(path.join(resources,'native','cuda',process.platform==='win32'?'riko-native.dll':'libriko-native.so'),'test');
  try{
    for(const directory of [install,path.join(install,'data'),path.join(install,'data','nested'),resources])assert.throws(()=>saveSetup(directory,form,resources,install),/outside the application folder/);
    assert.ok(!fs.existsSync(path.join(install,'data')));
    assert.equal(saveSetup(sibling,form,resources,install),sibling);assert.ok(fs.existsSync(path.join(sibling,'character_config.yaml')));
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
