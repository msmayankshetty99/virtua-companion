import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const {configuration,saveSetup,startSovits,watchListening,LISTENING}=require('../release.cjs');
const YAML=require('yaml');
function fixture(){const root=fs.mkdtempSync(path.join(os.tmpdir(),'riko-release-'));const folder=path.join(root,'native','cuda');fs.mkdirSync(folder,{recursive:true});fs.writeFileSync(path.join(folder,process.platform==='win32'?'riko-native.dll':'libriko-native.so'),'test');return root;}
const form={backend:'cuda',repo:'owner/model',filename:'model.gguf',context:8192,output:1024,threads:4,memories:'I like tea.\nMy name is Alex.',sovitsAuto:false};
test('setup separates persistent data from binaries and refuses overwrite',()=>{const root=fixture();const directory=fs.mkdtempSync(path.join(os.tmpdir(),'riko-data-'));try{saveSetup(directory,form,root);const config=YAML.parse(fs.readFileSync(path.join(directory,'character_config.yaml'),'utf8'));assert.equal(config.runtime.native_library,'bundled:cuda');assert.equal(config.memory.default_memories.length,2);assert.equal(config.tools.require_approval,true);assert.equal(config.emotion.probe.enabled,false);assert.ok(fs.existsSync(path.join(directory,'models')));assert.throws(()=>saveSetup(directory,form,root),/EEXIST/);assert.throws(()=>saveSetup(root,form,root),/outside/);}finally{fs.rmSync(root,{recursive:true,force:true});fs.rmSync(directory,{recursive:true,force:true});}});
test('setup validates budgets, model and explicit speech consent',()=>{const root=fixture();try{assert.throws(()=>configuration({...form,output:8192},root),/budget/);assert.throws(()=>configuration({...form,filename:'../secret.gguf'},root),/GGUF/);assert.throws(()=>configuration({...form,sovitsAuto:true},root),/GPT-SoVITS/);assert.equal(startSovits({auto_start:false}),null);assert.throws(()=>startSovits({auto_start:true,executable:'relative.exe'}),/missing/);}finally{fs.rmSync(root,{recursive:true,force:true});}});
test('main learns the backend holds its port only from a complete marker line',()=>{let calls=0;const watch=watchListening(()=>calls++);watch('Riko AI server: http://127.0.0.1:8765\nRIKO_BACKEND_');assert.equal(calls,0);watch('LISTENING_NOT\n');assert.equal(calls,0);watch(LISTENING+'\r\nINFO later\n');assert.equal(calls,1);watch(LISTENING+'\n');assert.equal(calls,1);const split=watchListening(()=>calls++);for(const part of ['RIKO_BACK','END_LISTEN','ING','\n'])split(Buffer.from(part));assert.equal(calls,2);});
