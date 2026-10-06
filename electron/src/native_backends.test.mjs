import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const {MANIFEST,libraryName,backendsFor,nativeLibrary,preferredBackend}=require('../native_backends.cjs');
// Written out, not derived: each OS is checked on every host (tests/test_release_config.py holds the same table).
const LIBRARIES={win32:'riko-native.dll',linux:'libriko-native.so',darwin:'libriko-native.dylib'};
const SHIPPED={win32:['cuda','vulkan'],linux:['cuda','vulkan'],darwin:['metal']};

test('every OS names its native library and ships its own backends',()=>{
  assert.deepEqual(Object.keys(MANIFEST.platforms).sort(),Object.keys(LIBRARIES).sort());
  for(const [platform,library] of Object.entries(LIBRARIES)){
    assert.equal(libraryName(platform),library);
    assert.deepEqual(backendsFor(platform).map(backend=>backend.id),SHIPPED[platform]);
    for(const backend of SHIPPED[platform])assert.equal(nativeLibrary('/app/resources',backend,platform),path.join('/app/resources','native',backend,library));
    for(const backend of ['cuda','vulkan','metal'].filter(id=>!SHIPPED[platform].includes(id)))assert.throws(()=>nativeLibrary('/app/resources',backend,platform),new RegExp('No '+backend+' native backend ships for '+platform));
    for(const backend of ['../../other','','CUDA'])assert.throws(()=>nativeLibrary('/app/resources',backend,platform),/native backend ships/);
  }
  for(const platform of ['freebsd','constructor','__proto__'])assert.throws(()=>libraryName(platform),/No native library is defined/);
});

test('setup prefers CUDA only when nvidia-smi finds a GPU, and Metal on macOS',()=>{
  assert.equal(preferredBackend('linux',{cuda:'NVIDIA GeForce RTX 4090, 24564, 550.54.15',vulkan:''}),'cuda');
  assert.equal(preferredBackend('win32',{cuda:'',vulkan:'GPU0: AMD Radeon'}),'vulkan');
  assert.equal(preferredBackend('linux',{}),'vulkan');
  assert.equal(preferredBackend('darwin',{cuda:'NVIDIA'}),'metal');
  assert.equal(preferredBackend('freebsd',{}),null);
});

test('a new bundle is one manifest entry: HIP preferred when detected, CPU never preselected over a GPU backend',()=>{
  const manifest={platforms:MANIFEST.platforms,backends:[...MANIFEST.backends.slice(0,1),{id:'hip',label:'ROCm',description:'AMD',platforms:['linux'],detect:{command:'rocminfo',args:[],label:'ROCm',missing:'none',required:true}},...MANIFEST.backends.slice(1),{id:'cpu',label:'CPU',description:'no GPU',platforms:['win32','linux','darwin']}]};
  assert.deepEqual(backendsFor('linux',manifest).map(backend=>backend.id),['cuda','hip','vulkan','cpu']);
  assert.equal(preferredBackend('linux',{hip:'gfx1100'},manifest),'hip');
  assert.equal(preferredBackend('linux',{},manifest),'vulkan');
  assert.equal(preferredBackend('darwin',{},manifest),'metal');
  assert.equal(nativeLibrary('/r','cpu','darwin',manifest),path.join('/r','native','cpu','libriko-native.dylib'));
});
