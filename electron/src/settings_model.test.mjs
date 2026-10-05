import assert from 'node:assert/strict';
import {test} from 'node:test';
import {parseSetting,settingsPatch,inputValues,llamaServerCommand} from './settings_model.mjs';

test('numeric inputs validate immediately and do not coerce blanks to zero',()=>{
  const field={kind:'number',integer:true,min:2,max:4};
  assert.deepEqual(parseSetting(field,'3'),{value:3});
  for(const value of ['','NaN','1','5','2.5'])assert.ok(parseSetting(field,value).error);
});
test('nullable automatic fields and JSON drafts are typed correctly',()=>{
  assert.deepEqual(parseSetting({kind:'number',nullable:true},''),{value:null});
  assert.deepEqual(parseSetting({kind:'json'},'[1, 2]'),{value:[1,2]});
  assert.ok(parseSetting({kind:'json'},'[invalid]').error);
});
test('only changed values are submitted and boolean false remains false',()=>{
  const fields=[{path:'enabled',kind:'boolean'},{path:'count',kind:'number',integer:true}];
  assert.deepEqual(settingsPatch(fields,{enabled:true,count:2},{enabled:false,count:'2'}),{changes:{enabled:false},errors:{}});
});
test('native cross-field validation enforces physical batch and V-cache constraints',()=>{
  const values={'runtime.provider':'llama_cpp','runtime.model_path':'model.gguf','runtime.n_batch':128,'runtime.n_ubatch':256,'runtime.type_v':'q8_0','runtime.flash_attn':false};
  const result=settingsPatch([],{},values);
  assert.ok(result.errors['runtime.n_ubatch']);assert.ok(result.errors['runtime.type_v']);
});
test('form hydration preserves zero, false and JSON arrays',()=>{
  const values={zero:0,flag:false,split:[1,1]};
  const fields=[{path:'zero',kind:'number'},{path:'flag',kind:'boolean'},{path:'split',kind:'json'}];
  const inputs=inputValues({values,fields});
  assert.equal(inputs.zero,0);assert.equal(inputs.flag,false);assert.deepEqual(JSON.parse(inputs.split),[1,1]);
});

test('the llama-server command matches what the backend checks: largest budget per slot, configured port',()=>{
  assert.equal(llamaServerCommand({'runtime.parallel_slots':2,'runtime.n_ctx':'6144','initiative.context_window_tokens':8192,'memory.reflection_context_window_tokens':4096,'runtime.base_url':'http://127.0.0.1:8081/v1'}),
    'llama-server -m model.gguf --port 8081 --parallel 2 --ctx-size 16384 --jinja');
  assert.equal(llamaServerCommand({'runtime.parallel_slots':'3','runtime.n_ctx':16384,'runtime.base_url':'not a url'}), 'llama-server -m model.gguf --port 8080 --parallel 3 --ctx-size 49152 --jinja');
});
