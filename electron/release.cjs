const path=require('path');
const fs=require('fs');
const os=require('os');
const {execFile,spawn}=require('child_process');
const YAML=require('yaml');
const native=require('./native_backends.cjs');
const run=(file,args)=>new Promise(resolve=>execFile(file,args,{timeout:5000,windowsHide:true},(error,stdout)=>resolve(error?'':stdout.trim())));

// Runs only the detection tools of this OS's bundles (macOS ships Metal and has neither nvidia-smi nor vulkaninfo).
async function hardware(platform=process.platform,probe=run){
 const backends=native.backendsFor(platform),probed=backends.filter(backend=>backend.detect);
 const outputs=await Promise.all(probed.map(backend=>probe(backend.detect.command,backend.detect.args)));
 const detected=Object.fromEntries(probed.map((backend,index)=>[backend.id,outputs[index]]));
 return {cpu:os.cpus()[0]?.model||'Unknown',threads:os.cpus().length,ramGB:Math.round(os.totalmem()/2**30),platform,
  backends:backends.map(({id,label,description})=>({id,label,description})),
  detections:probed.map(backend=>({backend:backend.id,label:backend.detect.label,required:!!backend.detect.required,output:detected[backend.id]||backend.detect.missing})),
  preferred:native.preferredBackend(platform,detected)};
}
function configuration(input,resources,platform=process.platform){
 if(input.sovitsAuto&&(!path.isAbsolute(input.sovitsExecutable||'')||!fs.existsSync(input.sovitsExecutable)))throw new Error('Choose an existing absolute GPT-SoVITS executable');
 const shipped=native.backendsFor(platform);
 if(!shipped.some(backend=>backend.id===input.backend))throw new Error(shipped.length?'Choose '+shipped.map(backend=>backend.label).join(' or '):'No native backend ships for '+platform);
 const library=native.nativeLibrary(resources,input.backend,platform);
 if(!fs.existsSync(library))throw new Error('Packaged native backend is missing: '+library);
 const context=Number(input.context),output=Number(input.output),threads=Number(input.threads);
 if(!Number.isInteger(context)||context<2048||context>131072||!Number.isInteger(output)||output<64||output>=context)throw new Error('Invalid context/output budget');
 if(!Number.isInteger(threads)||threads<1||threads>1024)throw new Error('Invalid CPU thread count');
 if(!input.modelPath&&(!input.repo||!input.filename||!input.filename.endsWith('.gguf')||input.filename.includes('..')||input.filename.startsWith('/')))throw new Error('Choose a local GGUF or exact Hugging Face repository/file');
 if(input.modelPath&&(!path.isAbsolute(input.modelPath)||!input.modelPath.toLowerCase().endsWith('.gguf')||!fs.existsSync(input.modelPath)))throw new Error('Local GGUF does not exist');
 // The name may be several words, but the wake detector enrolls one: the wake name is the name's first word.
 const name=String(input.name||'').trim()||'Riko';
 return {runtime:{provider:'llama_cpp',native_library:'bundled:'+input.backend,model_path:input.modelPath||null,hf_repo_id:input.repo||null,hf_filename:input.filename||null,hf_revision:input.revision||'main',n_ctx:context,max_output_tokens:output,n_threads:threads,n_gpu_layers:input.cpuOnly?0:-1,parallel_slots:2,flash_attn:'auto',type_k:'f16',type_v:'f16',warmup:false},
  presets:{default:{name,system_prompt:input.prompt||'You are a helpful local companion.'}},
  memory:{context_window_tokens:context-output,default_memories:String(input.memories||'').split('\n').filter(t=>t.trim()).map(text=>({text,memory_type:'factual',importance:.8})),embeddings_enabled:!!input.embeddings,system1_enabled:!!input.julia,reflection_enabled:!!input.reflection},
  emotion:{enabled:!!input.julia,device:'cpu',probe:{enabled:false}},voice:{wake_word:name.split(/\s+/)[0],asr_device:'cpu',asr_compute_type:'int8'},tools:{require_approval:true},initiative:{enabled:false},desktop:{setup_on_startup_error:true},
  sovits_ping_config:{auto_start:!!input.sovitsAuto,executable:input.sovitsExecutable||null,arguments:[],url:input.sovitsUrl||'http://127.0.0.1:9880/tts',ref_audio_path:input.referenceAudio||'',prompt_text:input.referenceText||'',text_lang:'en',prompt_lang:'en',sample_rate:32000}};
}
// The folder an update replaces and an uninstall deletes (NSIS ends with RMDir /r $INSTDIR): the .app bundle on
// macOS, the executable's folder elsewhere.
function installRoot(executable,platform=process.platform){return platform==='darwin'?path.resolve(executable,'..','..','..'):path.dirname(executable);}
function inside(parent,child){const relative=path.relative(path.resolve(parent),path.resolve(child));return relative===''||relative!=='..'&&!relative.startsWith('..'+path.sep)&&!path.isAbsolute(relative);}
function saveSetup(directory,input,resources,install=resources,platform=process.platform){
 if(!path.isAbsolute(directory))throw new Error('Choose an absolute data directory');
 if([install,resources].some(root=>inside(root,directory)))throw new Error('Choose a data folder outside the application folder ('+path.resolve(install)+'); updates and uninstall delete everything inside it');
 const config=configuration(input,resources,platform);
 fs.mkdirSync(directory,{recursive:true});
 for(const folder of ['models','persistent_memories','logs'])fs.mkdirSync(path.join(directory,folder),{recursive:true});
 // Never overwrite an existing user configuration, even after a failed first run. The backend's PyYAML reads YAML 1.1,
 // so quote what it would not read as text (a companion named Yes or On would otherwise become a boolean).
 fs.writeFileSync(path.join(directory,'character_config.yaml'),YAML.stringify(config,{compat:'yaml-1.1'}),{flag:'wx',mode:0o600});
 return directory;
}
// The backend prints this line once it holds 127.0.0.1:8765. Main sends the API token only after it, so a
// process that grabbed the port while the backend was still starting never sees this start's token.
const LISTENING='RIKO_BACKEND_LISTENING';
function watchListening(onListening){
 let tail='';
 return chunk=>{
  if(tail===null)return;
  tail=(tail+String(chunk)).slice(-4096);
  if(tail.split(/\r?\n/).slice(0,-1).includes(LISTENING)){tail=null;onListening();}
 };
}
// Finder and the Dock start apps with launchd's PATH (/usr/bin:/bin:/usr/sbin:/sbin), which hides Homebrew's node,
// npx, uvx and ffmpeg from MCP servers and the Discord worker; put its prefixes first, as a Terminal does.
const HOMEBREW_PATHS=['/opt/homebrew/bin','/opt/homebrew/sbin','/usr/local/bin'];
function finderPath(current){const parts=String(current||'/usr/bin:/bin:/usr/sbin:/sbin').split(':').filter(Boolean);return [...HOMEBREW_PATHS.filter(dir=>!parts.includes(dir)),...parts].join(':');}
// macOS attributes the packaged backend's microphone use to this app: the hardened runtime needs the audio-input
// entitlement, and the prompt shows NSMicrophoneUsageDescription. Ask before the backend starts, so the prompt names Riko
// at launch instead of interrupting the first voice turn. Resolves true/false once answered, null when nothing was asked.
// 'blocked' when macOS already denies the microphone: the backend's input then opens but records silence.
function askMicrophone(preferences,platform=process.platform){
 if(platform!=='darwin')return Promise.resolve(null);
 const status=preferences.getMediaAccessStatus('microphone');
 if(status==='denied'||status==='restricted')return Promise.resolve('blocked');
 if(status!=='not-determined')return Promise.resolve(null);
 return Promise.resolve().then(()=>preferences.askForMediaAccess('microphone')).catch(()=>false);
}
function startBackend(directory,resources,secrets={},onListening=()=>{}){
 const executable=path.join(resources,'backend',process.platform==='win32'?'riko-backend.exe':'riko-backend');
 const logPath=path.join(directory,'logs','backend-launch.log');
 const log=fs.openSync(logPath,'a');
 // This start's API token and confirmation key go to the backend only; it removes them from its own environment.
 const own=secrets.api_token&&secrets.confirm_key?{RIKO_API_TOKEN:secrets.api_token,RIKO_CONFIRM_KEY:secrets.confirm_key}:{};
 const env={...process.env,...own,...(process.platform==='darwin'?{PATH:finderPath(process.env.PATH)}:{}),RIKO_MANAGED:'1',RIKO_DATA_DIR:directory,RIKO_CONFIG:path.join(directory,'character_config.yaml'),HF_HOME:path.join(directory,'models','huggingface'),TORCH_HOME:path.join(directory,'models','torch'),XDG_CACHE_HOME:path.join(directory,'models','cache'),RIKO_BUNDLE_ROOT:resources};
 let child;
 try{child=spawn(executable,[],{cwd:directory,env,stdio:['pipe','pipe',log],windowsHide:true});}finally{fs.closeSync(log);}
 child.stdin.on('error',()=>{});
 const output=fs.createWriteStream(logPath,{flags:'a'}),listening=watchListening(onListening);
 child.stdout.on('data',chunk=>{output.write(chunk);listening(chunk);});
 child.stdout.on('end',()=>output.end());
 return child;
}
function startSovits(settings){
 if(!settings?.auto_start)return null;
 const executable=settings.executable,args=settings.arguments||[];
 if(!executable||!path.isAbsolute(executable)||!fs.statSync(executable).isFile())throw new Error('GPT-SoVITS executable is missing');
 if(!Array.isArray(args)||args.some(a=>typeof a!=='string'))throw new Error('GPT-SoVITS arguments must be a list of strings');
 return spawn(executable,args,{cwd:path.dirname(executable),stdio:'ignore',windowsHide:true,shell:false});
}
// Ask first; the backend gives itself 14 s. uvicorn ignores SIGTERM while already shutting down, so escalate to SIGKILL.
function stopBackend(child,grace=15000,force=3000){return new Promise(resolve=>{if(!child||child.exitCode!==null||child.signalCode!==null){resolve();return;}let timer=setTimeout(()=>{child.kill();timer=setTimeout(()=>{child.kill('SIGKILL');resolve();},force);},grace);child.once('exit',()=>{clearTimeout(timer);resolve();});child.stdin.end('shutdown\n');});}
module.exports={hardware,configuration,saveSetup,installRoot,askMicrophone,startBackend,nativeLibrary:native.nativeLibrary,startSovits,stopBackend,watchListening,finderPath,LISTENING};
