const path=require('path');
const fs=require('fs');
const os=require('os');
const {execFile,spawn}=require('child_process');
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
// The folder an update replaces and an uninstall deletes (NSIS ends with RMDir /r $INSTDIR): the .app bundle on
// macOS, the executable's folder elsewhere.
function installRoot(executable,platform=process.platform){return platform==='darwin'?path.resolve(executable,'..','..','..'):path.dirname(executable);}
function inside(parent,child){const relative=path.relative(path.resolve(parent),path.resolve(child));return relative===''||relative!=='..'&&!relative.startsWith('..'+path.sep)&&!path.isAbsolute(relative);}
// The packaged backend. Main runs it to serve the app (startBackend) and, on the first run, to turn the setup page's choices
// into its configuration (saveSetup).
function backendExecutable(resources,platform=process.platform){return path.join(resources,'backend',platform==='win32'?'riko-backend.exe':'riko-backend');}
// One answer from the backend's command line (Code/run_server.py): input as JSON on stdin, one line of JSON on stdout.
// command is [file, ...arguments]; development and tests pass Python and Code/run_server.py.
function backendAnswer(command,args,input,env,timeout=120000){
 return new Promise((resolve,reject)=>{
  let child,output='',errors='';
  try{child=spawn(command[0],[...command.slice(1),...args],{env,stdio:['pipe','pipe','pipe'],windowsHide:true});}catch(error){reject(error);return;}
  const timer=setTimeout(()=>{child.kill();reject(new Error('The backend did not answer in time'));},timeout);
  child.stdout.setEncoding('utf8');child.stderr.setEncoding('utf8');
  child.stdout.on('data',chunk=>{output+=chunk;});child.stderr.on('data',chunk=>{errors=(errors+chunk).slice(-2000);});
  child.on('error',error=>{clearTimeout(timer);reject(new Error('The backend could not start: '+error.message));});
  child.on('close',code=>{
   clearTimeout(timer);let answer=null;
   try{answer=JSON.parse(output.trim().split(/\r?\n/).at(-1));}catch{}
   if(!answer||typeof answer!=='object')reject(new Error('The backend gave no answer (exit '+code+')'+(errors.trim()?': '+errors.trim():'')));
   else if(answer.error)reject(new Error(answer.error));else resolve(answer);
  });
  child.stdin.on('error',()=>{});child.stdin.end(JSON.stringify(input));
 });
}
// First run: the backend turns the choices into the configuration it will load, over what a packaged build starts with, and
// checks it as Settings checks an edit (Code/process/app_core/configuration/first_run.py). Main keeps the data folder's
// rules and writes the file, never over an existing one, even after a failed first run.
async function saveSetup(directory,input,resources,install=resources,platform=process.platform,command=[backendExecutable(resources,platform)]){
 if(!path.isAbsolute(directory))throw new Error('Choose an absolute data directory');
 if([install,resources].some(root=>inside(root,directory)))throw new Error('Choose a data folder outside the application folder ('+path.resolve(install)+'); updates and uninstall delete everything inside it');
 const file=path.join(directory,'character_config.yaml');
 if(fs.existsSync(file))throw Object.assign(new Error(`EEXIST: file already exists, open '${file}'`),{code:'EEXIST'});
 fs.mkdirSync(directory,{recursive:true});
 const {config}=await backendAnswer(command,['--setup-config',directory],input,{...process.env,RIKO_BUNDLE_ROOT:resources});
 if(typeof config!=='string'||!config.trim())throw new Error('The backend returned no configuration');
 for(const folder of ['models','persistent_memories','logs'])fs.mkdirSync(path.join(directory,folder),{recursive:true});
 fs.writeFileSync(file,config,{flag:'wx',mode:0o600});
 return directory;
}
// The backend (Code/run_server.py) prints listeningLine(port) once it holds 127.0.0.1:<port>. Main sends the API token only
// after the line names the port it gave the backend, so a process that grabbed the port while the backend was still
// starting never sees this start's token.
const LISTENING='RIKO_BACKEND_LISTENING';
const listeningLine=port=>LISTENING+' port='+port;
function watchListening(port,onListening){
 let tail='';const line=listeningLine(port);
 return chunk=>{
  if(tail===null)return;
  tail=(tail+String(chunk)).slice(-4096);
  if(tail.split(/\r?\n/).slice(0,-1).includes(line)){tail=null;onListening();}
 };
}
// A free loopback port for the packaged backend. Should another process take it before the backend binds it, the backend
// exits after its retries without ever naming it, and main shows 'Backend stopped'.
function freePort(net=require('net')){return new Promise((resolve,reject)=>{const server=net.createServer();server.unref();server.once('error',reject);server.listen(0,'127.0.0.1',()=>{const {port}=server.address();server.close(()=>resolve(port));});});}
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
function startBackend(directory,resources,secrets={},port,onListening=()=>{}){
 const executable=backendExecutable(resources);
 const logPath=path.join(directory,'logs','backend-launch.log');
 const log=fs.openSync(logPath,'a');
 // This start's API token and confirmation key go to the backend only; it removes them from its own environment.
 const own=secrets.api_token&&secrets.confirm_key?{RIKO_API_TOKEN:secrets.api_token,RIKO_CONFIRM_KEY:secrets.confirm_key}:{};
 const env={...process.env,...own,...(process.platform==='darwin'?{PATH:finderPath(process.env.PATH)}:{}),RIKO_MANAGED:'1',RIKO_PORT:String(port),RIKO_DATA_DIR:directory,RIKO_CONFIG:path.join(directory,'character_config.yaml'),HF_HOME:path.join(directory,'models','huggingface'),TORCH_HOME:path.join(directory,'models','torch'),XDG_CACHE_HOME:path.join(directory,'models','cache'),RIKO_BUNDLE_ROOT:resources};
 let child;
 try{child=spawn(executable,[],{cwd:directory,env,stdio:['pipe','pipe',log],windowsHide:true});}finally{fs.closeSync(log);}
 child.stdin.on('error',()=>{});
 const output=fs.createWriteStream(logPath,{flags:'a'}),listening=watchListening(port,onListening);
 child.stdout.on('data',chunk=>{output.write(chunk);listening(chunk);});
 child.stdout.on('end',()=>output.end());
 return child;
}
// Only a YAML boolean true starts it (launch_config.cjs reads the file as the backend does): a quoted 'no' is not consent.
function startSovits(settings){
 if(settings?.auto_start!==true)return null;
 const executable=settings.executable,args=settings.arguments||[];
 if(!executable||!path.isAbsolute(executable)||!fs.statSync(executable).isFile())throw new Error('GPT-SoVITS executable is missing');
 if(!Array.isArray(args)||args.some(a=>typeof a!=='string'))throw new Error('GPT-SoVITS arguments must be a list of strings');
 return spawn(executable,args,{cwd:path.dirname(executable),stdio:'ignore',windowsHide:true,shell:false});
}
// Ask first; the backend gives itself 14 s. uvicorn ignores SIGTERM while already shutting down, so escalate to SIGKILL.
function stopBackend(child,grace=15000,force=3000){return new Promise(resolve=>{if(!child||child.exitCode!==null||child.signalCode!==null){resolve();return;}let timer=setTimeout(()=>{child.kill();timer=setTimeout(()=>{child.kill('SIGKILL');resolve();},force);},grace);child.once('exit',()=>{clearTimeout(timer);resolve();});child.stdin.end('shutdown\n');});}
module.exports={hardware,saveSetup,backendExecutable,backendAnswer,installRoot,askMicrophone,startBackend,nativeLibrary:native.nativeLibrary,startSovits,stopBackend,watchListening,listeningLine,freePort,finderPath,LISTENING};
