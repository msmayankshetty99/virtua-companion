// The native-backend manifest (native_backends.json) names the bundled llama.cpp builds each OS ships, their library
// file names and build flags; the Python backend (Code/process/app_core/configuration/native_backends.py) and
// tools/release/build.py read the same file. Every function takes the platform, so tests cover each OS on any host.
const path=require('path');
const MANIFEST=require('./native_backends.json');
function libraryName(platform=process.platform,manifest=MANIFEST){if(!Object.hasOwn(manifest.platforms,platform))throw new Error('No native library is defined for '+platform);return manifest.platforms[platform].library;}
// Shipped for the platform, in preference order.
function backendsFor(platform=process.platform,manifest=MANIFEST){return manifest.backends.filter(backend=>backend.platforms.includes(platform));}
function nativeLibrary(resources,backend,platform=process.platform,manifest=MANIFEST){
 if(!backendsFor(platform,manifest).some(item=>item.id===backend))throw new Error('No '+backend+' native backend ships for '+platform);
 return path.join(resources,'native',backend,libraryName(platform,manifest));
}
// detected: {backend id: detection output}. The first shipped backend that needs no detection or whose detection found hardware.
function preferredBackend(platform=process.platform,detected={},manifest=MANIFEST){return backendsFor(platform,manifest).find(backend=>!backend.detect?.required||detected[backend.id])?.id||null;}
module.exports={MANIFEST,libraryName,backendsFor,nativeLibrary,preferredBackend};
