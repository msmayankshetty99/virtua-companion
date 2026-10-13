import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {PoseInterpolator,MotionEngine,proceduralPose,gesturePose,flipVRM0,BODY_BONES} from './motion_engine.mjs';
import {DesktopWalker,nearbyPointer} from './motion_geometry.mjs';
import {MotionAssetSampler} from './motion_assets.mjs';

test('interrupted transition starts at current displayed pose',()=>{
  const q=new THREE.Quaternion(),target=new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,1,0),1);
  const blend=new PoseInterpolator(new Map([['head',q]]));
  blend.step(new Map([['head',target]]),'first',.1,.4);
  const before=blend.displayed.get('head').clone();
  const after=blend.step(new Map([['head',q]]),'second',0,.4).get('head');
  assert.ok(before.angleTo(after)<1e-6);
  for(let i=0;i<10;i++)blend.step(new Map([['head',q]]),'second',.1,.4);
  assert.ok(blend.displayed.get('head').angleTo(q)<1e-6);
});

test('walker clamps bounds, completes once, and hold cancels',()=>{
  const events=[],walker=new DesktopWalker((...args)=>events.push(args));
  const geometry={x:0,y:0,width:100,height:100,screen:0};
  const action={id:'walk',payload:{target:{x:2000,y:0},speed:240}};
  let result;
  for(let i=0;i<100;i++){const next=walker.step(action,geometry,{width:400,height:400},.1);if(next)result=next;}
  assert.equal(result.x,300);assert.equal(result.final,true);
  assert.equal(events.filter(e=>e[1]==='completed').length,1);
  walker.step({...action,id:'second'},geometry,{width:400,height:400},.1);
  walker.step({...action,id:'second'},geometry,{width:400,height:400},.1,true);
  assert.equal(walker.active,null);assert.equal(events.at(-1)[1],'cancelled');
});

test('pointer hysteresis avoids flickering at boundary',()=>{
  const rect={x:0,y:0,width:100,height:100};
  assert.equal(nearbyPointer(rect,{x:125,y:50},false).near,false);
  assert.equal(nearbyPointer(rect,{x:125,y:50},true).near,true);
});

test('procedural held and walking poses differ from idle',()=>{
  assert.ok(proceduralPose('held',1).get('leftUpperArm').angleTo(proceduralPose('idle',1).get('leftUpperArm'))>.1);
  assert.ok(proceduralPose('walking',1).get('leftUpperLeg').angleTo(proceduralPose('idle',1).get('leftUpperLeg'))>.1);
});

test('pose sampling respects masks and reports once',()=>{
  const sampler=new MotionAssetSampler({}),events=[];
  const asset={id:'pose',kind:'pose',mask:['head'],pose:{bones:{head:[.1,0,0],neck:[0,.1,0]},expressions:{happy:.5}}};
  const sample=sampler.sample(asset,0,'action',(...args)=>events.push(args));
  assert.deepEqual([...sample.bones.keys()],['head']);assert.equal(sample.expressions.get('happy'),.5);
  sampler.retain(new Set(['pose']));sampler.sample(asset,1,'action',(...args)=>events.push(args));
  assert.equal(events.length,1);sampler.dispose();
});

test('held authority suppresses imported skeletal overlays',()=>{
  const head=new THREE.Object3D(),arm=new THREE.Object3D();
  const vrm={humanoid:{getNormalizedBoneNode:name=>name==='head'?head:name==='leftUpperArm'?arm:null}};
  let sampled=0;
  const engine=new MotionEngine(vrm,{sample:()=>{sampled++;return null;},retain:()=>{},dispose:()=>{}});
  engine.update({actions:[{id:'base',kind:'motion.base',status:'running',payload:{procedural:'held',asset:{id:'clip'}}}],interaction:{held:true}},.1);
  assert.equal(sampled,0);engine.dispose();
});

test('late asset load is disposed and cannot become active',async()=>{
  let resolve;
  const sampler=new MotionAssetSampler({},{load:()=>new Promise(done=>{resolve=done;})});
  sampler.sample({id:'clip',kind:'vrma'},0,'action',()=>assert.fail('stale report'));
  await Promise.resolve();sampler.dispose();
  resolve({scene:new THREE.Scene(),userData:{}});
  await new Promise(done=>setImmediate(done));
  assert.equal(sampler.cache.size,0);
});

test('VRMA sampling retargets rotations, excludes translations and restores live pose',async()=>{
  const scene=new THREE.Scene(),head=new THREE.Object3D();head.name='Head';scene.add(head);
  const animation={duration:1,humanoidTracks:{rotation:new Map([['head',new THREE.QuaternionKeyframeTrack('head.quaternion',[0,1],[0,0,0,1,0,Math.sin(.5),0,Math.cos(.5)])]]),translation:new Map()},expressionTracks:{preset:new Map(),custom:new Map()},lookAtTrack:null};
  const vrm={scene,meta:{metaVersion:'1'},humanoid:{getNormalizedBoneNode:name=>name==='head'?head:null},expressionManager:{getExpression:()=>null},lookAt:null};
  const sampler=new MotionAssetSampler(vrm,{load:async()=>({scene:new THREE.Scene(),userData:{vrmAnimations:[animation]}})});
  const asset={id:'vrma',kind:'vrma',mask:['head'],loop:false,speed:1};
  const events=[];
  sampler.sample(asset,0,'action',(...args)=>events.push(args));
  await new Promise(done=>setImmediate(done));
  const pose=sampler.sample(asset,.5,'action',(...args)=>events.push(args));
  assert.ok(pose?.bones.get('head').angleTo(new THREE.Quaternion())>.4);
  assert.ok(head.quaternion.angleTo(new THREE.Quaternion())<1e-6);
  sampler.sample(asset,1.1,'action',(...args)=>events.push(args));
  sampler.sample(asset,1.2,'action',(...args)=>events.push(args));
  assert.equal(events.filter(e=>e[0]==='started').length,1);
  assert.equal(events.filter(e=>e[0]==='completed').length,1);
  sampler.dispose();
});

// A two-bone arm per side. VRM 1.0 faces +z, so her left arm points to +x; a VRM 0.x model faces -z and its left arm points to -x.
function armRig(version){
  const scene=new THREE.Group(),nodes=new Map(BODY_BONES.map(name=>[name,new THREE.Object3D()])),side=version==='0'?-1:1;
  for(const [arm,lower,sign] of [['leftUpperArm','leftLowerArm',side],['rightUpperArm','rightLowerArm',-side]]){nodes.get(lower).position.x=.25*sign;nodes.get(arm).add(nodes.get(lower));scene.add(nodes.get(arm));}
  return {scene,nodes,vrm:{scene,meta:{metaVersion:version},humanoid:{getNormalizedBoneNode:name=>nodes.get(name)||null}}};
}
const still={sample:()=>null,retain:()=>{},dispose:()=>{}};
const armDirection=(nodes,arm,lower)=>{nodes.get(arm).updateMatrixWorld(true);return nodes.get(lower).getWorldPosition(new THREE.Vector3()).sub(nodes.get(arm).getWorldPosition(new THREE.Vector3())).normalize();};

test('canonical poses are flipped once for VRM 0.x and written unchanged for VRM 1.0, so idle arms hang down on both',()=>{
  for(const version of ['0','1']){
    const {nodes,vrm}=armRig(version),engine=new MotionEngine(vrm,still);
    for(let i=0;i<20;i++)engine.update({},.05);
    for(const bone of ['leftUpperArm','rightUpperArm','head']){
      const canonical=proceduralPose('idle',engine.time).get(bone),expected=version==='0'?flipVRM0(canonical):canonical;
      assert.ok(nodes.get(bone).quaternion.angleTo(expected)<1e-6,`${version} ${bone}`);
    }
    for(const [arm,lower] of [['leftUpperArm','leftLowerArm'],['rightUpperArm','rightLowerArm']])assert.ok(armDirection(nodes,arm,lower).y<-.8,`VRM ${version} ${arm} is raised`);
    engine.dispose();
  }
  const q=new THREE.Quaternion(.1,.2,.3,.927).normalize();assert.ok(flipVRM0(flipVRM0(q)).angleTo(q)<1e-9);
});

test('gestures keep their direction on VRM 0.x: a wave raises the right hand above the shoulder',()=>{
  for(const version of ['0','1']){
    const {nodes,vrm}=armRig(version),engine=new MotionEngine(vrm,still),wave=[{id:'wave',kind:'gesture',status:'running',duration:2,payload:{name:'wave',intensity:.8}}];
    for(let i=0;i<12;i++)engine.update({actions:wave},.05);
    assert.ok(armDirection(nodes,'rightUpperArm','rightLowerArm').y>.3,`VRM ${version} wave lowers the arm`);
    const expected=gesturePose('wave',engine.ages.get('wave'),new Map(),.8).get('rightUpperArm');
    assert.ok(nodes.get('rightUpperArm').quaternion.angleTo(version==='0'?flipVRM0(expected):expected)<1e-6);
    engine.dispose();
  }
});

test('a .pose.json is canonical on VRM 0.x and its expression targets go to the face layer, never to the manager',()=>{
  const {nodes,vrm}=armRig('0');vrm.expressionManager={setValue:()=>assert.fail('MotionEngine wrote an expression'),getValue:()=>0};
  const sampler=new MotionAssetSampler(vrm),asset={id:'pose',kind:'pose',mask:['head'],pose:{bones:{head:[.3,0,.2]},expressions:{happy:.5}}};
  const engine=new MotionEngine(vrm,sampler),actions=[{id:'preview',kind:'motion.preview',status:'running',payload:{asset}}];
  for(let i=0;i<20;i++)engine.update({actions},.05);
  assert.ok(nodes.get('head').quaternion.angleTo(flipVRM0(new THREE.Quaternion().setFromEuler(new THREE.Euler(.3,0,.2,'XYZ'))))<1e-6);
  assert.equal(engine.expressionTargets.get('happy'),.5);engine.dispose();
});

test('a .vrma clip on VRM 0.x reaches its node native, never flipped twice',async()=>{
  const scene=new THREE.Scene(),head=new THREE.Object3D();head.name='Head';scene.add(head);
  const q=new THREE.Quaternion(.3,.2,.4,.84).normalize(),native=flipVRM0(q),values=[...q.toArray(),...q.toArray()];
  const animation={duration:1,humanoidTracks:{rotation:new Map([['head',new THREE.QuaternionKeyframeTrack('head.quaternion',[0,1],values)]]),translation:new Map()},expressionTracks:{preset:new Map(),custom:new Map()},lookAtTrack:null};
  const vrm={scene,meta:{metaVersion:'0'},humanoid:{getNormalizedBoneNode:name=>name==='head'?head:null},expressionManager:{getExpression:()=>null},lookAt:null};
  const sampler=new MotionAssetSampler(vrm,{load:async()=>({scene:new THREE.Scene(),userData:{vrmAnimations:[animation]}})});
  const engine=new MotionEngine(vrm,sampler),actions=[{id:'preview',kind:'motion.preview',status:'running',payload:{asset:{id:'vrma',kind:'vrma',mask:['head'],loop:true,speed:1}}}];
  engine.update({actions},.05);await new Promise(done=>setImmediate(done));
  for(let i=0;i<20;i++)engine.update({actions},.05);
  assert.ok(head.quaternion.angleTo(native)<1e-5,'the clip is not where createVRMAnimationClip puts it');
  assert.ok(head.quaternion.angleTo(q)>.5,'the clip was flipped twice');
  engine.dispose();
});
