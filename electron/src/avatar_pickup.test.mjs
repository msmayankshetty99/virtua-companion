import test from 'node:test';
import assert from 'node:assert/strict';
import {Group,Bone,Vector3} from 'three';
import {AvatarPickup} from './avatar_pickup.mjs';
import {boneCatalog} from './avatar_secondary.mjs';
import {normalizeProfile} from './avatar_studio_settings.mjs';
test('feet pickup keeps its anchor, swings under gravity, triggers ragdoll once and recovers',()=>{
 const scene=new Group(),foot=new Bone(),hips=new Bone(),head=new Bone();foot.name='foot';hips.position.y=1;head.position.y=1;scene.add(foot);foot.add(hips);hips.add(head);
 const vrm={scene,humanoid:{getRawBoneNode:()=>foot}},events=[],pickup=new AvatarPickup(vrm,boneCatalog(vrm,'fixture'),kind=>events.push(kind)),settings=normalizeProfile({pickup:{ragdollEnabled:true}}).pickup;
 pickup.start('leftFoot',{x:0,y:0});pickup.move({x:500,y:0},1,settings);pickup.move({x:600,y:0},1,settings);assert.deepEqual(events,['ragdoll']);
 for(let i=0;i<300;i++){pickup.restore();pickup.step(settings,1/60);assert.ok(foot.getWorldPosition(new Vector3()).distanceTo(pickup.anchor)<1e-6);assert.ok(Number.isFinite(pickup.angle));}
 assert.ok(Math.abs(pickup.angle)>1);pickup.release();assert.deepEqual(events,['ragdoll','recover']);
 for(let i=0;i<200;i++){pickup.restore();pickup.step(settings,1/60);}assert.equal(pickup.active,false);assert.ok(scene.position.length()<1e-6);
});
test('ragdoll thresholds and disable switch are respected',()=>{
 const scene=new Group(),bone=new Bone();scene.add(bone);const vrm={scene,humanoid:{getRawBoneNode:()=>bone}},events=[],pickup=new AvatarPickup(vrm,boneCatalog(vrm,'fixture'),e=>events.push(e));
 const settings=normalizeProfile({pickup:{ragdollEnabled:true,distanceThreshold:1000,speedThreshold:10000}}).pickup;
 pickup.start('head',{x:0,y:0});pickup.move({x:100,y:0},1,settings);assert.deepEqual(events,[]);
 pickup.move({x:2000,y:0},1,{...settings,ragdollEnabled:false});assert.deepEqual(events,[]);
 pickup.step({...settings,enabled:false},.1);assert.equal(pickup.active,false);
});
test('a VRM 0.x root turned by PI keeps facing the camera while held and after recovery',()=>{
 const scene=new Group(),feet=new Bone(),chest=new Bone(),head=new Bone();chest.position.set(0,1,.05);head.position.set(0,.5,.05);scene.add(feet);feet.add(chest);chest.add(head);
 const vrm={scene,humanoid:{getRawBoneNode:()=>chest}},pickup=new AvatarPickup(vrm,boneCatalog(vrm,'fixture')),settings=normalizeProfile({}).pickup;
 const facing=()=>new Vector3(0,0,-1).applyQuaternion(scene.quaternion).z;  // a VRM 0.x model faces -z, so turned by PI it faces +z
 const frame=(time,held)=>{pickup.restore();scene.rotation.y=Math.PI+Math.sin(time*.7)*.025;if(held)pickup.move({x:10*time,y:0},1/60,settings);pickup.step(settings,1/60);};
 scene.rotation.y=Math.PI;pickup.start('chest',{x:0,y:0});
 for(let i=0;i<120;i++){frame(i/60,true);assert.ok(facing()>0,`frame ${i} faces away while held`);}
 pickup.release();for(let i=120;i<400&&pickup.active;i++){frame(i/60,false);assert.ok(facing()>0,`frame ${i} faces away while recovering`);}
 assert.equal(pickup.active,false);scene.rotation.y=Math.PI;assert.ok(facing()>.99);  // the renderer's next yaw write leaves her facing front
});
test('by default a held avatar bounces after a drag but never folds: no ragdoll, joints within their limit',()=>{
 // A VRoid-like chain: the spine and neck lean forward a little, the arms point sideways (the joints gravity bends most).
 const scene=new Group(),feet=new Bone(),hips=new Bone(),chest=new Bone(),neck=new Bone(),head=new Bone(),arm=new Bone(),hand=new Bone();
 hips.position.set(0,.9,0);chest.position.set(0,.3,.02);neck.position.set(0,.25,.04);head.position.set(0,.12,.04);arm.position.set(.2,.2,0);hand.position.set(.5,0,0);
 scene.add(feet);feet.add(hips);hips.add(chest);chest.add(neck,arm);neck.add(head);arm.add(hand);
 const vrm={scene,humanoid:{getRawBoneNode:()=>chest}},events=[],pickup=new AvatarPickup(vrm,boneCatalog(vrm,'fixture'),kind=>events.push(kind)),settings=normalizeProfile({}).pickup;
 assert.equal(settings.ragdollEnabled,false);const rest=neck.quaternion.clone(),after=[];let worst=0;pickup.start('chest',{x:0,y:0});
 for(let i=0;i<150;i++){pickup.restore();if(i<60)pickup.move({x:15*i,y:0},1/60,settings);pickup.step(settings,1/60);  // drag 900 px for 1 s, then hold still
  worst=Math.max(worst,...[...pickup.joints.values()].map(state=>state.angle.length()*180/Math.PI));if(i>=60)after.push(neck.quaternion.angleTo(rest)*180/Math.PI);}
 assert.deepEqual(events,[]);assert.ok(worst<=settings.jointLimit+1e-6,`a joint bent ${worst.toFixed(1)} degrees`);
 assert.ok(Math.max(...after)-Math.min(...after)>12,'the neck barely moved once the drag stopped');  // the earlier, stiff defaults: 7.5
});
