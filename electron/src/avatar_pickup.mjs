import {Vector3,Quaternion,Box3,Matrix4} from 'three';
import {flipVRM0} from './motion_engine.mjs';
const D=Math.PI/180,G=9.8,TAU=2*Math.PI,SUBSTEPS=4,HIT_STOP=.08,LETGO=.35,SETTLE=1.678/.1,SET_DOWN=1.678/.06,SET_RAMP=.1,VIEW=new Vector3(0,0,1);  // s, s; critically damped rates for 0.1 s and 0.06 s half-lives; s
const clamp=(v,a,b)=>Math.max(a,Math.min(b,Number.isFinite(v)?v:0));
const smooth=u=>{u=clamp(u,0,1);return u*u*(3-2*u);};
const smoother=u=>{u=clamp(u,0,1);return u*u*u*(u*(6*u-15)+10);};  // zero speed and acceleration at both ends
const soft=(x,limit)=>{const knee=.6*limit,a=Math.abs(x);return a<=knee?x:Math.sign(x)*(knee+(limit-knee)*Math.tanh((a-knee)/(limit-knee)));};  // linear, then easing into the limit
/** Soft limits on the lag rolls relative to the parent (radians): a shake near the limbs' own frequency must never fold a hip, shoulder or neck.
 * An arm rolling in toward her body stops at armIn while it hangs at her side (it would pass through her torso: on Miku a 36-degree lag
 * put the elbow 45 mm inside it, 12 degrees still 7 mm, 6 degrees leaves it 20 mm clear) and gets the full arms limit once it is held
 * forward or raised clear of her. */
const ROLL={legs:25*D,arms:45*D,armIn:6*D,head:30*D};
/** The camera safety net's margin around the skeleton's joints (m): the skin, shoes and hair reach at most 4 cm past them on Miku (the
 * catalog has the hair and finger joints); 0.1 m made a brisk drag shrink her 12-14%, 0.04 m 6-7% (her legs swing past the 480 px canvas). */
const MARGIN=.04;
/** A second-order follower in frequency (Hz) and damping ratio: x'' = w^2 (target - x) - 2 zeta w x'. */
class Spring{
 constructor(){this.x=0;this.v=0;}
 step(target,frequency,damping,h){const w=TAU*frequency;this.v+=(w*w*(target-this.x)-2*damping*w*this.v)*h;this.x+=this.v*h;return this.x;}
 reset(){this.x=0;this.v=0;}
 get still(){return Math.abs(this.x)<.05*D&&Math.abs(this.v)<.5*D;}
}
/** One critically damped step of x toward target (exact for any h). Returns [x, v]. */
function settle(x,v,target,w,h){const d=x-target,j=v+w*d,e=Math.exp(-w*h);return [target+(d+j*h)*e,(v-j*w*h)*e];}

/** The carry rig. Held by the scruff, her body swings as one piece from the grab point like something with weight:
 * - anchor: the grab point S (her upper chest, or the grabbed bone with style 'bone') keeps its world position, and with style 'scruff'
 *   the window slides over 0.25-0.6 s (slide(), in NDC) so S comes under the cursor. The anchor moves the raw hips, never vrm.scene:
 *   VRoid springs are simulated relative to a centre node under vrm.scene, so a scene translation would hide the body's real motion
 *   from the hair (on Miku a brisk drag flung her twintails over her head);
 * - one acceleration-driven pendulum: theta'' = -w^2 sin(theta) - 2 zeta w theta' - (a_x w^2 / g) gain cos(theta) + w^2 lean, shown
 *   through a tanh soft limit and applied as a roll about the view axis on the normalized hips BEFORE humanoid.update and the springs,
 *   so the hair and bust swing with the body (on Miku a carry swings her twintails 38-55 degrees, trailing the body by 20-28; the old
 *   root swing after the springs moved them about 6);
 * - limb lag springs (legs and arms catch up toward world-down a beat late), head stabilisation (a counter-roll), and a lift spring
 *   (lifted fast she straightens and points her toes; lowered fast her knees come up);
 * - set down, the swing, limb lag and lift turn critically damped over 0.1 s (0.06 s half-life: upright inside the handover), and over
 *   0.35 s with no jolt the hold on S hands over to the feet, which plant on the line she stood on until recoverySeconds (held clear of
 *   it while they still swing in under her, so they never slide along the floor); then the translation settles to 0 with a 0.1 s half-life.
 * There is no per-bone gravity: the held pose is authored (avatar_held_pose.mjs). Only normalized humanoid bones and the raw hips
 * position change (humanoid.update rewrites that every frame; restore() puts it back too); vrm.scene, and so a VRM 0.x root turned by
 * PI, is never touched. */
export class AvatarPickup{
 constructor(vrm,catalog,onTrigger=()=>{}){
  this.vrm=vrm;this.catalog=catalog;this.onTrigger=onTrigger;this.vrm0=vrm.meta?.metaVersion==='0';
  this.active=false;this.held=false;this.ragdoll=false;this.saved=null;this.bone=null;this.restToe=null;
  this.theta=0;this.omega=0;this.displayed=0;this.legs=new Spring();this.arms=new Spring();this.head=new Spring();this.lift=new Spring();
  this.offset=new Vector3();this.offsetVelocity=new Vector3();this.slideTarget={x:0,y:0};this.slideTime=.25;this.age=0;this.releaseAge=0;this.h=0;
  this.camera=null;this.cameraRest=null;this.cameraActive=false;this.distance=0;
  this.q=new Quaternion();this.r=new Quaternion();this.v=new Vector3();this.anchored=new Vector3();this.previous=new Vector3();this.a=new Vector3();this.b=new Vector3();this.m=new Matrix4();this.bounds=new Box3();
 }
 /** The displayed swing angle (radians, counter-clockwise as the viewer sees it; positive swings her legs to screen right). */
 get angle(){return this.displayed;}
 /** bone: the grabbed humanoid bone; point: the press point in screen px (the ragdoll distance origin); pointer: the press point in
  * canvas NDC, which the scruff slide brings S to (default: the grabbed bone's projection). */
 start(bone,point,camera,{style='scruff',pointer=null}={}){
  const humanoid=this.vrm.humanoid,grabbed=humanoid.getRawBoneNode?.(bone)||null;
  const scruff=['upperChest','chest','spine','hips'].map(name=>humanoid.getRawBoneNode?.(name)).find(Boolean)||grabbed;
  this.style=style==='bone'&&grabbed?'bone':'scruff';this.bone=this.style==='bone'?grabbed:scruff;if(!this.bone)return;
  // The hips still carry last frame's offset here (the renderer handles pointer samples before restore()), so a grab during the
  // settle continues from where she is.
  this.vrm.scene.updateMatrixWorld(true);this.anchorPoint=this.bone.getWorldPosition(new Vector3());
  if(!this.active){this.restToe=this.toeY();this.restAside=this.aside();this.theta=this.omega=this.displayed=0;for(const spring of [this.legs,this.arms,this.head,this.lift])spring.reset();this.offset.set(0,0,0);this.offsetVelocity.set(0,0,0);}
  this.slideTarget={x:0,y:0};this.slideTime=.25;this.anchorNDC=null;
  if(camera){
   this.camera=camera;if(!this.cameraActive){this.cameraRest={position:camera.position.clone(),quaternion:camera.quaternion.clone()};this.distance=camera.position.z-this.anchorPoint.z;}
   this.anchorNDC=this.anchorPoint.clone().project(camera);
   const target=this.style==='scruff'?pointer||(grabbed&&grabbed!==this.bone?grabbed.getWorldPosition(new Vector3()).project(camera):null):null;
   if(target){
    const dx=target.x-this.anchorNDC.x,dy=target.y-this.anchorNDC.y,tan=Math.tan(camera.fov*D/2),depth=camera.position.distanceTo(this.anchorPoint);
    this.slideTarget={x:dx,y:dy};this.slideTime=Math.min(.6,.25+.25*Math.hypot(dx*tan*camera.aspect*depth,dy*tan*depth)/.5);  // 0.25 s plus 0.25 s per 0.5 m
   }
  }
  this.origin={...point};this.last={...point};this.slow=0;this.active=true;this.held=true;this.ragdoll=false;this.age=0;this.releaseAge=0;
 }
 /** Pointer samples while held: only the opt-in ragdoll thresholds. */
 move(point,dt,settings){if(!this.active||!this.held)return;
  const speed=Math.hypot(point.x-this.last.x,point.y-this.last.y)/Math.max(.008,dt);this.last={...point};
  if(settings.ragdollEnabled&&!this.ragdoll&&(Math.hypot(point.x-this.origin.x,point.y-this.origin.y)>settings.distanceThreshold||speed>settings.speedThreshold)){this.ragdoll=true;this.slow=0;this.onTrigger('ragdoll');}
 }
 release(){if(!this.active||!this.held)return;this.held=false;this.releaseAge=0;if(this.ragdoll){this.ragdoll=false;this.onTrigger('recover');}}
 restore(){if(!this.saved)return;this.saved.node.position.copy(this.saved.position);this.saved=null;}
 /** The canvas offset (NDC) that brings S under the cursor; the renderer adds it to the drag geometry every frame. */
 slide(){if(!this.active)return {x:0,y:0};const u=smooth(this.age/this.slideTime);return {x:this.slideTarget.x*u,y:this.slideTarget.y*u};}
 /** How far (m) her feet are to the side of S: about 0 standing, up to L sin(angle) while her body or legs swing. */
 aside(){
  const humanoid=this.vrm.humanoid;let x=0,n=0;
  for(const side of ['left','right']){const node=humanoid.getRawBoneNode?.(side+'Toes')||humanoid.getRawBoneNode?.(side+'Foot');if(node){x+=node.getWorldPosition(this.v).x;n++;}}
  return this.bone&&n?Math.abs(x/n-this.bone.getWorldPosition(this.v).x):0;
 }
 toeY(){
  const humanoid=this.vrm.humanoid,ys=[];
  for(const side of ['left','right']){const node=humanoid.getRawBoneNode?.(side+'Toes')||humanoid.getRawBoneNode?.(side+'Foot');if(node)ys.push(node.getWorldPosition(this.v).y);}
  return ys.length?Math.min(...ys):null;
 }
 /** Rolls a normalized bone about the world view axis through its joint: its parent's frame decides the local axis. */
 roll(node,angle){
  if(!node||!(Math.abs(angle)>1e-7))return;
  if(node.parent)node.parent.getWorldQuaternion(this.q);else this.q.identity();
  node.quaternion.premultiply(this.r.setFromAxisAngle(this.v.copy(VIEW).applyQuaternion(this.q.invert()),angle));
 }
 /** Adds a canonical x rotation (degrees) in a normalized bone's own frame: knee bend, toe point, hip flexion (x-). */
 bend(node,degrees){if(!node||!degrees)return;const q=this.r.setFromAxisAngle(this.v.set(1,0,0),degrees*D);node.quaternion.multiply(this.vrm0?flipVRM0(q):q);}
 /** After MotionEngine, before humanoid.update. motion: window acceleration and velocity (px/s^2, px/s) from AvatarInertia, the
  * world size of a pixel, and the hold (HoldBehaviour's step): dizzy or limp makes the body limp and the head unsteady. */
 pose(settings,dt,{ax=0,ay=0,vx=0,vy=0,worldPerPixel=0,hold=null}={}){
  if(!this.active)return;if(!settings.enabled){this.stop();return;}
  const h=clamp(dt,0,.1);this.h=h;this.age+=h;if(!this.held)this.releaseAge+=h;
  // The opt-in ragdoll recovers once the window has stayed under the speed threshold for 1 s (or on release).
  if(this.ragdoll){this.slow=Math.hypot(vx,vy)<settings.speedThreshold?this.slow+h:0;if(this.slow>=1||!settings.ragdollEnabled){this.ragdoll=false;this.origin={...this.last};this.onTrigger('recover');}}
  const limp=this.ragdoll?'ragdoll':hold?.state==='dizzy'||hold?.pose?.limp>.5?'limp':'';  // the first 0.08 s after the grab: a hit-stop
  const node=name=>this.vrm.humanoid.getNormalizedBoneNode?.(name)||null,neck=node('neck');
  // Set down (released), the dynamics turn critically damped toward upright over the first 0.1 s (r 0 to 1), and the limbs stop chasing
  // the body and settle on their own, so the swing, limb lag and lift settle from where they are with no jolt and no rocking on planted
  // feet: upright inside the 0.35 s handover. A flight (later) keeps the held dynamics instead.
  const r=this.held?0:smoother(this.releaseAge/SET_RAMP),mix=(held,down)=>held+(down-held)*r,fDown=SET_DOWN/TAU;
  const w=TAU*mix(settings.swingFrequency,fDown),zeta=mix(settings.swingDamping,1),limit=settings.swingLimit*D,carry=(1-r)*(this.age>HIT_STOP?1:0);
  const a=carry*clamp(ax*worldPerPixel,-3*G,3*G)*settings.carryGain,lean=-carry*clamp(vx/1000,-1,1)*settings.airLean*D,lift=(1-r)*clamp(-ay*worldPerPixel/G,-1,1);
  const frequency=mix(settings.limbFrequency*(limp==='ragdoll'?.4:limp?.6:1),fDown),damping=mix(limp==='ragdoll'?.15:settings.limbDamping,1),steady=limp?0:settings.headStabilize;
  const shown=theta=>limit>0?limit*Math.tanh(theta/limit):0,step=h/SUBSTEPS;
  for(let i=0;i<SUBSTEPS;i++){
   this.omega+=(-w*w*Math.sin(this.theta)-2*zeta*w*this.omega-(a*w*w/G)*Math.cos(this.theta)+w*w*lean)*step;
   this.theta+=this.omega*step;if(Math.abs(this.theta)>1.4){this.theta=Math.sign(this.theta)*1.4;this.omega=0;}  // never over the top
   const angle=(1-r)*shown(this.theta);
   this.legs.step(-.85*angle,frequency,damping,step);this.arms.step(-.9*angle,mix(frequency*.83,fDown),mix(damping*.85,1),step);this.head.step(-steady*angle,mix(3,fDown),mix(.7,1),step);this.lift.step(lift,mix(2.5,fDown),mix(.5,1),step);
  }
  this.displayed=shown(this.theta);
  const legs=soft(this.legs.x,ROLL.legs),head=soft(this.head.x,ROLL.head);
  // Per arm: the roll toward her body (negative for the arm on screen right) is limited by how clear of her torso the arm is.
  const hips=node('hips')?.getWorldPosition(this.a).x??0,arm=side=>{
   const upper=node(side+'UpperArm'),lower=node(side+'LowerArm'),x=this.arms.x;if(!upper||!lower)return soft(x,ROLL.arms);
   const at=upper.getWorldPosition(this.b),d=lower.getWorldPosition(this.v).sub(at).normalize(),inward=(at.x>hips?-1:1)*x>0;
   return soft(x,inward?ROLL.armIn+(ROLL.arms-ROLL.armIn)*smooth(Math.max(d.z,d.y)/.45):ROLL.arms);
  },arms={left:arm('left'),right:arm('right')};
  this.roll(node('hips'),this.displayed);
  for(const side of ['left','right']){this.roll(node(side+'UpperLeg'),legs);this.roll(node(side+'UpperArm'),arms[side]);}
  this.roll(neck,.4*head);this.roll(node('head'),(neck?.6:1)*head);
  const up=this.lift.x;
  if(Math.abs(up)>1e-3)for(const side of ['left','right']){
   const knee=node(side+'LowerLeg');
   if(up>0){const bent=knee?2*Math.acos(Math.min(1,Math.abs(knee.quaternion.w)))/D:0;this.bend(knee,-Math.min(10*up,bent));this.bend(node(side+'Foot'),10*up);}  // straighten, never past straight
   else{this.bend(node(side+'UpperLeg'),10*up);this.bend(knee,-10*up);}
  }
 }
 /** After humanoid.update, before the springs: holds S on its anchor while held; once released, eases from that hold to planting the
  * feet on the line they stood on (smootherstep over 0.35 s, so the translation keeps its speed and acceleration: no pop), plants them
  * until settings.recoverySeconds, then settles to 0. Then the camera safety net. */
 anchor(settings){
  if(!this.active)return;
  const root=this.vrm.scene,hips=this.vrm.humanoid.getRawBoneNode?.('hips'),moved=hips?.parent?hips:root;  // without raw hips (a bare fixture), the root
  this.saved={node:moved,position:moved.position.clone()};
  root.updateMatrixWorld(true);this.previous.copy(this.offset);
  const anchored=this.anchored.copy(this.anchorPoint).sub(this.bone.getWorldPosition(this.v));
  if(this.held)this.offset.copy(anchored);
  else if(this.releaseAge<settings.recoverySeconds){const toe=this.toeY(),clear=Math.min(.15,.5*Math.max(0,this.aside()-(this.restAside||0)));this.offset.set(0,toe===null||this.restToe===null?0:this.restToe-toe+clear,0).lerp(anchored,1-smoother(this.releaseAge/LETGO));}  // feet still swinging under her stay clear of the floor until they are below her
  else for(const axis of ['x','y','z'])[this.offset[axis],this.offsetVelocity[axis]]=settle(this.offset[axis],this.offsetVelocity[axis],0,SETTLE,this.h);
  if(this.held||this.releaseAge<settings.recoverySeconds)this.offsetVelocity.copy(this.offset).sub(this.previous).divideScalar(Math.max(1e-4,this.h)).clampLength(0,3);
  // The world offset in the moved node's parent space: S moves rigidly with the hips, so this places it exactly.
  const inverse=this.m.copy(moved.parent?.matrixWorld||new Matrix4()).invert(),at=moved.getWorldPosition(this.a);
  moved.position.add(this.b.copy(at).add(this.offset).applyMatrix4(inverse).sub(at.applyMatrix4(inverse)));root.updateMatrixWorld(true);
  this.frame();
  if(!this.held&&this.releaseAge>=settings.recoverySeconds&&Math.abs(this.theta)<.2*D&&Math.abs(this.omega)<1*D&&[this.legs,this.arms,this.head,this.lift].every(s=>s.still)&&this.offset.length()<5e-4&&this.offsetVelocity.length()<1e-3&&!this.cameraActive)this.finish();
 }
 /** Safety net: zoom out about S (its NDC unchanged) when the swinging skeleton would leave the canvas; ease back once it fits. */
 frame(){
  const camera=this.camera;if(!camera||!this.cameraRest||!this.anchorNDC)return;
  const rest=this.cameraRest.position.z-this.anchorPoint.z,tan=Math.tan(camera.fov*D/2),tx=tan*camera.aspect,x=clamp(this.anchorNDC.x,-.95,.95),y=clamp(this.anchorNDC.y,-.95,.95),p=this.anchorPoint;
  let desired=rest;
  if(this.held){
   const bounds=this.bounds.makeEmpty();for(const e of this.catalog.entries)bounds.expandByPoint(e.node.getWorldPosition(this.v));bounds.expandByScalar(MARGIN);
   desired=Math.max(rest,(bounds.max.x-p.x)/(tx*(1-x)),(p.x-bounds.min.x)/(tx*(1+x)),(bounds.max.y-p.y)/(tan*(1-y)),(p.y-bounds.min.y)/(tan*(1+y)));
  }
  if(!this.cameraActive&&desired<=rest+1e-6)return;
  if(desired>this.distance){this.distance=desired;this.distanceVelocity=0;}else[this.distance,this.distanceVelocity]=settle(this.distance,this.distanceVelocity||0,desired,1.678/.3,this.h);  // out at once, back critically damped (0.3 s half-life)
  if(this.distance-rest<1e-4){this.distance=rest;this.distanceVelocity=0;this.cameraActive=false;camera.position.copy(this.cameraRest.position);camera.quaternion.copy(this.cameraRest.quaternion);camera.updateMatrixWorld(true);return;}
  this.cameraActive=true;const d=Math.max(.1,this.distance);
  camera.position.set(p.x-x*d*tx,p.y-y*d*tan,p.z+d);camera.quaternion.copy(this.cameraRest.quaternion);camera.updateMatrixWorld(true);
 }
 finish(){this.active=false;this.held=false;this.ragdoll=false;this.offset.set(0,0,0);this.offsetVelocity.set(0,0,0);this.theta=this.omega=this.displayed=0;for(const spring of [this.legs,this.arms,this.head,this.lift])spring.reset();}
 /** Pickup switched off mid-hold: everything back at once. */
 stop(){if(this.ragdoll)this.onTrigger('recover');this.restore();this.finish();if(this.camera&&this.cameraRest&&this.cameraActive){this.camera.position.copy(this.cameraRest.position);this.camera.quaternion.copy(this.cameraRest.quaternion);this.camera.updateMatrixWorld(true);}this.cameraActive=false;}
}
