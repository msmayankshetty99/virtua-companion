/** The held behaviour: being picked up is a short scene, not a steady wobble. A press pops her brows and blinks; a grab startles
 * her into surprise; held still she dangles, breathes and glances up at whoever holds her; carried fast she looks alarmed; shaken she
 * gets dizzy; after protestSeconds still she pouts and kicks in bursts, after resignSeconds she sighs and goes limp; a hair tug makes
 * her wince. Sources: VPet's raise loop (a move over 20 px restarts the struggle), Shimeji's resist bursts (2.5 Hz with pauses), Mate
 * Engine's 0.3 s drag lock and breathing mouth, startle-then-emotion timing.
 * Stepped once per animation frame with performance.now() in seconds; window velocity (px/s) and acceleration (px/s^2) come from
 * AvatarInertia. Pure: no three.js, no timers. Segment C adds the flight (falling, landing) behind release(); here every release
 * sets her down in place with a short settle. */
export const HELD_STATES=new Set(['grabbed','dangling','carried','dizzy','protesting','resigned']);
const POSITIVE=new Set(['joy','love','amusement','excitement']);
const T={startle:.12,grab:.35,surprise:.9,holder:.15,poke:.6,carryExit:.6,dizzyExit:1.2,annoyed:3,perk:.4,sigh:3,sighMouth:1,relief:1.8,annoyedLook:1.2,settleRise:.18};
const FAST=700,FAST_ACCEL=9000,SLOW=300,SHAKE=1200,SHAKE_WINDOW=1.5,SHAKES=4,MOVE=20,MIRROR=200,MIRROR_TIME=.4,MIN_PROTEST=3,SET_DOWN=.4;  // px/s, px/s^2, px; s; settle squash
const BREATH={grabbed:[2.2,0],dangling:[2.2,1.5],carried:[1.6,1.5],protesting:[1.4,1.8],dizzy:[2,1.2],resigned:[2.8,1.5],recovering:[2,1.5]};  // period (s), chest depth (degrees)
const FACE={dangling:'worried',carried:'alarm',dizzy:'dizzy',protesting:'pout',resigned:'resigned',tugged:'wince'};
const TAU=2*Math.PI;
const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,Number.isFinite(v)?v:0));
const smooth=u=>{u=clamp(u);return u*u*(3-2*u);};
const wrap=phase=>phase%TAU;

export class HoldBehaviour{
 constructor({random=Math.random}={}){this.random=random;this.breathPhase=0;this.breathDepth=0;this.kickPhase=0;this.kickRamp=0;this.cancel();}
 /** Back to no reaction at once, with no events (the model unloads, or reactions were switched off). */
 cancel(){this.state=null;this.since=0;this.transition=.25;this.prior=null;this.variant='tuck';this.mirror=this.mirrorTarget=this.mirrorEase=1;this.kickRamp=0;this.squash=0;this.squashVelocity=0;this.pending={events:[],blinks:[],suppress:0};}
 enter(state,now){
  const from=this.state;if(from===state)return;
  this.transition=state==='recovering'?.35:HELD_STATES.has(state)&&HELD_STATES.has(from)?.4:.25;
  this.state=state;this.since=now;this.pending.events.push({from,to:state,at:now});
  if(state==='protesting'){this.burstStart=now;this.burstEnd=now+.9+.6*this.random();this.nextBurst=this.burstEnd+2+2*this.random();}
  if(state==='resigned')this.sighUntil=now+T.sigh;
  if(state==='dangling'||state==='protesting'){this.glanceUntil=0;this.nextGlance=now+3+3*this.random();}
 }
 /** Pointer down on her body: a brow pop and a blink. */
 press(now){this.pressAt=now;this.enter('pressed',now);this.pending.blinks.push({kind:'press',delay:.03});}
 /** Released before the drag threshold: the brow raise holds 0.6 s (the renderer plays the 'clicked' pose). */
 click(now){this.enter('poked',now);}
 /** The drag threshold was crossed. kind 'spring' (hair or bust) is a tug, face only; 'body' picks her up. */
 grab(now,{region=null,kind='body',emotion=null}={}){
  this.grabAt=now;this.region=region;this.kind=kind;
  if(kind==='spring'){this.enter('tugged',now);return;}
  this.variant=POSITIVE.has(emotion?.primary)?'swing':this.random()<.7?'tuck':'reach';
  this.prior=null;this.mirror=this.mirrorTarget=this.mirrorEase=1;this.position={x:0,y:0};this.still={x:0,y:0};this.stillSince=now;this.reversals=[];this.shakeSign=0;this.lastShake=-Infinity;this.slowSince=null;this.annoyedUntil=0;this.perkUntil=0;
  this.enter('grabbed',now);this.pending.suppress=Math.max(this.pending.suppress,T.surprise);
 }
 /** Pointer up after a drag. Returns the next state. Options {vx, vy, mode, throwSpeed} are for segment C's flight; for now a held
  * body always settles in place ('recovering'), and a tug or press just ends. */
 release(now,options={}){
  if(this.state==='pressed'||this.state==='tugged')this.enter(null,now);
  if(!HELD_STATES.has(this.state))return this.state;
  this.prior=this.state;this.squash=0;this.squashVelocity=0;
  this.enter('recovering',now);this.pending.blinks.push({kind:'relief',delay:.05});
  return this.state;
 }
 /** Held: tracks moves, speed and shaking, and changes state. */
 track(now,h,{vx=0,vy=0,ax=0,ay=0,ragdoll=false},settings){
  const speed=Math.hypot(vx,vy),fast=speed>FAST||Math.hypot(ax,ay)>FAST_ACCEL;
  this.position.x+=vx*h;this.position.y+=vy*h;
  const moved=Math.hypot(this.position.x-this.still.x,this.position.y-this.still.y)>MOVE;
  if(moved){this.still={...this.position};this.stillSince=now;}
  if(Math.abs(vx)>MIRROR)this.mirrorTarget=Math.sign(vx);  // latched: nothing between -200 and 200 px/s flips it; step() eases toward it
  if(Math.abs(vx)>SHAKE){const sign=Math.sign(vx);if(this.shakeSign&&sign!==this.shakeSign)this.reversals.push(now);if(this.shakeSign&&sign!==this.shakeSign||this.state==='dizzy')this.lastShake=now;this.shakeSign=sign;}  // dizzy lasts while she is still moved hard
  while(this.reversals.length&&now-this.reversals[0]>SHAKE_WINDOW)this.reversals.shift();
  if(speed<SLOW)this.slowSince??=now;else this.slowSince=null;
  if(ragdoll)this.lastShake=now;
  const state=this.state,age=now-this.since,still=now-this.stillSince;
  if(state!=='dizzy'&&(ragdoll||this.reversals.length>=SHAKES)){this.enter('dizzy',now);return;}
  if(state==='grabbed'){if(age>=T.grab-1e-9)this.enter(fast?'carried':'dangling',now);}
  else if(state==='dizzy'){if(now-this.lastShake>=T.dizzyExit&&age>=T.dizzyExit){this.reversals=[];this.annoyedUntil=now+T.annoyed;this.enter('dangling',now);}}
  else if(state==='carried'){if(this.slowSince!==null&&now-Math.max(this.slowSince,this.since)>=T.carryExit-1e-9)this.enter('dangling',now);}
  else if(state==='dangling'){if(fast)this.enter('carried',now);else if(still>=settings.protestSeconds)this.enter('protesting',now);}
  else if(state==='protesting'){if(moved)this.enter(fast?'carried':'dangling',now);else if(still>=Math.max(settings.resignSeconds,settings.protestSeconds+MIN_PROTEST))this.enter('resigned',now);}  // she always protests a while first
  else if(state==='resigned'&&moved){this.perkUntil=now+T.perk;this.enter('carried',now);}
 }
 faceIntent(now){
  const state=this.state;
  if(state==='pressed')return now-this.pressAt<T.poke?'poke':null;
  if(state==='poked')return 'poke';
  if(state==='recovering'){const age=now-this.since;return this.prior==='protesting'||this.prior==='dizzy'?(age<T.annoyedLook?'annoyed':null):age<T.relief?'relief':null;}
  if(HELD_STATES.has(state)){const g=now-this.grabAt;
   if(g<T.startle)return 'startle';if(g<T.surprise)return 'surprise';
   if(state==='carried'&&now<this.perkUntil)return 'perk';
   if(state==='resigned'&&now-this.since<T.sighMouth)return 'sigh';
   if(state==='dangling'&&now<this.annoyedUntil)return 'annoyed';
  }
  return FACE[state]||null;
 }
 /** Where she looks: 'viewer', 'holder', 'floor', 'cursor' or null (normal tracking). Segment C's gaze consumes it. */
 look(now){
  const state=this.state,g=now-this.grabAt;
  if(HELD_STATES.has(state)&&state!=='dizzy'&&g<T.surprise)return g>=T.holder?'holder':'viewer';
  if(state==='dangling'||state==='protesting')return now<this.glanceUntil?'holder':'viewer';
  if(state==='recovering')return (this.prior==='protesting'||this.prior==='dizzy')&&now-this.since<T.annoyedLook?'cursor':null;
  return {carried:'viewer',resigned:'floor',tugged:'cursor'}[state]||null;
 }
 /** steps: now (s), dt (s), window motion {vx, vy, ax, ay} from AvatarInertia, ragdoll (the pickup's opt-in limp trigger), settings
  * {enabled, protestSeconds, resignSeconds, recoverySeconds}. Returns null when there is nothing to show, else {state, age, transition
  * (pose crossfade seconds), pose (null for face-only states), breath {rate, depth, phase}, face {intent, intensity}, look, blinks
  * [{kind, delay}], suppress (s of no spontaneous blinks), events [{from, to, at}]}. The frame a reaction ends still reports it. */
 step(now,dt,motion={},settings={}){
  if(settings.enabled===false){if(this.state)this.cancel();return null;}
  const h=clamp(dt,0,.1),before=this.state;
  if(!this.state&&!this.pending.events.length)return null;
  if(this.state==='poked'&&now-this.since>=T.poke)this.enter(null,now);
  if(HELD_STATES.has(this.state))this.track(now,h,motion,{protestSeconds:6,resignSeconds:18,...settings});
  if(this.state==='recovering'){
   const age=now-this.since;
   if(age<T.settleRise){this.squash=SET_DOWN*smooth(age/T.settleRise);this.squashVelocity=0;}
   else{const w=TAU*2.5,z=.6;this.squashVelocity+=(-w*w*this.squash-2*z*w*this.squashVelocity)*h;this.squash+=this.squashVelocity*h;}  // about 10% overshoot into a small stretch
   if(age>=(settings.recoverySeconds??2.5))this.enter(null,now);
  }
  const state=this.state;
  if(state==='protesting'&&now>=this.nextBurst){this.burstStart=this.nextBurst;this.burstEnd=this.burstStart+.9+.6*this.random();this.nextBurst=this.burstEnd+2+2*this.random();}
  const burst=state==='protesting'&&now>=this.burstStart&&now<this.burstEnd;
  this.kickRamp=clamp(this.kickRamp+(burst?h/.15:-h/.3));
  this.mirrorEase=clamp(this.mirrorEase+Math.sign(this.mirrorTarget-this.mirrorEase)*Math.min(Math.abs(this.mirrorTarget-this.mirrorEase),2*h/MIRROR_TIME),-1,1);this.mirror=2*smooth((this.mirrorEase+1)/2)-1;  // a change of drag direction crossfades the legs over 0.4 s
  const noise=.6*Math.sin(1.7*now)+.4*Math.sin(2.9*now+1);this.kickPhase=wrap(this.kickPhase+TAU*2.4*(1+.2*noise)*h);
  if((state==='dangling'||state==='protesting')&&now>=this.nextGlance){this.glanceUntil=now+.6+.4*this.random();this.nextGlance=this.glanceUntil+3+3*this.random();}
  const sigh=state==='resigned'&&now<this.sighUntil,[period,depth]=sigh?[3,BREATH.resigned[1]*2]:BREATH[state]||[2.2,0];
  this.breathDepth+=(depth-this.breathDepth)*(1-Math.exp(-h/.3));this.breathPhase=wrap(this.breathPhase+TAU*h/period);
  const out={state,age:state?now-this.since:0,transition:this.transition,pose:this.pose(now),breath:{rate:1/period,depth:this.breathDepth,phase:this.breathPhase},
   face:{intent:state?this.faceIntent(now):null,intensity:1},look:state?this.look(now):null,blinks:this.pending.blinks,suppress:this.pending.suppress,events:this.pending.events};
  this.pending={events:[],blinks:[],suppress:0};
  return state||before||out.events.length?out:null;
 }
 /** The body's part: variant ('tuck', 'reach' or 'swing'), startle, alarm, limp, kick and kickPhase, sway, squash (the settle), mirror
  * (1 to -1, easing toward the sign of the drag so the more bent knee trails). null for the face-only states. */
 pose(now){
  const state=this.state;if(!HELD_STATES.has(state)&&state!=='recovering')return null;
  const g=now-this.grabAt,startle=HELD_STATES.has(state)&&g<T.grab?(g<T.startle?smooth(g/T.startle):smooth(1-(g-T.startle)/(T.grab-T.startle))):0;
  return {variant:this.variant,startle,alarm:state==='carried'?1:0,limp:state==='dizzy'||state==='resigned'?1:0,kick:smooth(this.kickRamp),kickPhase:this.kickPhase,
   sway:state==='grabbed'||state==='dangling'||state==='protesting'?1:0,squash:state==='recovering'?this.squash:0,mirror:this.mirror};
 }
}

/** A stand-in for the held pose when reactions are off: the plain dangle, limp while the opt-in ragdoll is active. Shared, read-only. */
const PASSIVE=[false,true].map(ragdoll=>({state:ragdoll?'dizzy':'dangling',transition:.25,pose:{variant:'tuck',startle:0,alarm:0,limp:ragdoll?1:0,kick:0,kickPhase:0,sway:0,squash:0,mirror:1},breath:{rate:0,depth:0,phase:0},face:{intent:null,intensity:1},look:null,blinks:[],suppress:0,events:[]}));
export const passiveHold=(ragdoll=false)=>PASSIVE[ragdoll?1:0];

/** Applies a step's face part to an AvatarFace: react on a change of intent, forced blinks, and suppressed spontaneous blinks. */
export class HoldFace{
 constructor(face){this.face=face;this.intent=null;this.intensity=1;}
 apply(out){
  const face=this.face;if(!face)return;
  const intent=out?.face?.intent||null,intensity=out?.face?.intensity??1;
  if(intent!==this.intent||intensity!==this.intensity){face.react(intent,{intensity});this.intent=intent;this.intensity=intensity;}
  for(const blink of out?.blinks||[])face.blink(blink.kind,blink.delay);
  if(out?.suppress>0)face.blinker.suppress(out.suppress);
 }
}
